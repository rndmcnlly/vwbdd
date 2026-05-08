//! Native `.vwbdd` dump/load format.
//!
//! The arena is already a serialized byte stream; dump wraps it with a
//! small header describing format version, offset width, and root list.
//! Load parses the header, mmap's (well, reads) the arena bytes back into
//! a fresh `Manager`, walks once to rebuild the unique table, and returns
//! the roots.
//!
//! Also ships `absorb`, which takes a dumped file and merges its contents
//! into an existing manager, re-interning every node through the unique
//! table so any subgraphs that already existed in the receiver's arena
//! dedupe automatically. This is the multi-process merge primitive
//! referenced in the session notes: workers dump their partitions, the
//! parent absorbs each and ORs the roots.
//!
//! ## Format (v1)
//!
//! Everything little-endian. See [`Header`] for the fixed-size prefix.
//!
//! ```text
//! [32 B header]
//! [arena_len bytes of raw codec-encoded node stream]
//! [num_roots × u64 of root refs (ref_to_wire encoding)]
//! [optional: length-prefixed UTF-8 root names]
//! [4 B CRC32 of everything above]
//! ```
//!
//! Root refs are encoded via [`ref_to_wire`]:
//!   - `0` → `Terminal(false)`
//!   - `1` → `Terminal(true)`
//!   - `2 + offset` → `Node(offset)` (absolute offset, not a delta)
//!
//! The header's `offset_width` field declares whether the arena was built
//! with `u32` or `u64` offsets. A `u32` dump can load into either width;
//! a `u64` dump cannot load into a `u32` engine if any offset exceeds
//! `u32::MAX` (checked at load).

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::codec::{ArenaOffset, Leb128Codec, NodeCodec, Ref};
use crate::manager::Manager;
use crate::unique::unique_key_hash;

const MAGIC: [u8; 8] = *b"VWBDD\0\0\0";
const FORMAT_VERSION: u16 = 1;
const FLAG_NAMED_ROOTS: u8 = 0b0000_0001;

/// Fixed 32-byte dump header.
#[derive(Debug, Clone, Copy)]
struct Header {
    // 0..8:   magic (VWBDD\0\0\0)
    // 8..10:  format_version (u16 LE)
    // 10:     offset_width (u8; 4 for u32, 8 for u64)
    // 11:     flags (u8; bit 0: has_root_names)
    // 12..16: num_vars (u32 LE)
    // 16..20: num_roots (u32 LE)
    // 20..28: arena_len (u64 LE)
    // 28..32: reserved (u32 LE = 0)
    format_version: u16,
    offset_width: u8,
    flags: u8,
    num_vars: u32,
    num_roots: u32,
    arena_len: u64,
}

impl Header {
    const SIZE: usize = 32;

    fn write_to(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&self.format_version.to_le_bytes());
        out.push(self.offset_width);
        out.push(self.flags);
        out.extend_from_slice(&self.num_vars.to_le_bytes());
        out.extend_from_slice(&self.num_roots.to_le_bytes());
        out.extend_from_slice(&self.arena_len.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes()); // reserved
    }

    fn parse(buf: &[u8]) -> Result<Self, DumpError> {
        if buf.len() < Self::SIZE {
            return Err(DumpError::Truncated);
        }
        if buf[0..8] != MAGIC {
            return Err(DumpError::BadMagic);
        }
        let format_version = u16::from_le_bytes([buf[8], buf[9]]);
        if format_version != FORMAT_VERSION {
            return Err(DumpError::UnsupportedVersion(format_version));
        }
        let offset_width = buf[10];
        let flags = buf[11];
        let num_vars = u32::from_le_bytes([buf[12], buf[13], buf[14], buf[15]]);
        let num_roots = u32::from_le_bytes([buf[16], buf[17], buf[18], buf[19]]);
        let arena_len = u64::from_le_bytes([
            buf[20], buf[21], buf[22], buf[23], buf[24], buf[25], buf[26], buf[27],
        ]);
        // buf[28..32] is reserved; ignored.
        Ok(Header {
            format_version,
            offset_width,
            flags,
            num_vars,
            num_roots,
            arena_len,
        })
    }
}

/// Encode a root `Ref<O>` as a u64 for on-disk storage.
/// Reuses the child-code convention (0=F, 1=T, 2+off=Node) but absolute
/// instead of delta. At u32 arena width, `off` fits in 32 bits and the
/// u64 wire encoding simply zero-extends.
#[inline]
fn ref_to_wire<O: ArenaOffset>(r: Ref<O>) -> u64 {
    match r {
        Ref::Terminal(false) => 0,
        Ref::Terminal(true) => 1,
        Ref::Node(off) => 2 + off.to_u64(),
    }
}

/// Decode a root from wire form back to `Ref<O>`. Returns an error if
/// the encoded offset won't fit in the target `O`.
#[inline]
fn wire_to_ref<O: ArenaOffset>(code: u64) -> Result<Ref<O>, DumpError> {
    match code {
        0 => Ok(Ref::Terminal(false)),
        1 => Ok(Ref::Terminal(true)),
        c => {
            let off_u64 = c - 2;
            // Guard against loading a u64 dump into a u32 engine when
            // the arena exceeded the u32 ceiling.
            if off_u64 > O::MAX.to_u64() {
                return Err(DumpError::OffsetOverflowsTarget(off_u64));
            }
            Ok(Ref::Node(O::from_u64(off_u64)))
        }
    }
}

/// Errors arising during dump serialization or load parsing.
#[derive(Debug)]
pub enum DumpError {
    Io(std::io::Error),
    BadMagic,
    UnsupportedVersion(u16),
    Truncated,
    CrcMismatch { expected: u32, actual: u32 },
    /// A dumped offset (u64) exceeds the target engine's offset width.
    /// E.g., a `LargeManager` dump whose arena > 4 GiB being loaded into
    /// a `DefaultManager` (u32).
    OffsetOverflowsTarget(u64),
    /// Root-name block exists but doesn't parse cleanly.
    MalformedNames,
}

impl std::fmt::Display for DumpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {}", e),
            Self::BadMagic => write!(f, "not a vwbdd dump (magic bytes mismatch)"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported format version: {}", v),
            Self::Truncated => write!(f, "dump file truncated"),
            Self::CrcMismatch { expected, actual } => write!(
                f, "CRC32 mismatch: expected {:08x}, got {:08x}", expected, actual
            ),
            Self::OffsetOverflowsTarget(off) => write!(
                f, "offset {} exceeds target engine's width", off
            ),
            Self::MalformedNames => write!(f, "root-name block malformed"),
        }
    }
}

impl std::error::Error for DumpError {}

impl From<std::io::Error> for DumpError {
    fn from(e: std::io::Error) -> Self {
        DumpError::Io(e)
    }
}

/// CRC32 (IEEE polynomial 0xedb88320). 30-line implementation to avoid
/// pulling in a crate dependency. Not the fastest (no slicing-by-8, no
/// hardware intrinsics), but serialization is bounded by disk I/O anyway.
fn crc32(bytes: &[u8]) -> u32 {
    static TABLE: std::sync::OnceLock<[u32; 256]> = std::sync::OnceLock::new();
    let table = TABLE.get_or_init(|| {
        let mut t = [0u32; 256];
        for i in 0..256 {
            let mut c = i as u32;
            for _ in 0..8 {
                c = if c & 1 != 0 { 0xedb88320 ^ (c >> 1) } else { c >> 1 };
            }
            t[i] = c;
        }
        t
    });
    let mut crc = 0xffff_ffffu32;
    for &b in bytes {
        crc = table[((crc ^ b as u32) & 0xff) as usize] ^ (crc >> 8);
    }
    crc ^ 0xffff_ffff
}

// --- Implementation surface on Manager ---

impl<C: NodeCodec<O>, O: ArenaOffset> Manager<C, O> {
    /// Dump the manager's reachable-from-roots subset to a file in the
    /// native vwbdd binary format. Named roots are not written.
    ///
    /// **Clean-bytes invariant**: the dumped file is function-canonical
    /// for the declared roots. Internally runs `drop_roots(roots)`
    /// before writing so the file contains no scratch. This mutates
    /// the manager (same side effects as `drop_roots`); clone first if
    /// you need to preserve state.
    ///
    /// Prior versions of this method dumped the entire arena verbatim
    /// and asked the caller to `gc` first for compactness. That
    /// exposed dirty bytes across the public (file) boundary and is
    /// no longer done: all publicly-observable vwbdd artifacts are
    /// clean by construction.
    pub fn dump(&mut self, path: impl AsRef<Path>, roots: &[Ref<O>]) -> Result<(), DumpError> {
        let cleaned = self.drop_roots(roots);
        self.dump_inner(path.as_ref(), &cleaned, None)
    }

    /// Dump with per-root names. Names are UTF-8 strings; the loader
    /// returns them in the same order as the root list. Useful for
    /// transition-system artifacts that want to preserve `init`, `trans`,
    /// etc. as first-class labels across dump/load.
    ///
    /// Same clean-bytes invariant as [`Self::dump`].
    pub fn dump_named<S: AsRef<str>>(
        &mut self,
        path: impl AsRef<Path>,
        roots_and_names: &[(Ref<O>, S)],
    ) -> Result<(), DumpError> {
        let roots: Vec<Ref<O>> = roots_and_names.iter().map(|(r, _)| *r).collect();
        let names: Vec<&str> = roots_and_names.iter().map(|(_, n)| n.as_ref()).collect();
        let cleaned = self.drop_roots(&roots);
        self.dump_inner(path.as_ref(), &cleaned, Some(&names))
    }

    fn dump_inner(
        &self,
        path: &Path,
        roots: &[Ref<O>],
        names: Option<&[&str]>,
    ) -> Result<(), DumpError> {
        let arena = self.arena_bytes();
        let mut buf = Vec::with_capacity(Header::SIZE + arena.len() + roots.len() * 8 + 4);

        // Header.
        let header = Header {
            format_version: FORMAT_VERSION,
            offset_width: std::mem::size_of::<O>() as u8,
            flags: if names.is_some() { FLAG_NAMED_ROOTS } else { 0 },
            num_vars: self.num_vars(),
            num_roots: roots.len() as u32,
            arena_len: arena.len() as u64,
        };
        header.write_to(&mut buf);

        // Arena bytes (raw).
        buf.extend_from_slice(arena);

        // Root references.
        for &r in roots {
            buf.extend_from_slice(&ref_to_wire::<O>(r).to_le_bytes());
        }

        // Optional root names.
        if let Some(names) = names {
            for name in names {
                let bytes = name.as_bytes();
                assert!(
                    bytes.len() <= u16::MAX as usize,
                    "root name too long (> 64 KiB)"
                );
                buf.extend_from_slice(&(bytes.len() as u16).to_le_bytes());
                buf.extend_from_slice(bytes);
            }
        }

        // CRC32 of everything above.
        let crc = crc32(&buf);
        buf.extend_from_slice(&crc.to_le_bytes());

        // Single write pass for atomicity (modulo filesystem semantics).
        let mut out = BufWriter::new(File::create(path)?);
        out.write_all(&buf)?;
        out.flush()?;
        Ok(())
    }

    /// Load a dumped manager from a file, constructing a fresh engine.
    /// Uses [`ManagerConfig::default()`] for cache sizing; for custom
    /// config use [`Manager::load_with_config`].
    ///
    /// Generic over the target `C, O`. For the common case where you want
    /// the default engine (`Leb128Codec`, `u32`), prefer the inherent
    /// `Manager::load(path)` which lets type inference pick these for you.
    pub fn load_generic(path: impl AsRef<Path>) -> Result<(Self, LoadedRoots<O>), DumpError>
    where
        Self: Sized,
    {
        Self::load_with_config(path, crate::manager::ManagerConfig::default())
    }

    /// Load a dumped manager, with an explicit config.
    pub fn load_with_config(
        path: impl AsRef<Path>,
        config: crate::manager::ManagerConfig,
    ) -> Result<(Self, LoadedRoots<O>), DumpError> {
        let buf = read_and_verify(path.as_ref())?;
        let (header, payload) = split_header(&buf)?;
        let (arena, roots_raw, names_raw) = split_payload(&header, payload)?;

        // Build a fresh manager and reconstruct its arena + unique table.
        let mut mgr = Self::with_config(config);
        for _ in 0..header.num_vars {
            mgr.new_var();
        }
        mgr.absorb_arena_bytes(arena);

        // Decode roots. Since arena offsets in the dump are absolute and
        // we wrote the arena verbatim into the fresh engine, roots are
        // valid as-is (modulo offset-width fitting, checked in wire_to_ref).
        let mut roots: Vec<Ref<O>> = Vec::with_capacity(header.num_roots as usize);
        for i in 0..(header.num_roots as usize) {
            let code = u64::from_le_bytes(
                roots_raw[i * 8..i * 8 + 8]
                    .try_into()
                    .expect("8-byte slice"),
            );
            roots.push(wire_to_ref::<O>(code)?);
        }

        let names = if header.flags & FLAG_NAMED_ROOTS != 0 {
            Some(parse_names(names_raw, header.num_roots as usize)?)
        } else {
            None
        };

        Ok((mgr, LoadedRoots { roots, names }))
    }

    /// Absorb a dumped file into this manager, de-duplicating through the
    /// unique table. Returns the translated roots (offsets now point into
    /// this manager's arena).
    ///
    /// This is the multi-process merge primitive: each worker dumps its
    /// partition, the parent absorbs them one by one. Any subgraph that
    /// appears in multiple workers' arenas collapses to a single node in
    /// the parent through the unique table's canonicalization.
    ///
    /// Variable declarations in the absorbed file must be a prefix of
    /// this manager's declarations (same var numbers mean the same
    /// variables). We check `num_vars <= self.num_vars()` but not
    /// semantic equality; the caller's responsibility.
    pub fn absorb(&mut self, path: impl AsRef<Path>) -> Result<Vec<Ref<O>>, DumpError> {
        let buf = read_and_verify(path.as_ref())?;
        let (header, payload) = split_header(&buf)?;
        let (arena, roots_raw, _names_raw) = split_payload(&header, payload)?;

        assert!(
            (header.num_vars as u32) <= self.num_vars(),
            "absorbed dump uses {} vars but this manager only has {}",
            header.num_vars, self.num_vars()
        );

        // Walk the foreign arena in construction order, decoding each
        // node and re-interning in self. Build a translation map from
        // foreign-offset to this-manager's Ref.
        let mut translation: std::collections::HashMap<u64, Ref<O>> =
            std::collections::HashMap::new();
        let mut pos: usize = 0;
        while pos < arena.len() {
            let foreign_off_u64 = pos as u64;
            let foreign_off = O::from_u64(foreign_off_u64);
            // Decode using a *fresh* buffer view at the foreign offset.
            let (node, consumed) = C::decode(&arena[pos..], foreign_off);
            pos += consumed;

            // Translate children: each child Ref in the decoded node
            // still holds *foreign* offsets. Rewrite them through our
            // translation map (terminals pass through).
            let lo = translate_child::<O>(node.lo, &translation);
            let hi = translate_child::<O>(node.hi, &translation);

            // Re-intern. `make_node` handles canonicalization; if this
            // exact (var, lo, hi) already lives in our arena, we get the
            // existing offset back (that's the dedup).
            let new_ref = self.make_node(node.var, lo, hi);
            translation.insert(foreign_off_u64, new_ref);
        }

        // Translate roots through the map.
        let mut roots = Vec::with_capacity(header.num_roots as usize);
        for i in 0..(header.num_roots as usize) {
            let code = u64::from_le_bytes(
                roots_raw[i * 8..i * 8 + 8]
                    .try_into()
                    .expect("8-byte slice"),
            );
            let foreign_ref = wire_to_ref::<O>(code)?;
            let translated = translate_child::<O>(foreign_ref, &translation);
            roots.push(translated);
        }
        Ok(roots)
    }

    // --- internal helpers exposed only to this module ---

    fn arena_bytes(&self) -> &[u8] {
        self.arena_slice(0)
    }

    /// Append raw arena bytes verbatim (used by `load` — the dump's
    /// arena is already canonical and re-hashable). Also rebuilds the
    /// unique table over the full new arena.
    fn absorb_arena_bytes(&mut self, bytes: &[u8]) {
        self.buf_mut().extend_from_slice(bytes);
        self.rebuild_unique_from_arena();
    }
}

/// Roots and optional names returned from a successful [`Manager::load`].
#[derive(Debug, Clone)]
pub struct LoadedRoots<O: ArenaOffset> {
    pub roots: Vec<Ref<O>>,
    pub names: Option<Vec<String>>,
}

#[inline]
fn translate_child<O: ArenaOffset>(
    r: Ref<O>,
    translation: &std::collections::HashMap<u64, Ref<O>>,
) -> Ref<O> {
    match r {
        Ref::Terminal(_) => r,
        Ref::Node(off) => *translation
            .get(&off.to_u64())
            .expect("foreign child offset not yet translated (DAG order broken?)"),
    }
}

/// Read a dump file and verify its CRC32. Returns the whole buffer
/// including the trailing CRC (callers should work with the payload via
/// [`split_header`] + [`split_payload`]).
fn read_and_verify(path: &Path) -> Result<Vec<u8>, DumpError> {
    let f = File::open(path)?;
    let mut reader = BufReader::new(f);
    let mut buf = Vec::new();
    reader.read_to_end(&mut buf)?;
    if buf.len() < 4 {
        return Err(DumpError::Truncated);
    }
    let (body, trailer) = buf.split_at(buf.len() - 4);
    let expected = u32::from_le_bytes([trailer[0], trailer[1], trailer[2], trailer[3]]);
    let actual = crc32(body);
    if expected != actual {
        return Err(DumpError::CrcMismatch { expected, actual });
    }
    buf.truncate(buf.len() - 4);
    Ok(buf)
}

fn split_header(buf: &[u8]) -> Result<(Header, &[u8]), DumpError> {
    if buf.len() < Header::SIZE {
        return Err(DumpError::Truncated);
    }
    let header = Header::parse(&buf[..Header::SIZE])?;
    Ok((header, &buf[Header::SIZE..]))
}

fn split_payload<'a>(
    header: &Header,
    payload: &'a [u8],
) -> Result<(&'a [u8], &'a [u8], &'a [u8]), DumpError> {
    let arena_len = header.arena_len as usize;
    let roots_len = (header.num_roots as usize) * 8;
    if payload.len() < arena_len + roots_len {
        return Err(DumpError::Truncated);
    }
    let arena = &payload[..arena_len];
    let roots_raw = &payload[arena_len..arena_len + roots_len];
    let names_raw = &payload[arena_len + roots_len..];
    Ok((arena, roots_raw, names_raw))
}

fn parse_names(buf: &[u8], n: usize) -> Result<Vec<String>, DumpError> {
    let mut out = Vec::with_capacity(n);
    let mut pos = 0;
    for _ in 0..n {
        if pos + 2 > buf.len() {
            return Err(DumpError::MalformedNames);
        }
        let len = u16::from_le_bytes([buf[pos], buf[pos + 1]]) as usize;
        pos += 2;
        if pos + len > buf.len() {
            return Err(DumpError::MalformedNames);
        }
        let name =
            std::str::from_utf8(&buf[pos..pos + len]).map_err(|_| DumpError::MalformedNames)?;
        out.push(name.to_owned());
        pos += len;
    }
    Ok(out)
}

// Compile-time reminder that the Leb128 codec is the only one we know how
// to dump today. (Future codec impls would need their own magic or a code
// byte in the header.)
const _: () = {
    // If we ever ship another NodeCodec, turning this on at compile time
    // would be a good checkpoint moment.
    const _DUMP_SUPPORTED_CODECS: &[&str] = &[<Leb128Codec as NodeCodec<u32>>::NAME];
};

// Silence unused-import in a single-codec build.
#[allow(dead_code)]
fn _silence_unused_hash<O: ArenaOffset>(var: u32, lo: Ref<O>, hi: Ref<O>) -> u64 {
    unique_key_hash::<O>(var, lo, hi)
}

// --- Inherent-on-default convenience wrappers ---
//
// The generic `impl<C, O> Manager<C, O>` block provides `dump`,
// `load_generic`, `load_with_config`, and `absorb`, but Rust's type
// inference can't pick `C, O` at a bare call site like `Manager::load(path)`.
// These inherent impls on the concrete default type make the unqualified
// call work (same pattern as `Manager::new()` in `manager.rs`). Callers
// wanting a non-default engine write `Manager::<Leb128Codec, u64>::load_generic`
// explicitly.
impl Manager<Leb128Codec, u32> {
    /// Default-engine load. Wrapper that pins `Leb128Codec` and `u32` so
    /// `Manager::load(path)` doesn't need turbofish.
    pub fn load(
        path: impl AsRef<Path>,
    ) -> Result<(Self, LoadedRoots<u32>), DumpError> {
        <Manager<Leb128Codec, u32>>::load_generic(path)
    }
}
