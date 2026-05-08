//! Native dump/load roundtrip tests.
//!
//! Covers: single-root, multi-root, named roots, the absorb merge primitive,
//! canonicity preservation, and error paths (magic, CRC, truncation).

use std::path::PathBuf;

use vwbdd::{DumpError, Manager, Ref};

fn tmp_path(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("vwbdd_test_{}_{}", std::process::id(), name));
    p
}

// Helper: build a small BDD and return a canonical `Ref`.
fn build_formula(m: &mut Manager) -> Ref {
    // (x0 ∧ x1) ∨ (¬x0 ∧ x2)
    let _ = m.new_var();
    let _ = m.new_var();
    let _ = m.new_var();
    let f = m.r#false();
    let t = m.r#true();
    let x0 = m.make_node(0, f, t);
    let x1 = m.make_node(1, f, t);
    let x2 = m.make_node(2, f, t);
    let a = m.and(x0, x1);
    let nx0 = m.not(x0);
    let b = m.and(nx0, x2);
    m.or(a, b)
}

fn build_formula_labeled(m: &mut Manager) -> (Ref, Ref, Ref) {
    // Variables as above.
    let _ = m.new_var();
    let _ = m.new_var();
    let _ = m.new_var();
    let f = m.r#false();
    let t = m.r#true();
    let x0 = m.make_node(0, f, t);
    let x1 = m.make_node(1, f, t);
    let x2 = m.make_node(2, f, t);
    let ab = m.and(x0, x1);
    let xy = m.or(x0, x2);
    let xyz = m.xor(x1, x2);
    (ab, xy, xyz)
}

#[test]
fn single_root_roundtrip() {
    let path = tmp_path("single");

    let mut m1 = Manager::new();
    let r = build_formula(&mut m1);
    let before_nodes = m1.num_nodes();
    let before_buf_len = m1.buf_len();

    // Separately, compute the reachable-from-root node count and the
    // byte size those nodes take. Under the clean-bytes invariant,
    // `dump` pre-GCs to the declared roots, so the dumped file holds
    // exactly this cleaned-down subset — not the full construction
    // history.
    let mut m_reference = Manager::new();
    let r_ref = build_formula(&mut m_reference);
    let _ = m_reference.drop_roots(&[r_ref]);
    let cleaned_nodes = m_reference.num_nodes();
    let cleaned_buf_len = m_reference.buf_len();
    assert!(
        cleaned_nodes <= before_nodes,
        "cleaned ({}) should be ≤ raw ({})",
        cleaned_nodes, before_nodes
    );

    m1.dump(&path, &[r]).expect("dump");

    let (m2, loaded) = Manager::load(&path).expect("load");
    assert_eq!(loaded.roots.len(), 1, "single root survived roundtrip");
    assert!(loaded.names.is_none(), "no names dumped → no names loaded");

    // Loaded manager holds the cleaned subset, not the full history.
    assert_eq!(
        m2.num_nodes(), cleaned_nodes,
        "dumped file contained the function-canonical subset: {} nodes (not the {} nodes \
         that existed in the manager's arena before dump)",
        cleaned_nodes, before_nodes
    );
    assert_eq!(m2.buf_len(), cleaned_buf_len);
    // m1 itself was also cleaned by dump (it ran drop_roots internally).
    assert_eq!(m1.num_nodes(), cleaned_nodes);
    assert_eq!(m1.buf_len(), cleaned_buf_len);
    let _ = before_buf_len; // retained for documentation

    // Root should decode as a Node ref pointing into the loaded arena.
    assert!(
        matches!(loaded.roots[0], Ref::Node(_)),
        "OR root is a non-terminal node"
    );

    std::fs::remove_file(&path).ok();
}

#[test]
fn multi_root_roundtrip() {
    let path = tmp_path("multi");

    let mut m1 = Manager::new();
    let (r1, r2, r3) = build_formula_labeled(&mut m1);

    m1.dump(&path, &[r1, r2, r3]).expect("dump");

    let (_m2, loaded) = Manager::load(&path).expect("load");
    assert_eq!(loaded.roots.len(), 3);
    assert!(loaded.names.is_none());

    std::fs::remove_file(&path).ok();
}

#[test]
fn named_roots_roundtrip() {
    let path = tmp_path("named");

    let mut m1 = Manager::new();
    let (r1, r2, r3) = build_formula_labeled(&mut m1);

    m1.dump_named(
        &path,
        &[(r1, "and"), (r2, "or"), (r3, "xor")],
    )
    .expect("dump named");

    let (_m2, loaded) = Manager::load(&path).expect("load");
    assert_eq!(loaded.roots.len(), 3);
    let names = loaded.names.expect("names block present");
    assert_eq!(names, vec!["and", "or", "xor"]);

    std::fs::remove_file(&path).ok();
}

#[test]
fn absorb_dedupes_shared_subgraphs() {
    // Two workers each build a formula. Both formulas share the subgraph
    // (x0 ∧ x1). When the parent absorbs both dumps, the unique table
    // should collapse the shared piece to a single node.
    let path_a = tmp_path("abs_a");
    let path_b = tmp_path("abs_b");

    // Worker A: dumps just (x0 ∧ x1).
    let mut wa = Manager::new();
    let (ra, _, _) = build_formula_labeled(&mut wa);
    wa.dump(&path_a, &[ra]).expect("dump a");

    // Worker B: dumps (x0 ∧ x1) ∨ x2 — shares (x0 ∧ x1) with A.
    let mut wb = Manager::new();
    let _v0 = wb.new_var();
    let _v1 = wb.new_var();
    let _v2 = wb.new_var();
    let f = wb.r#false();
    let t = wb.r#true();
    let x0 = wb.make_node(0, f, t);
    let x1 = wb.make_node(1, f, t);
    let x2 = wb.make_node(2, f, t);
    let ab = wb.and(x0, x1);
    let rb = wb.or(ab, x2);
    wb.dump(&path_b, &[rb]).expect("dump b");

    // Parent: fresh, declare matching variables, absorb both.
    let mut parent = Manager::new();
    let _ = parent.new_var();
    let _ = parent.new_var();
    let _ = parent.new_var();
    assert_eq!(parent.num_nodes(), 0);

    let roots_a = parent.absorb(&path_a).expect("absorb a");
    let after_a = parent.num_nodes();
    assert!(after_a > 0);

    let roots_b = parent.absorb(&path_b).expect("absorb b");
    let after_b = parent.num_nodes();

    // Reference: build both formulas directly in a single manager. The
    // node count after absorbing A + B must match the single-manager
    // build, because the unique table deduplicates (x0 ∧ x1).
    let mut reference = Manager::new();
    let _ = reference.new_var();
    let _ = reference.new_var();
    let _ = reference.new_var();
    let rf = reference.r#false();
    let rt = reference.r#true();
    let rx0 = reference.make_node(0, rf, rt);
    let rx1 = reference.make_node(1, rf, rt);
    let rx2 = reference.make_node(2, rf, rt);
    let ref_ab = reference.and(rx0, rx1);
    let _ref_b = reference.or(ref_ab, rx2);
    // Now gc to retain both roots (ref_ab and _ref_b).
    let _ = reference.gc(&[ref_ab, _ref_b]);

    // After absorbing both, parent should hold exactly the same live set
    // (arena may have different layout but the canonical node set is
    // identical). Easier: just run gc on the absorbed parent too and
    // compare live counts.
    let _ = parent.gc(&[roots_a[0], roots_b[0]]);

    assert_eq!(
        parent.num_nodes(),
        reference.num_nodes(),
        "absorb dedup: parent node count after absorbing both ({}) \
         matches single-manager reference ({}). \
         after_a={}, after_b={}",
        parent.num_nodes(), reference.num_nodes(),
        after_a, after_b,
    );
    assert_eq!(roots_a.len(), 1);
    assert_eq!(roots_b.len(), 1);

    std::fs::remove_file(&path_a).ok();
    std::fs::remove_file(&path_b).ok();
}

#[test]
fn load_detects_bad_magic() {
    let path = tmp_path("badmagic");
    // Build a file whose CRC matches but whose magic bytes don't. CRC is
    // validated first (as the outermost integrity check), so a bad-magic
    // file must have a correct CRC to reach the magic-check branch.
    let body = b"NOT_VWBDD\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0";
    let mut bytes = body.to_vec();
    // Append correct CRC for the body.
    let crc_placeholder = {
        // compute the CRC the same way `crc32` in dump.rs does.
        let mut crc = 0xffff_ffffu32;
        let table: [u32; 256] = {
            let mut t = [0u32; 256];
            for i in 0..256 {
                let mut c = i as u32;
                for _ in 0..8 {
                    c = if c & 1 != 0 { 0xedb88320 ^ (c >> 1) } else { c >> 1 };
                }
                t[i] = c;
            }
            t
        };
        for &b in &bytes {
            crc = table[((crc ^ b as u32) & 0xff) as usize] ^ (crc >> 8);
        }
        crc ^ 0xffff_ffff
    };
    bytes.extend_from_slice(&crc_placeholder.to_le_bytes());
    std::fs::write(&path, &bytes).expect("write junk");

    let err = Manager::load(&path).err().expect("load should fail");
    assert!(matches!(err, DumpError::BadMagic), "got: {:?}", err);
    std::fs::remove_file(&path).ok();
}

#[test]
fn load_detects_crc_mismatch() {
    let path = tmp_path("badcrc");

    let mut m = Manager::new();
    let r = build_formula(&mut m);
    m.dump(&path, &[r]).expect("dump");

    // Corrupt one byte in the middle of the file.
    let mut bytes = std::fs::read(&path).expect("read");
    let mid = bytes.len() / 2;
    bytes[mid] ^= 0xff;
    std::fs::write(&path, &bytes).expect("rewrite");

    let err = Manager::load(&path).err().expect("load should fail");
    assert!(
        matches!(err, DumpError::CrcMismatch { .. }),
        "got: {:?}",
        err
    );
    std::fs::remove_file(&path).ok();
}

#[test]
fn load_detects_truncation() {
    let path = tmp_path("trunc");

    let mut m = Manager::new();
    let r = build_formula(&mut m);
    m.dump(&path, &[r]).expect("dump");

    // Chop the file in half.
    let bytes = std::fs::read(&path).expect("read");
    std::fs::write(&path, &bytes[..bytes.len() / 2]).expect("rewrite");

    let err = Manager::load(&path).err().expect("load should fail");
    // Could be CrcMismatch (the trailing 4 bytes we read aren't really
    // the CRC of the body) or Truncated depending on where we cut.
    assert!(
        matches!(err, DumpError::CrcMismatch { .. } | DumpError::Truncated),
        "got: {:?}",
        err
    );
    std::fs::remove_file(&path).ok();
}

#[test]
fn load_then_reuse_in_operations() {
    // Loaded manager must be a fully functional engine, not just a bag
    // of bytes. Round-trip + do more work on top.
    let path = tmp_path("reuse");

    let mut m1 = Manager::new();
    let r = build_formula(&mut m1);
    m1.dump(&path, &[r]).expect("dump");

    let (mut m2, loaded) = Manager::load(&path).expect("load");
    let loaded_root = loaded.roots[0];
    // Do a fresh operation post-load.
    let not_root = m2.not(loaded_root);
    let should_be_true = m2.or(loaded_root, not_root);
    assert_eq!(
        should_be_true,
        m2.r#true(),
        "p ∨ ¬p must reduce to true on a loaded engine"
    );

    std::fs::remove_file(&path).ok();
}
