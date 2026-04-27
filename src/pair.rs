//! Integer pairing functions used to pack a BDD node into one u128 before
//! LEB128 encoding.
//!
//! `interleave(x, y)` bit-interleaves two u64s into a u128. Symmetric; good
//! when both sides have similar magnitude.
//!
//! `minsky(x, y)` puts `y` in the low bits (biased-toward-zero costs little),
//! and `x` in the high bits.
//!
//! A BDD node packs as:
//!     packed = minsky(interleave(lo_code, hi_code), v_skip)

/// Spread the low 64 bits of `x` into the even bit positions of a u128.
fn spread(x: u64) -> u128 {
    let mut x = x as u128;
    x = (x | (x << 32)) & 0x0000_0000_ffff_ffff_0000_0000_ffff_ffff;
    x = (x | (x << 16)) & 0x0000_ffff_0000_ffff_0000_ffff_0000_ffff;
    x = (x | (x << 8)) & 0x00ff_00ff_00ff_00ff_00ff_00ff_00ff_00ff;
    x = (x | (x << 4)) & 0x0f0f_0f0f_0f0f_0f0f_0f0f_0f0f_0f0f_0f0f;
    x = (x | (x << 2)) & 0x3333_3333_3333_3333_3333_3333_3333_3333;
    x = (x | (x << 1)) & 0x5555_5555_5555_5555_5555_5555_5555_5555;
    x
}

fn gather(z: u128) -> u64 {
    let mut z = z & 0x5555_5555_5555_5555_5555_5555_5555_5555;
    z = (z | (z >> 1)) & 0x3333_3333_3333_3333_3333_3333_3333_3333;
    z = (z | (z >> 2)) & 0x0f0f_0f0f_0f0f_0f0f_0f0f_0f0f_0f0f_0f0f;
    z = (z | (z >> 4)) & 0x00ff_00ff_00ff_00ff_00ff_00ff_00ff_00ff;
    z = (z | (z >> 8)) & 0x0000_ffff_0000_ffff_0000_ffff_0000_ffff;
    z = (z | (z >> 16)) & 0x0000_0000_ffff_ffff_0000_0000_ffff_ffff;
    z = (z | (z >> 32)) & 0x0000_0000_0000_0000_ffff_ffff_ffff_ffff;
    z as u64
}

/// Bit-interleave two u64s into a u128. `x` goes in even bits, `y` in odd bits.
pub fn interleave(x: u64, y: u64) -> u128 {
    spread(x) | (spread(y) << 1)
}

pub fn deinterleave(z: u128) -> (u64, u64) {
    (gather(z), gather(z >> 1))
}

/// Minsky pairing: `(x << (1 + y)) | (1 << y)`. `y` is the count of trailing
/// zeros in the result, so small `y` costs almost nothing in bits.
pub fn minsky(x: u128, y: u32) -> u128 {
    (x << (y + 1)) | (1u128 << y)
}

pub fn unminsky(z: u128) -> (u128, u32) {
    debug_assert!(z != 0, "minsky output is never zero");
    let y = z.trailing_zeros();
    let x = z >> (y + 1);
    (x, y)
}
