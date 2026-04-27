use vwbdd::leb::{decode_u128, encode_u128};

fn roundtrip(x: u128) -> (u128, usize) {
    let mut buf = Vec::new();
    encode_u128(x, &mut buf);
    let (y, n) = decode_u128(&buf);
    assert_eq!(y, x);
    assert_eq!(n, buf.len());
    (y, n)
}

#[test]
fn zero_is_one_byte() {
    let (_, n) = roundtrip(0);
    assert_eq!(n, 1);
}

#[test]
fn small_values_one_byte() {
    for x in 0u128..=127 {
        let (_, n) = roundtrip(x);
        assert_eq!(n, 1, "x={}", x);
    }
}

#[test]
fn boundary_128_is_two_bytes() {
    let (_, n) = roundtrip(128);
    assert_eq!(n, 2);
}

#[test]
fn large_values() {
    roundtrip(16_383);
    roundtrip(16_384);
    roundtrip(u32::MAX as u128);
    roundtrip(u64::MAX as u128);
    roundtrip(u128::MAX);
}

#[test]
fn concatenated_stream() {
    let mut buf = Vec::new();
    let xs = [0u128, 1, 127, 128, 16_384, u64::MAX as u128, u128::MAX];
    for &x in &xs {
        encode_u128(x, &mut buf);
    }
    let mut pos = 0;
    for &x in &xs {
        let (y, n) = decode_u128(&buf[pos..]);
        assert_eq!(y, x);
        pos += n;
    }
    assert_eq!(pos, buf.len());
}
