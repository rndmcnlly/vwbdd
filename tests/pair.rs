use vwbdd::pair::{deinterleave, interleave, minsky, unminsky};

#[test]
fn interleave_roundtrips_small() {
    for x in 0u64..64 {
        for y in 0u64..64 {
            let z = interleave(x, y);
            assert_eq!(deinterleave(z), (x, y), "failed at ({}, {})", x, y);
        }
    }
}

#[test]
fn interleave_roundtrips_large() {
    let cases: &[(u64, u64)] = &[
        (0, 0),
        (1, 0),
        (0, 1),
        (u32::MAX as u64, 0),
        (0, u32::MAX as u64),
        (u32::MAX as u64, u32::MAX as u64),
        (u64::MAX, 0),
        (0, u64::MAX),
        (u64::MAX, u64::MAX),
        (0xdeadbeef_cafebabe, 0x0123456789abcdef),
    ];
    for &(x, y) in cases {
        let z = interleave(x, y);
        assert_eq!(deinterleave(z), (x, y), "failed at ({}, {})", x, y);
    }
}

#[test]
fn interleave_small_inputs_stay_small() {
    // Two 5-bit numbers interleave into at most 10 bits = fits in 2 LEB128 bytes.
    let z = interleave(31, 31);
    assert!(z < (1 << 10));
}

#[test]
fn minsky_roundtrips() {
    for x in 0u128..64 {
        for y in 0u32..20 {
            let z = minsky(x, y);
            assert_eq!(unminsky(z), (x, y), "failed at ({}, {})", x, y);
        }
    }
}

#[test]
fn minsky_y_zero_is_cheap() {
    assert_eq!(minsky(0, 0), 1);
    assert_eq!(minsky(1, 0), 3);
    assert_eq!(minsky(5, 0), 11);
    assert!(minsky(interleave(2, 3), 0) < 128, "must fit in 1 LEB byte");
}

#[test]
fn minsky_large_y_costs_bits() {
    let z = minsky(1, 7);
    assert!(z >= 128, "y=7 should push result past 1-byte LEB");
}

#[test]
fn interleave_full_width_is_exact_inverse() {
    // Stress test: ensure no bits are lost at the u64 ceiling.
    let (x, y) = (0xffff_ffff_ffff_ffffu64, 0xaaaa_5555_aaaa_5555u64);
    let z = interleave(x, y);
    assert_eq!(deinterleave(z), (x, y));
}
