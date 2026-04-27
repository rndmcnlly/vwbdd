//! Minimal unsigned LEB128 for u128. Encode to `Vec<u8>` / decode from a byte
//! slice, returning (value, bytes_consumed).

pub fn encode_u128(mut x: u128, out: &mut Vec<u8>) {
    loop {
        let byte = (x & 0x7f) as u8;
        x >>= 7;
        if x == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

pub fn decode_u128(buf: &[u8]) -> (u128, usize) {
    let mut result: u128 = 0;
    let mut shift: u32 = 0;
    for (i, &b) in buf.iter().enumerate() {
        let chunk = (b & 0x7f) as u128;
        result |= chunk << shift;
        if b & 0x80 == 0 {
            return (result, i + 1);
        }
        shift += 7;
        if shift >= 128 {
            panic!("LEB128 overflow");
        }
    }
    panic!("LEB128 truncated");
}
