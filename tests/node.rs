use vwbdd::node::{decode_node_at, encode_node_at, Node, Ref};

#[test]
fn node_with_two_terminal_children_roundtrip() {
    // One var total; node has var=0 with both children terminal.
    let mut buf = Vec::new();
    encode_node_at(0, Ref::Terminal(false), Ref::Terminal(true), 0, &mut buf);
    let (node, n) = decode_node_at(&buf, 0);
    assert_eq!(n, buf.len());
    assert_eq!(
        node,
        Node {
            var: 0,
            lo: Ref::Terminal(false),
            hi: Ref::Terminal(true),
        }
    );
}

#[cfg(not(any(feature = "encoding-per-field", feature = "encoding-fixed")))]
#[test]
fn node_with_terminal_children_packs_small() {
    // Interleaved: var=0 is 1 LEB byte; children interleave(0, 1) = 2, LEB = 1 byte.
    // Total = 2 bytes.
    let mut buf = Vec::new();
    encode_node_at(0, Ref::Terminal(false), Ref::Terminal(true), 0, &mut buf);
    assert_eq!(buf.len(), 2, "simple variable-x node should be 2 bytes");
}

#[cfg(feature = "encoding-per-field")]
#[test]
fn node_with_terminal_children_packs_small_per_field() {
    // Per-field: var=0, lo_code=0, hi_code=1 — three 1-byte LEBs.
    let mut buf = Vec::new();
    encode_node_at(0, Ref::Terminal(false), Ref::Terminal(true), 0, &mut buf);
    assert_eq!(buf.len(), 3, "per-field node should be 3 bytes");
}

#[cfg(feature = "encoding-fixed")]
#[test]
fn node_with_terminal_children_packs_small_fixed() {
    // Fixed: always 12 bytes.
    let mut buf = Vec::new();
    encode_node_at(0, Ref::Terminal(false), Ref::Terminal(true), 0, &mut buf);
    assert_eq!(buf.len(), 12, "fixed node should be 12 bytes");
}

#[test]
fn stacked_nodes_roundtrip() {
    // Two vars. Node A at offset 0 is var=1 with terminal children.
    // Node B at offset a_len is var=0 with lo=A, hi=true.
    let mut buf = Vec::new();

    encode_node_at(
        1,
        Ref::Terminal(false),
        Ref::Terminal(true),
        0,
        &mut buf,
    );
    let a_len = buf.len();

    let b_off = buf.len() as u64;
    encode_node_at(
        0,
        Ref::Node(0),
        Ref::Terminal(true),
        b_off,
        &mut buf,
    );

    let (a, _) = decode_node_at(&buf, 0);
    assert_eq!(a.var, 1);
    assert_eq!(a.lo, Ref::Terminal(false));
    assert_eq!(a.hi, Ref::Terminal(true));

    let (b, _) = decode_node_at(&buf[a_len..], b_off);
    assert_eq!(b.var, 0);
    assert_eq!(b.lo, Ref::Node(0));
    assert_eq!(b.hi, Ref::Terminal(true));
}
