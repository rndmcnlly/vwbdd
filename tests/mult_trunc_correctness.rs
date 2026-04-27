//! Quick correctness check: the truncated mult builder produces the node
//! counts iota reports in its demo.
//!
//! iota demo output (https://rndmcnlly.github.io/oxidd-wasm/iota.html):
//!   k=7  →   2,634 nodes
//!   k=10 →  58,533 nodes
//!   k=12 → 464,183 nodes
//!   k=13 → 1,292,160 nodes
//!
//! These values are oxidd's `node_count()` which includes both terminals
//! (vwbdd excludes). So expected vwbdd reachable = iota_count - 2.

use vwbdd::Manager;

mod mult_shared;
use mult_shared::{build_mult_trunc, vw_reachable};

#[test]
fn trunc_mult_k7_matches_iota() {
    let mut vw = Manager::new();
    let r = build_mult_trunc(&mut vw, 7);
    let n = vw_reachable(&vw, r);
    assert_eq!(n, 2634 - 2, "k=7: vwbdd got {}, iota reports 2634 (incl. terminals)", n);
}

#[test]
fn trunc_mult_k10_matches_iota() {
    let mut vw = Manager::new();
    let r = build_mult_trunc(&mut vw, 10);
    let n = vw_reachable(&vw, r);
    assert_eq!(n, 58533 - 2, "k=10: vwbdd got {}, iota reports 58533 (incl. terminals)", n);
}
