use vwbdd::{Manager, Ref};

#[test]
fn empty_manager_has_no_nodes() {
    let m = Manager::new();
    assert_eq!(m.num_nodes(), 0);
    assert_eq!(m.buf_len(), 0);
}

#[test]
fn terminals_are_not_buffer_nodes() {
    let m = Manager::new();
    assert_eq!(m.r#false(), Ref::Terminal(false));
    assert_eq!(m.r#true(), Ref::Terminal(true));
    assert_eq!(m.num_nodes(), 0);
}

#[test]
fn make_node_reduces_identical_children() {
    let mut m = Manager::new();
    let _x = m.new_var();
    let t = m.r#true();
    let r = m.make_node(0, t, t);
    assert_eq!(r, t, "lo == hi should reduce to that child");
    assert_eq!(m.num_nodes(), 0);
}

#[test]
fn make_node_canonicalizes() {
    let mut m = Manager::new();
    let _x = m.new_var();
    let f = m.r#false();
    let t = m.r#true();
    let a = m.make_node(0, f, t);
    let b = m.make_node(0, f, t);
    assert_eq!(a, b, "same (var, lo, hi) should return same ref");
    assert_eq!(m.num_nodes(), 1);
}

#[test]
fn distinct_functions_get_distinct_nodes() {
    let mut m = Manager::new();
    let _x = m.new_var();
    let f = m.r#false();
    let t = m.r#true();
    let pos_x = m.make_node(0, f, t); // x
    let neg_x = m.make_node(0, t, f); // ~x
    assert_ne!(pos_x, neg_x);
    assert_eq!(m.num_nodes(), 2);
}

#[test]
fn single_variable_bdd_has_one_node() {
    // The BDD for the function "x" over one variable is a single internal
    // node pointing to false/true. Classic smallest non-trivial BDD.
    let mut m = Manager::new();
    let _x = m.new_var();
    let f = m.r#false();
    let t = m.r#true();
    let _x_fn = m.make_node(0, f, t);
    assert_eq!(m.num_nodes(), 1);
}

#[test]
fn two_variable_and_bdd_has_two_nodes() {
    // x AND y, with x above y in the order. Reduced ordered BDD:
    //   x: if true -> y, if false -> false
    //   y: if true -> true, if false -> false
    // Two internal nodes.
    let mut m = Manager::new();
    let _x = m.new_var();
    let _y = m.new_var();
    let f = m.r#false();
    let t = m.r#true();
    let y_fn = m.make_node(1, f, t);
    let _and_fn = m.make_node(0, f, y_fn);
    assert_eq!(m.num_nodes(), 2);
}

#[test]
fn variable_ordering_violation_panics() {
    // Trying to make a node whose parent var is not strictly less than
    // its child's var should fail loudly.
    let mut m = Manager::new();
    let _x = m.new_var();
    let _y = m.new_var();
    let f = m.r#false();
    let t = m.r#true();
    let x_fn = m.make_node(0, f, t); // var 0
    // Now try to put var 1 above var 0 -- illegal.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut m = m;
        m.make_node(1, f, x_fn)
    }));
    assert!(result.is_err(), "expected panic on ordering violation");
}
