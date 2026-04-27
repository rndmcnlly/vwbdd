//! Lightweight phase-timing counters, toggleable at compile time.
//!
//! Design goal: when `profile-timing` is off, every `Guard::new` and drop is
//! a zero-cost inlined no-op — the compiler erases the entire struct. When
//! on, we accumulate (call count, total nanoseconds) per `Phase` into
//! thread-local atomics and expose a snapshot via [`snapshot`].
//!
//! The timing method is `std::time::Instant::now()` at enter and drop, which
//! on macOS/aarch64 is ~20 ns per call. That's a ~20 ns overhead per measured
//! phase — nontrivial when a phase is 10-30 ns, but it's the same overhead
//! across phases so relative % is preserved.

#[derive(Copy, Clone, Debug)]
pub enum Phase {
    /// `var_of`: decode just the var field.
    VarOf,
    /// `decode_node` public API; also used by `cofactor` when it needs lo/hi.
    DecodeNode,
    /// Unique-table lookup verify-on-decode (tag match → full decode).
    DecodeVerify,
    /// Unique-table resize rebuild decode.
    DecodeResize,
    /// `encode_node_at` in the make_node miss path.
    Encode,
}

impl Phase {
    pub const ALL: &'static [Phase] = &[
        Phase::VarOf,
        Phase::DecodeNode,
        Phase::DecodeVerify,
        Phase::DecodeResize,
        Phase::Encode,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Phase::VarOf => "var_of",
            Phase::DecodeNode => "decode_node",
            Phase::DecodeVerify => "decode_verify",
            Phase::DecodeResize => "decode_resize",
            Phase::Encode => "encode",
        }
    }

    fn idx(self) -> usize {
        match self {
            Phase::VarOf => 0,
            Phase::DecodeNode => 1,
            Phase::DecodeVerify => 2,
            Phase::DecodeResize => 3,
            Phase::Encode => 4,
        }
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub struct PhaseStat {
    pub calls: u64,
    pub ns: u64,
}

#[derive(Copy, Clone, Debug, Default)]
pub struct Snapshot {
    pub stats: [PhaseStat; 5],
}

impl Snapshot {
    pub fn get(&self, p: Phase) -> PhaseStat {
        self.stats[p.idx()]
    }

    pub fn total_ns(&self) -> u64 {
        self.stats.iter().map(|s| s.ns).sum()
    }

    /// Print a human-readable breakdown to stderr. `wall_ms` is the
    /// externally-measured wall time to contextualize the "unaccounted"
    /// portion.
    pub fn report(&self, label: &str, wall_ms: Option<f64>) {
        let total_ns = self.total_ns();
        eprintln!("--- profile: {} ---", label);
        for &p in Phase::ALL {
            let s = self.get(p);
            if s.calls == 0 {
                continue;
            }
            let pct = if total_ns == 0 { 0.0 } else { s.ns as f64 * 100.0 / total_ns as f64 };
            let ns_per = s.ns as f64 / s.calls as f64;
            eprintln!(
                "  {:<14} {:>11} calls  {:>9.2} ms  {:>5.1}%  {:>6.1} ns/call",
                p.name(),
                s.calls,
                s.ns as f64 / 1e6,
                pct,
                ns_per,
            );
        }
        eprintln!(
            "  {:<14} {:>11}         {:>9.2} ms",
            "accounted", "", total_ns as f64 / 1e6
        );
        if let Some(wall) = wall_ms {
            let wall_ns = wall * 1e6;
            let unacc_ns = wall_ns - total_ns as f64;
            let pct = unacc_ns * 100.0 / wall_ns.max(1.0);
            eprintln!(
                "  {:<14} {:>11}         {:>9.2} ms  {:>5.1}%",
                "unaccounted", "", unacc_ns / 1e6, pct,
            );
        }
    }
}

#[cfg(feature = "profile-timing")]
mod imp {
    use super::{Phase, PhaseStat, Snapshot};
    use std::cell::Cell;
    use std::time::Instant;

    thread_local! {
        static STATS: Cell<[(u64, u64); 5]> = const { Cell::new([(0, 0); 5]) };
    }

    pub struct Guard {
        phase: Phase,
        t0: Instant,
    }

    impl Guard {
        #[inline]
        pub fn new(phase: Phase) -> Self {
            Self { phase, t0: Instant::now() }
        }
    }

    impl Drop for Guard {
        #[inline]
        fn drop(&mut self) {
            let dt = self.t0.elapsed().as_nanos() as u64;
            let idx = self.phase.idx();
            STATS.with(|s| {
                let mut arr = s.get();
                arr[idx].0 += 1;
                arr[idx].1 += dt;
                s.set(arr);
            });
        }
    }

    pub fn snapshot() -> Snapshot {
        STATS.with(|s| {
            let arr = s.get();
            let mut out = Snapshot::default();
            for (i, (calls, ns)) in arr.iter().enumerate() {
                out.stats[i] = PhaseStat { calls: *calls, ns: *ns };
            }
            out
        })
    }

    pub fn reset() {
        STATS.with(|s| s.set([(0, 0); 5]));
    }
}

#[cfg(not(feature = "profile-timing"))]
mod imp {
    use super::{Phase, Snapshot};

    pub struct Guard;

    impl Guard {
        #[inline(always)]
        pub fn new(_: Phase) -> Self {
            Self
        }
    }

    pub fn snapshot() -> Snapshot {
        Snapshot::default()
    }

    pub fn reset() {}
}

pub use imp::{reset, snapshot, Guard};
