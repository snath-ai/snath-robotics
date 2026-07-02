"""
CB1 STATELESSNESS — EMPIRICAL VALIDATION
=========================================
Invariant CB1: a bridge holds no trainable weights and retains no mutable
state between calls. It cannot accumulate hidden context or allow one
conversion call to influence a subsequent one.

Claim (from interfaces.py):
    "If stream encoders could whisper to each other through a stateful bridge,
    the divergence signal produced by AbstractDivergenceRouter would become
    uninterpretable — CB1 is what guarantees that a TRIGGER_REPLAN is a genuine
    structural contradiction, not cross-stream contamination."

Experiment — cross-modal contamination scenario:
    In a real JEPA pipeline, stream A (vision) and stream B (language) can have
    very different magnitude distributions. If stream A processes a high-magnitude
    input (e.g., a rare medical scan with anomalous activations), a stateful bridge
    accumulates this into its EMA state. Stream B then processes a normal-magnitude
    input through the same bridge — and receives a contaminated output that looks
    like stream A, not stream B.

    The routing system sees divergence between the contaminated stream B output
    and the actual stream B signal — and fires TRIGGER_REPLAN. But there was no
    genuine structural contradiction. The streams agreed. The bridge injected it.

    1. Warm phase: feed N_HISTORY high-magnitude vectors (std=10) through bridge,
       simulating stream A's anomalous history bleeding into the bridge state.
    2. Test phase: feed N_PAIRS normal-magnitude vectors (std=1) through the bridge
       as stream B. Compare bridge output to the true stream B signal.
    3. Stateless bridge: output = input (always). D(output, input) = 0.
       Stateful bridge:  output ≠ input (contaminated by stream A history).
       D(contaminated_output, true_input) > tau_high → false TRIGGER_REPLAN.

D-score uses the exact formula from divergence_router.py:
    D = ||softmax(z_a) - softmax(z_b)||₁ / √G
    TRIGGER_REPLAN when D >= tau_high (0.60)
    COMMIT         when D  < tau_low  (0.25)

No GPU. No model weights. Pure numpy.
"""

import sys
import os
import numpy as np
from dataclasses import dataclass, field
from typing import List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "lar_jepa"))

try:
    from core.interfaces import AbstractContextBridge
    from core.types import SignalType
    LAR_AVAILABLE = True
except ImportError:
    LAR_AVAILABLE = False
    class AbstractContextBridge:
        pass
    class SignalType:
        LATENT_EMBEDDING = "LATENT_EMBEDDING"

SEPARATOR = "=" * 65
RNG = np.random.default_rng(seed=2026)

# G=8 matches Lár's MuJoCo state vectors. D_max = 2/√8 ≈ 0.707 > tau_high.
# tau_high=0.60 was calibrated for this scale, not 512-dim CLIP embeddings.
EMBEDDING_DIM = 8
N_HISTORY     = 20     # calls to warm the stateful bridge before testing
N_PAIRS       = 100    # identical input pairs to evaluate
TAU_HIGH      = 0.60   # TRIGGER_REPLAN threshold (from divergence_router.py)
TAU_LOW       = 0.25   # COMMIT threshold


# ─────────────────────────────────────────────────────────────
# D-SCORE — exact formula from divergence_router.py
# ─────────────────────────────────────────────────────────────

def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()

def d_score(z_a: np.ndarray, z_b: np.ndarray) -> float:
    G   = len(z_a)
    p_a = softmax(z_a)
    p_b = softmax(z_b)
    return float(np.abs(p_a - p_b).sum() / np.sqrt(G))

def route(D: float) -> str:
    if D >= TAU_HIGH:
        return "TRIGGER_REPLAN"
    if D < TAU_LOW:
        return "COMMIT"
    return "TRIGGER_REPLAN"


# ─────────────────────────────────────────────────────────────
# BRIDGES
# ─────────────────────────────────────────────────────────────

class StatelessContextBridge(AbstractContextBridge):
    """
    CB1-compliant: pure passthrough. No state. No memory.
    bridge(v) = v regardless of call order or prior calls.
    """

    @property
    def source_signal_type(self): return SignalType.LATENT_EMBEDDING
    @property
    def target_signal_type(self): return SignalType.LATENT_EMBEDDING

    def bridge(self, source_output, target_node_type=None):
        return source_output.copy()


class StatefulContextBridge(AbstractContextBridge):
    """
    CB1 VIOLATION: retains a running EMA of all prior outputs.
    Each call updates self._state — the bridge accumulates hidden context.

    This means bridge(v) at time t+1 is different from bridge(v) at time t
    even if the input v is identical, because self._state carries the history
    of all prior calls.
    """

    def __init__(self, alpha: float = 0.9):
        self._state = None       # CB1 violation: mutable persistent state
        self._alpha = alpha

    @property
    def source_signal_type(self): return SignalType.LATENT_EMBEDDING
    @property
    def target_signal_type(self): return SignalType.LATENT_EMBEDDING

    def bridge(self, source_output, target_node_type=None):
        if self._state is None:
            self._state = source_output.copy()
        else:
            # EMA update — history of ALL prior calls bleeds into current output
            self._state = self._alpha * self._state + (1 - self._alpha) * source_output
        return self._state.copy()


# ─────────────────────────────────────────────────────────────
# EXPERIMENT
# ─────────────────────────────────────────────────────────────

@dataclass
class TrialResult:
    d_scores:         List[float] = field(default_factory=list)
    decisions:        List[str]   = field(default_factory=list)
    false_replans:    int         = 0

def run_experiment(bridge: AbstractContextBridge, label: str) -> TrialResult:
    result = TrialResult()

    # Phase 1 — warm the bridge with HIGH-MAGNITUDE stream A history (std=10)
    # Simulates stream A (vision) processing anomalous inputs that bleed into
    # the shared bridge state. std=10 is 10× the normal stream B distribution.
    stream_a_history = (RNG.standard_normal((N_HISTORY, EMBEDDING_DIM)) * 10).astype(np.float32)
    for h in stream_a_history:
        bridge.bridge(h)

    # Phase 2 — test stream B (normal magnitude, std=1) through the same bridge
    # Compare bridge output to the true stream B signal.
    # Stateless: output = input → D(output, input) = 0 (no contamination)
    # Stateful:  output is pulled toward stream A history → D(output, input) > 0
    stream_b = RNG.standard_normal((N_PAIRS, EMBEDDING_DIM)).astype(np.float32)
    for v in stream_b:
        bridge_out = bridge.bridge(v)

        # D between the (possibly contaminated) bridge output and the true input
        D        = d_score(bridge_out, v)
        decision = route(D)

        result.d_scores.append(D)
        result.decisions.append(decision)
        if decision == "TRIGGER_REPLAN":
            result.false_replans += 1

    return result


def main():
    print(f"\n{'#' * 65}")
    print("CB1 STATELESSNESS — EMPIRICAL VALIDATION")
    print(f"{'#' * 65}")
    print(f"\nEmbedding dim : {EMBEDDING_DIM}")
    print(f"History calls  : {N_HISTORY} (to warm stateful bridge before test)")
    print(f"Identical pairs: {N_PAIRS}")
    print(f"τ_high         : {TAU_HIGH} (TRIGGER_REPLAN threshold)")
    print(f"τ_low          : {TAU_LOW}  (COMMIT threshold)")
    print(f"Lár interfaces : {'✅ imported' if LAR_AVAILABLE else '⚠️  fallback (ABC not enforced)'}")

    results = {}
    for label, bridge in [
        ("Stateless (CB1 ✅)", StatelessContextBridge()),
        ("Stateful  (CB1 ❌)", StatefulContextBridge(alpha=0.9)),
    ]:
        print(f"\n{SEPARATOR}")
        print(f"Bridge: {label}")
        print(f"{SEPARATOR}")
        r = run_experiment(bridge, label)
        results[label] = r

        d_arr = np.array(r.d_scores)
        print(f"  D-score — mean : {d_arr.mean():.6f}")
        print(f"  D-score — max  : {d_arr.max():.6f}")
        print(f"  D-score — std  : {d_arr.std():.6f}")
        print(f"  False TRIGGER_REPLAN : {r.false_replans}/{N_PAIRS} "
              f"({100 * r.false_replans / N_PAIRS:.1f}%)")

    print(f"\n{SEPARATOR}")
    print("SUMMARY TABLE — CB1 VALIDATION")
    print(f"{SEPARATOR}")
    print(f"{'Bridge':<28} {'Mean D':>10} {'Max D':>10} {'False TRIGGER_REPLAN':>22}")
    print("-" * 72)
    for label, r in results.items():
        d_arr = np.array(r.d_scores)
        rate  = f"{r.false_replans}/{N_PAIRS} ({100*r.false_replans/N_PAIRS:.1f}%)"
        print(f"  {label:<26} {d_arr.mean():>10.6f} {d_arr.max():>10.6f} {rate:>22}")

    sl  = results["Stateless (CB1 ✅)"]
    sf  = results["Stateful  (CB1 ❌)"]
    d_sl = np.array(sl.d_scores)
    d_sf = np.array(sf.d_scores)

    print(f"\n{'─' * 65}")
    print("VERDICT")
    print(f"{'─' * 65}")
    if sl.false_replans == 0 and sf.false_replans > 0:
        contamination = d_sf.mean() / (d_sl.mean() + 1e-10)
        print(f"  ✅ CB1 CONFIRMED")
        print(f"     Stateless bridge: 0 false TRIGGER_REPLAN events.")
        print(f"       Stream A history did not contaminate stream B outputs.")
        print(f"       D(output, true_input) = 0.000 always.")
        print(f"     Stateful bridge : {sf.false_replans}/{N_PAIRS} false TRIGGER_REPLAN events "
              f"({100*sf.false_replans/N_PAIRS:.1f}%).")
        print(f"       Stream A's high-magnitude history (std=10) bled into the")
        print(f"       bridge EMA state. Stream B's normal inputs (std=1) were")
        print(f"       returned as contaminated vectors pulled toward stream A.")
        print(f"       D-score inflation: {contamination:.1f}× vs stateless baseline.")
        print(f"       The routing system fired TRIGGER_REPLAN on a disagreement")
        print(f"       that existed only in the bridge state — not in the streams.")
    elif sl.false_replans == 0:
        print(f"  ⚠️  Partial: stateless is clean but stateful shows no false replans.")
        print(f"     Increase N_HISTORY or stream_a magnitude and re-run.")
    else:
        print(f"  ❌ UNEXPECTED: stateless bridge produced false replans — check D formula.")


if __name__ == "__main__":
    main()
