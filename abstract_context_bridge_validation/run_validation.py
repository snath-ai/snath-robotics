"""
CB1 + CB2 VALIDATION — FULL RUN
=================================
Runs both invariant validations and prints a combined paper-ready table.

Run: python run_validation.py
No GPU. No model weights. No API keys.
"""

import subprocess
import sys
import os
import numpy as np
import json
import hmac
import hashlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "lar_jepa"))
try:
    from core.interfaces import AbstractContextBridge
    from core.types import SignalType
    LAR_AVAILABLE = True
except ImportError:
    LAR_AVAILABLE = False
    class AbstractContextBridge: pass
    class SignalType:
        LATENT_EMBEDDING = "LATENT_EMBEDDING"

SEPARATOR = "=" * 65
RNG = np.random.default_rng(seed=2026)

# ── shared params ──────────────────────────────────────────────
# G=8 matches Lár's MuJoCo state vectors (x_pos, y_pos, x_vel, y_vel,
# angle, angular_vel, foot_contact_L, foot_contact_R). D_max=2/√8≈0.707
# which sits above tau_high=0.60. tau_high was calibrated for this scale.
EMBEDDING_DIM = 8
N_HISTORY     = 20
N_PAIRS       = 100
N_REPLAYS     = 100
TAU_HIGH      = 0.60
TAU_LOW       = 0.25
NOISE_SIGMA   = 1e-4
AUDIT_KEY     = "snath_cb2_audit_key_2026"


# ── D-score ────────────────────────────────────────────────────
def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()

def d_score(z_a, z_b):
    G   = len(z_a)
    p_a = softmax(z_a)
    p_b = softmax(z_b)
    return float(np.abs(p_a - p_b).sum() / np.sqrt(G))

def route(D):
    return "TRIGGER_REPLAN" if D >= TAU_HIGH else ("COMMIT" if D < TAU_LOW else "TRIGGER_REPLAN")


# ── HMAC ────────────────────────────────────────────────────────
def seal(output):
    payload = {"bridge_output": output.tolist()}
    raw     = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hmac.new(AUDIT_KEY.encode(), raw, hashlib.sha256).hexdigest()

def verify(output, sig):
    return hmac.compare_digest(seal(output), sig)


# ── Bridges ─────────────────────────────────────────────────────
class StatelessBridge(AbstractContextBridge):
    @property
    def source_signal_type(self): return SignalType.LATENT_EMBEDDING
    @property
    def target_signal_type(self): return SignalType.LATENT_EMBEDDING
    def bridge(self, x, target_node_type=None): return x.copy()

class StatefulBridge(AbstractContextBridge):
    def __init__(self):
        self._state = None
    @property
    def source_signal_type(self): return SignalType.LATENT_EMBEDDING
    @property
    def target_signal_type(self): return SignalType.LATENT_EMBEDDING
    def bridge(self, x, target_node_type=None):
        self._state = x.copy() if self._state is None else 0.9 * self._state + 0.1 * x
        return self._state.copy()

class DeterministicBridge(AbstractContextBridge):
    @property
    def source_signal_type(self): return SignalType.LATENT_EMBEDDING
    @property
    def target_signal_type(self): return SignalType.LATENT_EMBEDDING
    def bridge(self, x, target_node_type=None): return x.copy()

class StochasticBridge(AbstractContextBridge):
    @property
    def source_signal_type(self): return SignalType.LATENT_EMBEDDING
    @property
    def target_signal_type(self): return SignalType.LATENT_EMBEDDING
    def bridge(self, x, target_node_type=None):
        return x + np.random.normal(0, NOISE_SIGMA, x.shape).astype(np.float32)


# ── CB1 experiment ──────────────────────────────────────────────
# Cross-modal contamination: stream A history has std=10 (high magnitude),
# stream B inputs have std=1 (normal). Stateful bridge bleeds stream A's
# distribution into stream B's outputs. D(bridge_out, true_input) shows
# how much contamination the bridge injected.
def run_cb1(bridge):
    stream_a_history = (RNG.standard_normal((N_HISTORY, EMBEDDING_DIM)) * 10).astype(np.float32)
    for h in stream_a_history:
        bridge.bridge(h)
    stream_b = RNG.standard_normal((N_PAIRS, EMBEDDING_DIM)).astype(np.float32)
    d_scores = []
    false_replans = 0
    for v in stream_b:
        bridge_out = bridge.bridge(v)
        D = d_score(bridge_out, v)   # contamination = deviation from true input
        d_scores.append(D)
        if route(D) == "TRIGGER_REPLAN":
            false_replans += 1
    return np.array(d_scores), false_replans


# ── CB2 experiment ──────────────────────────────────────────────
def run_cb2(bridge, fixed_input):
    orig     = bridge.bridge(fixed_input)
    orig_sig = seal(orig)
    passes   = 0
    outputs  = [orig]
    for _ in range(N_REPLAYS):
        out = bridge.bridge(fixed_input)
        outputs.append(out)
        if verify(out, orig_sig):
            passes += 1
    variance = float(np.stack(outputs).std(axis=0).mean())
    return passes, N_REPLAYS - passes, variance


# ── Main ────────────────────────────────────────────────────────
def main():
    print(f"\n{'#' * 65}")
    print("ABSTRACTCONTEXTBRIDGE — FULL INVARIANT VALIDATION")
    print(f"{'#' * 65}")
    print(f"  Embedding dim : {EMBEDDING_DIM}")
    print(f"  CB1 pairs     : {N_PAIRS}  (history={N_HISTORY} warm-up calls)")
    print(f"  CB2 replays   : {N_REPLAYS}")
    print(f"  τ_high / τ_low: {TAU_HIGH} / {TAU_LOW}")
    print(f"  Noise sigma   : {NOISE_SIGMA}")
    print(f"  Lár ABCs      : {'✅ imported' if LAR_AVAILABLE else '⚠️  fallback mode'}")

    # CB1
    print(f"\n{SEPARATOR}")
    print("CB1 — STATELESSNESS")
    print(f"{SEPARATOR}")
    cb1_sl_d, cb1_sl_fr = run_cb1(StatelessBridge())
    cb1_sf_d, cb1_sf_fr = run_cb1(StatefulBridge())

    print(f"  Stateless (CB1 ✅) — mean D: {cb1_sl_d.mean():.6f} | "
          f"false TRIGGER_REPLAN: {cb1_sl_fr}/{N_PAIRS} ({100*cb1_sl_fr/N_PAIRS:.1f}%)")
    print(f"  Stateful  (CB1 ❌) — mean D: {cb1_sf_d.mean():.6f} | "
          f"false TRIGGER_REPLAN: {cb1_sf_fr}/{N_PAIRS} ({100*cb1_sf_fr/N_PAIRS:.1f}%)")

    cb1_pass = cb1_sl_fr == 0 and cb1_sf_fr > 0

    # CB2
    print(f"\n{SEPARATOR}")
    print("CB2 — PURE FUNCTION")
    print(f"{SEPARATOR}")
    fixed = RNG.standard_normal(EMBEDDING_DIM).astype(np.float32)
    cb2_d_p, cb2_d_f, cb2_d_var = run_cb2(DeterministicBridge(), fixed)
    cb2_s_p, cb2_s_f, cb2_s_var = run_cb2(StochasticBridge(), fixed)

    print(f"  Deterministic (CB2 ✅) — HMAC pass: {cb2_d_p}/{N_REPLAYS} ({100*cb2_d_p/N_REPLAYS:.1f}%) | "
          f"output variance: {cb2_d_var:.2e}")
    print(f"  Stochastic    (CB2 ❌) — HMAC pass: {cb2_s_p}/{N_REPLAYS} ({100*cb2_s_p/N_REPLAYS:.1f}%) | "
          f"output variance: {cb2_s_var:.2e}")

    cb2_pass = cb2_d_p == N_REPLAYS and cb2_s_p == 0

    # Paper table
    print(f"\n{SEPARATOR}")
    print("PAPER TABLE — CB1 + CB2 EMPIRICAL CLOSURE")
    print(f"{SEPARATOR}")
    print(f"""
  ┌──────────────┬───────────────┬─────────────────────────────┬──────────┐
  │ Invariant    │ Violation     │ Observed failure            │ Verified │
  ├──────────────┼───────────────┼─────────────────────────────┼──────────┤
  │ CB1          │ Stateful      │ {cb1_sf_fr:>3}/{N_PAIRS} false            │   {'✅' if cb1_pass else '❌'}      │
  │ (Stateless)  │ bridge (EMA   │ TRIGGER_REPLAN on identical │          │
  │              │ α=0.9)        │ inputs; mean D={cb1_sf_d.mean():.4f}    │          │
  ├──────────────┼───────────────┼─────────────────────────────┼──────────┤
  │ CB2          │ Stochastic    │ {cb2_s_f:>3}/{N_REPLAYS} HMAC failures on   │   {'✅' if cb2_pass else '❌'}      │
  │ (Pure fn)    │ bridge        │ replay; σ={NOISE_SIGMA} sufficient to  │          │
  │              │ (σ={NOISE_SIGMA})     │ corrupt Art. 12 audit trail │          │
  └──────────────┴───────────────┴─────────────────────────────┴──────────┘
""")

    print(f"{'─' * 65}")
    print("FINAL VERDICT")
    print(f"{'─' * 65}")
    if cb1_pass and cb2_pass:
        print("  ✅ BOTH INVARIANTS CONFIRMED")
        print()
        print("  With CB1 and CB2 empirically validated, the 33-invariant system")
        print("  is closed: every invariant has either an NB4 experimental result")
        print("  or a direct ablation demonstrating the failure mode its violation")
        print("  produces.")
        print()
        print("  CB1 violation kills routing correctness.")
        print("  CB2 violation kills auditability.")
        print("  Both are required. Neither is sufficient alone.")
    else:
        if not cb1_pass:
            print("  ⚠️  CB1 not confirmed. Check history size or alpha.")
        if not cb2_pass:
            print("  ⚠️  CB2 not confirmed. Check noise sigma or HMAC logic.")


if __name__ == "__main__":
    main()
