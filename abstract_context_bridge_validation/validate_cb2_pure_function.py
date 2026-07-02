"""
CB2 PURE FUNCTION — EMPIRICAL VALIDATION
==========================================
Invariant CB2: bridge(source_output) is deterministic and side-effect-free.
The same source signal always produces the same adapted output regardless of
call order, call count, or any external state.

Claim (from interfaces.py):
    "This makes every cross-modal conversion independently verifiable and
    auditable by the Lár GraphExecutor's HMAC trail — a non-deterministic
    bridge would corrupt the Art. 12 audit record."

Experiment:
    1. Seal the bridge output for a fixed input with HMAC-SHA256.
    2. Replay: run the same input through the bridge again.
    3. Verify the replayed output against the sealed signature.

    Deterministic bridge (CB2 ✅):
        output is identical on every call → HMAC verifies on 100% of replays.

    Stochastic bridge (CB2 ❌):
        output contains noise → differs on replay → HMAC fails on 100% of replays.
        The Art. 12 audit record is permanently corrupted: the inspector cannot
        verify what the bridge actually produced at step N.

HMAC pattern matches GxPAuditLogger from lar_pharma_local.py and
GraphExecutor's hmac_secret audit trail.

No GPU. No model weights. Pure numpy + stdlib hmac.
"""

import sys
import os
import json
import hmac
import hashlib
import numpy as np
from dataclasses import dataclass

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
RNG        = np.random.default_rng(seed=2026)

EMBEDDING_DIM  = 8    # MuJoCo state dim; CB2 holds at any G, but consistent with CB1
N_REPLAYS      = 100      # times to replay the same input and verify HMAC
NOISE_SIGMA    = 1e-4     # stochastic bridge noise — tiny but enough to corrupt HMAC
AUDIT_KEY      = "snath_cb2_audit_key_2026"


# ─────────────────────────────────────────────────────────────
# HMAC SEAL / VERIFY
# Matches the pattern in GxPAuditLogger (lar_pharma_local.py)
# and GraphExecutor's verify_step_integrity (logger.py)
# ─────────────────────────────────────────────────────────────

def seal(output: np.ndarray) -> str:
    payload = {"bridge_output": output.tolist()}
    raw     = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hmac.new(AUDIT_KEY.encode(), raw, hashlib.sha256).hexdigest()

def verify(output: np.ndarray, expected_sig: str) -> bool:
    return hmac.compare_digest(seal(output), expected_sig)


# ─────────────────────────────────────────────────────────────
# BRIDGES
# ─────────────────────────────────────────────────────────────

class DeterministicContextBridge(AbstractContextBridge):
    """
    CB2-compliant: pure passthrough. Same input → same output. Always.
    Every replay produces an output that HMAC-verifies against the original seal.
    """

    @property
    def source_signal_type(self): return SignalType.LATENT_EMBEDDING
    @property
    def target_signal_type(self): return SignalType.LATENT_EMBEDDING

    def bridge(self, source_output, target_node_type=None):
        return source_output.copy()


class StochasticContextBridge(AbstractContextBridge):
    """
    CB2 VIOLATION: adds small Gaussian noise to every output.
    Same input → different output on every call.

    Consequence: the HMAC seal from call 1 cannot be verified against
    call 2's output. The Art. 12 audit record is permanently broken —
    an inspector replaying the log sees a different bridge output than
    what was originally produced, with no way to detect which is real.
    """

    def __init__(self, sigma: float = NOISE_SIGMA):
        self._sigma = sigma

    @property
    def source_signal_type(self): return SignalType.LATENT_EMBEDDING
    @property
    def target_signal_type(self): return SignalType.LATENT_EMBEDDING

    def bridge(self, source_output, target_node_type=None):
        noise = np.random.normal(0, self._sigma, source_output.shape).astype(np.float32)
        return source_output + noise   # CB2 violation: non-deterministic


# ─────────────────────────────────────────────────────────────
# EXPERIMENT
# ─────────────────────────────────────────────────────────────

@dataclass
class ReplayResult:
    passes: int = 0
    fails:  int = 0
    output_variance: float = 0.0

def run_replay_experiment(bridge: AbstractContextBridge, fixed_input: np.ndarray) -> ReplayResult:
    # Step 1 — original call: produce the output and seal it
    original_output = bridge.bridge(fixed_input)
    original_sig    = seal(original_output)

    outputs = [original_output]
    result  = ReplayResult()

    # Step 2 — replay N_REPLAYS times with the same input
    for _ in range(N_REPLAYS):
        replayed_output = bridge.bridge(fixed_input)
        outputs.append(replayed_output)

        if verify(replayed_output, original_sig):
            result.passes += 1
        else:
            result.fails  += 1

    # Measure actual output variance across replays
    stacked          = np.stack(outputs)
    result.output_variance = float(stacked.std(axis=0).mean())
    return result


def main():
    print(f"\n{'#' * 65}")
    print("CB2 PURE FUNCTION — EMPIRICAL VALIDATION")
    print(f"{'#' * 65}")
    print(f"\nEmbedding dim  : {EMBEDDING_DIM}")
    print(f"Replays        : {N_REPLAYS}")
    print(f"Noise sigma    : {NOISE_SIGMA} (stochastic bridge)")
    print(f"HMAC algorithm : SHA-256")
    print(f"Lár interfaces : {'✅ imported' if LAR_AVAILABLE else '⚠️  fallback (ABC not enforced)'}")

    fixed_input = RNG.standard_normal(EMBEDDING_DIM).astype(np.float32)
    print(f"\nFixed input    : {EMBEDDING_DIM}-dim vector, norm={np.linalg.norm(fixed_input):.4f}")

    results = {}
    for label, bridge in [
        ("Deterministic (CB2 ✅)", DeterministicContextBridge()),
        ("Stochastic    (CB2 ❌)", StochasticContextBridge(sigma=NOISE_SIGMA)),
    ]:
        print(f"\n{SEPARATOR}")
        print(f"Bridge: {label}")
        print(f"{SEPARATOR}")
        r = run_replay_experiment(bridge, fixed_input)
        results[label] = r

        pass_rate = 100 * r.passes / N_REPLAYS
        print(f"  HMAC verified  : {r.passes}/{N_REPLAYS} ({pass_rate:.1f}%)")
        print(f"  HMAC failed    : {r.fails}/{N_REPLAYS} ({100*r.fails/N_REPLAYS:.1f}%)")
        print(f"  Output variance: {r.output_variance:.2e} (mean std across dims)")

        if r.fails == 0:
            print(f"  ✅ Every replay matches the original seal.")
            print(f"     Art. 12 audit record: INTACT — inspector can verify any step.")
        else:
            print(f"  ❌ {r.fails} replays produced outputs that don't match the original seal.")
            print(f"     Art. 12 audit record: CORRUPTED — replay ≠ original execution.")
            print(f"     An inspector cannot verify what the bridge produced at step N.")
            print(f"     Noise sigma={NOISE_SIGMA} is sufficient to break the audit trail.")

    print(f"\n{SEPARATOR}")
    print("SUMMARY TABLE — CB2 VALIDATION")
    print(f"{SEPARATOR}")
    print(f"{'Bridge':<28} {'HMAC pass rate':>16} {'Output variance':>18}")
    print("-" * 65)
    for label, r in results.items():
        pass_rate = f"{r.passes}/{N_REPLAYS} ({100*r.passes/N_REPLAYS:.1f}%)"
        print(f"  {label:<26} {pass_rate:>16} {r.output_variance:>18.2e}")

    det = results["Deterministic (CB2 ✅)"]
    sto = results["Stochastic    (CB2 ❌)"]

    print(f"\n{'─' * 65}")
    print("VERDICT")
    print(f"{'─' * 65}")
    if det.passes == N_REPLAYS and sto.fails == N_REPLAYS:
        print(f"  ✅ CB2 CONFIRMED")
        print(f"     Deterministic bridge: 100% HMAC pass rate — audit record intact.")
        print(f"     Stochastic bridge   : 0% HMAC pass rate — audit record corrupted.")
        print(f"     Noise sigma {NOISE_SIGMA} is sufficient to break every seal.")
        print(f"     CB2 is not a stylistic constraint — it is the physical")
        print(f"     precondition for Art. 12 audit integrity.")
    elif det.passes == N_REPLAYS:
        print(f"  ⚠️  Partial: deterministic is clean but stochastic bridge partially verifies.")
        print(f"     Increase NOISE_SIGMA and re-run.")
    else:
        print(f"  ❌ UNEXPECTED: deterministic bridge failed HMAC — check seal/verify logic.")

    print(f"\n{'─' * 65}")
    print("PAPER INSIGHT")
    print(f"{'─' * 65}")
    print("""
  CB2 is unique among the 33 invariants: it is the only one whose
  violation is detectable not by observing routing behaviour, but by
  observing the audit record.

  A stochastic bridge produces outputs that look correct at inference
  time — the downstream node receives plausible latent vectors. The
  corruption is invisible to the routing spine. It only becomes visible
  at replay: the signed record no longer matches what the bridge would
  produce from the same input.

  This means CB2 violation is a silent audit failure, not a runtime
  failure. A CB2-violating system passes all functional tests and fails
  only at the regulatory inspection.

  CB1 violation kills routing correctness. CB2 violation kills auditability.
  Both are required. Neither is sufficient alone.
""")


if __name__ == "__main__":
    main()
