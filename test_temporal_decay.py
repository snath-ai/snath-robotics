"""
Temporal decay regression tests — Snath Robotics.

Tests the W = exp(-λ · Δt) gate across all three failure classes and
confirms the identification/correction trust asymmetry:
  - System 1 centroid match fires regardless of adapter age.
  - System 2 LoRA injection is refused when W < min_trust.
"""

import math
import sys
import os
import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dmn.adapter_router import _decay_weight, _LAMBDA, RoboticsAdapterRouter
from core.types import RouteDecision


def _iso(years_ago: float) -> str:
    dt = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
        days=years_ago * 365.25
    )
    return dt.isoformat()


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_environmental_transient_fast_decay():
    """Ice / glare adapters age quickly: W < 0.40 after ~1.8 years."""
    W_fresh = _decay_weight(_iso(0.0), "environmental_transient")
    W_stale = _decay_weight(_iso(2.0), "environmental_transient")
    assert W_fresh > 0.99, f"fresh adapter should be trusted: {W_fresh}"
    assert W_stale < 0.40, f"2-year-old env adapter should be stale: {W_stale}"


def test_hardware_structural_slow_decay():
    """Motor-wear adapters are durable: W > 0.90 after 5 years."""
    W_5yr = _decay_weight(_iso(5.0), "hardware_structural")
    assert W_5yr > 0.90, f"hardware adapter should still be trusted after 5yr: {W_5yr}"


def test_sensor_drift_medium_decay():
    """Calibration drift adapters: intermediate decay (λ=0.20)."""
    W_1yr = _decay_weight(_iso(1.0), "sensor_drift")
    expected = math.exp(-0.20 * 1.0)
    assert abs(W_1yr - expected) < 1e-4, f"W mismatch: {W_1yr} vs {expected}"


def test_missing_timestamp_returns_one():
    """No created_at → W = 1.0 (treat as freshly minted)."""
    assert _decay_weight(None) == 1.0
    assert _decay_weight("")   == 1.0


def test_min_trust_floor():
    """min_trust=0.40 is the injection gate across all three repos."""
    W_just_above = _decay_weight(_iso(1.3), "environmental_transient")
    W_just_below = _decay_weight(_iso(1.9), "environmental_transient")
    assert W_just_above > 0.40, "adapter just above threshold should be injectable"
    assert W_just_below < 0.40, "adapter just below threshold should be refused"


def test_system1_trust_invariant():
    """
    System 1 centroid match fires regardless of adapter age.
    RoboticsAdapterRouter._nearest() carries NO temporal gate.
    Only resolve() checks W before System 2 injection.
    """
    import inspect
    src = inspect.getsource(RoboticsAdapterRouter._nearest)
    assert "_decay_weight" not in src, (
        "System 1 must be trust-invariant — _decay_weight should not appear "
        "in _nearest(). The temporal gate belongs in resolve() only."
    )


def test_system2_refuses_stale():
    """
    resolve() returns a STALE note when W < min_trust and does NOT call
    load_lora() on the encoder.
    """
    import json, tempfile, pathlib

    class _MockEnc:
        def __init__(self): self.lora_loaded = False
        def load_lora(self, _): self.lora_loaded = True

    # Write a centroid JSON and a fake .pt that looks stale (2 years old)
    with tempfile.TemporaryDirectory() as td:
        import hmac as _hmac, hashlib
        stale_ts = _iso(2.0)
        # 8-dim concept vectors — matches new embed_dim=8
        centroid = [0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2]
        _KEY = b"snath_robotics_adapter_sovereignty_2026"
        immutable = {
            "failure_class":    "environmental_transient",
            "centroid_vision":  centroid,
            "centroid_proprio": [0.0] * 8,
            "winner":           "vision",
            "win_rate":         0.9,
            "n_events":         10,
        }
        sig = _hmac.new(_KEY, json.dumps(immutable, sort_keys=True).encode(),
                        hashlib.sha256).hexdigest()
        cjson = {**immutable, "created_at": stale_ts, "sig": sig}
        pathlib.Path(td, "environmental_transient.json").write_text(json.dumps(cjson))

        import torch
        A = torch.zeros(8, 1)
        B = torch.zeros(1, 8)
        a_hash = hashlib.sha256(A.numpy().tobytes()).hexdigest()[:16]
        b_hash = hashlib.sha256(B.numpy().tobytes()).hexdigest()[:16]
        pt_sig = _hmac.new(_KEY,
            f"environmental_transient|vision|{a_hash}|{b_hash}".encode(),
            hashlib.sha256).hexdigest()
        pt_payload = {
            "A": A, "B": B,
            "target_encoder": "vision",
            "failure_class":  "environmental_transient",
            "created_at":     stale_ts,
            "hmac_hex":       pt_sig,
        }
        torch.save(pt_payload, os.path.join(td, "environmental_transient.pt"))

        enc_v = _MockEnc()
        enc_p = _MockEnc()
        ar = RoboticsAdapterRouter(adapter_dir=td, tau_sim=0.0, min_trust=0.40)
        import numpy as np
        _, note = ar.resolve(
            z_vision=np.array(centroid),
            z_proprio=np.array([0.0] * 8),
            base_decision=RouteDecision.TRIGGER_REPLAN,
            conf_vision=0.8, conf_proprio=0.8,
            enc_vision=enc_v, enc_proprio=enc_p,
        )
        assert "STALE" in note or "System 1 only" in note, f"Expected stale note: {note}"
        assert not enc_v.lora_loaded, "load_lora must NOT be called for stale adapter"
        assert not enc_p.lora_loaded, "load_lora must NOT be called for stale adapter"


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_environmental_transient_fast_decay,
        test_hardware_structural_slow_decay,
        test_sensor_drift_medium_decay,
        test_missing_timestamp_returns_one,
        test_min_trust_floor,
        test_system1_trust_invariant,
        test_system2_refuses_stale,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  ✓  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  ✗  {t.__name__}: {e}")

    print(f"\n{passed}/{len(tests)} passed")
    sys.exit(0 if passed == len(tests) else 1)
