"""
D_hard queue — Snath Robotics.
==============================
Records every sensor disagreement event that triggered TRIGGER_REPLAN or
STRUCTURAL_IMPASSE. Each event captures the latent vectors from both streams
at the moment of divergence, the routing decision, and (once resolved by a
human operator or ground-truth sensor) which stream was correct.

The D_hard curriculum (from DAS paper §4):
    D_hard = { i : Δᵢ ≥ δ  and  rᵢ = TRIGGER_REPLAN }

Events in this set become the training data for the overnight DMN
consolidation cycle, which generates signed LoRA adapters that
the AdapterRouter applies at the next inference.

Failure classes and temporal decay constants (config.json):
  environmental_transient  λ=0.50  ice, wet floor, sun glare
  hardware_structural      λ=0.02  motor wear, joint degradation
  sensor_drift             λ=0.20  gradual calibration error
  default                  λ=0.10

HMAC-SHA256 signed on write. AdapterRouter only trusts verified events.
"""

import json
import os
import hmac as _hmac
import hashlib
from dataclasses import dataclass, asdict, field
from typing import List, Optional


_DHARD_KEY = b"snath_robotics_dhard_sovereignty_2026"

_DHARD_DECISIONS = {"TRIGGER_REPLAN", "STRUCTURAL_IMPASSE"}


@dataclass
class RoboticsDHardEvent:
    """
    One sensor-disagreement event from the live routing pipeline.

    Fields set at event time:
      z_vision      Stream A latent at divergence
      z_proprio     Stream B latent at divergence
      divergence    D = ||Δ||₁ / √G (the routing trigger scalar)
      decision      The routing decision that created this event
      failure_class Temporal decay class (set by router heuristic)
      scenario_id   Human-readable label for this run / scenario

    Fields set post-hoc (overnight / by operator):
      winner        "vision" | "proprio" — which stream was correct
      hmac_hex      HMAC-SHA256 signature computed at write time
    """
    z_vision:      List[float]
    z_proprio:     List[float]
    divergence:    float
    decision:      str
    failure_class: str                = "default"
    scenario_id:   str                = ""
    winner:        Optional[str]      = None
    hmac_hex:      str                = ""

    def _canonical(self) -> bytes:
        return json.dumps({
            "z_vision":   [round(v, 6) for v in self.z_vision],
            "z_proprio":  [round(v, 6) for v in self.z_proprio],
            "divergence": round(self.divergence, 6),
            "decision":   self.decision,
            "failure_class": self.failure_class,
            "scenario_id":   self.scenario_id,
        }, sort_keys=True).encode()

    def sign(self) -> "RoboticsDHardEvent":
        self.hmac_hex = _hmac.new(
            _DHARD_KEY, self._canonical(), hashlib.sha256
        ).hexdigest()
        return self

    def verify(self) -> bool:
        expected = _hmac.new(
            _DHARD_KEY, self._canonical(), hashlib.sha256
        ).hexdigest()
        return _hmac.compare_digest(self.hmac_hex, expected)


class DHardQueue:
    """JSONL-backed D_hard event store."""

    def __init__(self, path: str = "d_hard.jsonl"):
        self.path = path

    def push(self, event: RoboticsDHardEvent) -> None:
        """Sign and append an event."""
        event.sign()
        with open(self.path, "a") as fh:
            fh.write(json.dumps(asdict(event)) + "\n")

    def all(self) -> List[RoboticsDHardEvent]:
        if not os.path.exists(self.path):
            return []
        events = []
        with open(self.path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        events.append(RoboticsDHardEvent(**json.loads(line)))
                    except Exception:
                        pass
        return events

    def verified(self) -> List[RoboticsDHardEvent]:
        return [e for e in self.all() if e.verify()]

    def resolved(self) -> List[RoboticsDHardEvent]:
        return [e for e in self.verified() if e.winner is not None]

    def stats(self) -> dict:
        all_e      = self.all()
        verified   = [e for e in all_e if e.verify()]
        resolved   = [e for e in verified if e.winner is not None]
        by_class   = {}
        for e in all_e:
            by_class[e.failure_class] = by_class.get(e.failure_class, 0) + 1
        return {
            "total":    len(all_e),
            "verified": len(verified),
            "resolved": len(resolved),
            "by_class": by_class,
            "path":     self.path,
        }
