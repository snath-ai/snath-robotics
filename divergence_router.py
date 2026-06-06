"""
DivergenceRouter — V1–V6 sensor-stream routing for Snath Robotics.
===================================================================
Measures total-variation distance between the visual and proprioceptive
probability distributions and routes to one of four decisions:

    COMMIT_TRAJECTORY   — streams agree, proceed with motion plan
    TRIGGER_REPLAN      — recoverable contradiction, load adapter and retry
    STRUCTURAL_IMPASSE  — irreconcilable, drop to physics-safe fallback
    DEFER               — one stream is uncertain, lean on the confident arm

V1–V6 invariants (from AbstractDivergenceRouter, github.com/snath-ai/Lar-JEPA):
  V1  Both streams present at every call.
  V2  Divergence computed from normalised probability vectors only.
  V3  Decision is a pure function of (D, conf_a, conf_b, δ, τ_high, τ_low).
  V4  Content-blind: the router never reads z_vision or z_proprio directly.
      It operates on the scalar divergence and confidence values only.
  V5  STRUCTURAL_IMPASSE is always reachable regardless of stream content.
  V6  COMMIT_TRAJECTORY is only returned when D < τ_low AND both conf ≥ τ_low.

Divergence metric
-----------------
    p_a = softmax(z_vision)       # (G,) probability vector
    p_b = softmax(z_proprio)      # (G,) probability vector
    Δ   = p_a − p_b               # (G,) signed delta
    D   = ||Δ||₁ / √G             # total variation / √dims

Division by √G normalises for embedding dimension so τ_high/τ_low are
comparable across domains (identical to Basis / Aviation / Research).
"""

import math
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple

from core.types import RouteDecision
from dhard import DHardQueue, RoboticsDHardEvent


@dataclass
class RoutingResult:
    decision:      RouteDecision
    divergence:    float
    delta:         torch.Tensor   # (G,) probability-space delta — stored in D_hard
    conf_vision:   float
    conf_proprio:  float
    note:          str


class DivergenceRouter:
    """
    V1–V6 routing contract implementation for humanoid sensor fusion.

    Args:
        tau_high:   Divergence threshold above which TRIGGER_REPLAN fires.
        tau_low:    Divergence threshold below which COMMIT_TRAJECTORY is safe.
        delta:      Minimum divergence for a D_hard event to be curriculum-worthy.
        dhard:      Optional D_hard queue. When provided, qualifying events are
                    written on every TRIGGER_REPLAN / STRUCTURAL_IMPASSE call.
    """

    def __init__(
        self,
        tau_high: float    = 0.60,
        tau_low:  float    = 0.25,
        delta:    float    = 0.35,
        dhard:    DHardQueue | None = None,
    ):
        self.tau_high = tau_high
        self.tau_low  = tau_low
        self.delta    = delta
        self.dhard    = dhard

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(
        self,
        z_vision:  torch.Tensor,   # (G,) or (1, G)
        z_proprio: torch.Tensor,   # (G,) or (1, G)
        scenario_id: str = "",
    ) -> RoutingResult:
        """
        Compute divergence and return a routing decision.

        V4 — content-blind: this method never branches on the values inside
        z_vision or z_proprio, only on the scalar D and confidence values.
        """
        z_a = z_vision.flatten()
        z_b = z_proprio.flatten()
        G   = z_a.shape[0]

        # Probability vectors (V2)
        p_a = F.softmax(z_a, dim=0)
        p_b = F.softmax(z_b, dim=0)

        # Total variation distance, normalised by √G
        delta_vec = p_a - p_b
        D         = float(delta_vec.abs().sum() / math.sqrt(G))

        # Confidence: peakedness of the softmax distribution.
        # (max(p) - 1/G) / (1 - 1/G) — 0 when uniform, 1 when fully peaked.
        # Correct for concept distributions; sigmoid-mean is always constant
        # for softmax outputs since mean(p) = 1/G regardless of peakedness.
        conf_a = float(max(0.0, (float(p_a.max()) - 1.0/G) / (1.0 - 1.0/G)))
        conf_b = float(max(0.0, (float(p_b.max()) - 1.0/G) / (1.0 - 1.0/G)))

        decision, note = self._decide(D, conf_a, conf_b)

        # Write to D_hard queue if this event is curriculum-worthy (V3)
        if self.dhard is not None and decision in (
            RouteDecision.TRIGGER_REPLAN,
            RouteDecision.STRUCTURAL_IMPASSE,
        ) and D >= self.delta:
            failure_class = self._infer_failure_class(conf_a, conf_b)
            ev = RoboticsDHardEvent(
                z_vision=z_a.tolist(),
                z_proprio=z_b.tolist(),
                divergence=D,
                decision=decision.value,
                failure_class=failure_class,
                scenario_id=scenario_id,
            )
            self.dhard.push(ev)

        return RoutingResult(
            decision=decision,
            divergence=D,
            delta=delta_vec,
            conf_vision=conf_a,
            conf_proprio=conf_b,
            note=note,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _decide(
        self, D: float, conf_a: float, conf_b: float
    ) -> Tuple[RouteDecision, str]:
        """Pure routing function. V3: function of scalars only."""
        both_confident = (conf_a >= self.tau_low) and (conf_b >= self.tau_low)

        if D < self.tau_low and both_confident:
            return (RouteDecision.COMMIT_TRAJECTORY,
                    f"streams agree (D={D:.3f} < τ_low={self.tau_low})")

        if not both_confident:
            low_stream = "vision" if conf_a < conf_b else "proprio"
            return (RouteDecision.DEFER,
                    f"stream '{low_stream}' uncertain "
                    f"(conf_a={conf_a:.2f}, conf_b={conf_b:.2f}) — deferring")

        if D >= self.tau_high:
            return (RouteDecision.STRUCTURAL_IMPASSE,
                    f"irreconcilable sensor contradiction "
                    f"(D={D:.3f} ≥ τ_high={self.tau_high}) — brace")

        return (RouteDecision.TRIGGER_REPLAN,
                f"recoverable contradiction "
                f"(τ_low={self.tau_low} ≤ D={D:.3f} < τ_high={self.tau_high})")

    def _infer_failure_class(self, conf_a: float, conf_b: float) -> str:
        """
        Heuristic failure class assignment for D_hard events.
        A low-confidence visual stream with a confident proprioceptive stream
        suggests an environmental transient (glare, occlusion).
        A low-confidence proprioceptive stream suggests hardware/sensor drift.
        """
        if conf_a < 0.50 and conf_b >= 0.70:
            return "environmental_transient"
        if conf_b < 0.50 and conf_a >= 0.70:
            return "hardware_structural"
        return "default"
