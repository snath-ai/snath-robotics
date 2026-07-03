"""
Option A router — BiomedCLIP shared embedding space.

Both encoders produce L2-normalised 512-dim vectors in BiomedCLIP's
shared biomedical representation space. Cosine distance is semantically
calibrated: agreeing image-text pairs land near 0, contradicting pairs
land near 1.0+.

V1–V6 invariants all hold:
  V1 Stream Independence  — image and text encoders are separate model heads;
                            no mutable state shared between encode calls.
  V2 Geometric Divergence — cosine distance ∈ [0, 2] ⊆ ℝ≥0
  V3 Symmetry Breaking    — cosine distance is symmetric (allowed by V3)
  V4 Content Blindness    — route() sees only (c_a, c_b, D); never z_a or z_b
  V5 Routing Completeness — exactly one RouteDecision for all inputs
  V6 Safety-Learning      — STRUCTURAL_IMPASSE reachable via both_low branch
"""

import os, sys, torch, torch.nn.functional as F

_HERE     = os.path.dirname(os.path.abspath(__file__))
_LAR_JEPA = os.path.normpath(os.path.join(_HERE, "..", "..", "..", "lar_jepa"))
if _LAR_JEPA not in sys.path:
    sys.path.insert(0, _LAR_JEPA)

from core import AbstractDivergenceRouter
from core.types import RouteDecision


class BiomedCLIPDivergenceRouter(AbstractDivergenceRouter):
    """
    Option A: BiomedCLIP image + text encoders in shared 512-dim space.

    Routing canonical four rules (DAS paper §3):
        Execute     → COMMIT_TRAJECTORY   (both confident, D < δ — streams agree)
        Investigate → TRIGGER_REPLAN      (both confident, D ≥ δ — streams contradict)
        Defer       → COMMIT_TRAJECTORY   (one stream uncertain — trust the other)
        Halt        → STRUCTURAL_IMPASSE  (both uncertain — escalate)
    """

    def __init__(
        self,
        tau_high: float = 0.75,
        tau_low:  float = 0.20,
        delta:    float = 0.50,
        device:   str   = "mps",
    ):
        self.tau_high = tau_high
        self.tau_low  = tau_low
        self.delta    = delta
        self.device   = device

    def encode_stream_a(self, z_a: torch.Tensor, c_a: float):
        """V1: pass-through — encoding done externally to preserve V1."""
        return z_a, c_a

    def encode_stream_b(self, z_b: torch.Tensor, c_b: float):
        """V1: pass-through — encoding done externally to preserve V1."""
        return z_b, c_b

    def divergence(self, z_a: torch.Tensor, z_b: torch.Tensor) -> float:
        """V2 + V3: Cosine distance in BiomedCLIP shared space ∈ [0, 2]."""
        z_a_n = F.normalize(z_a.float(), dim=-1)
        z_b_n = F.normalize(z_b.float(), dim=-1)
        cos_sim = (z_a_n * z_b_n).sum(dim=-1).clamp(-1.0, 1.0)
        return float((1.0 - cos_sim).mean().item())

    def route(
        self,
        confidence_a: float,
        confidence_b: float,
        divergence:   float,
    ) -> RouteDecision:
        """
        V4 Content Blindness: only scalars — z_a and z_b never seen here.
        V5 Routing Completeness: exactly one RouteDecision returned.
        V6 Safety-Learning: STRUCTURAL_IMPASSE = max learning signal.
        """
        tau_h, tau_l, d = self.tau_high, self.tau_low, self.delta
        both_high = confidence_a >= tau_h and confidence_b >= tau_h
        both_low  = confidence_a <  tau_l and confidence_b <  tau_l

        if both_high and divergence < d:
            return RouteDecision.COMMIT_TRAJECTORY   # Execute

        if both_high and divergence >= d:
            return RouteDecision.TRIGGER_REPLAN      # Investigate

        if both_low:
            return RouteDecision.STRUCTURAL_IMPASSE  # Halt

        return RouteDecision.COMMIT_TRAJECTORY       # Defer
