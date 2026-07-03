"""
Option B router — both streams are BioBERT embeddings.

Stream A: BioBERT(MeSH description)   — what the *image* says in text
Stream B: BioBERT(radiology report)   — what the *radiologist* wrote

All V1–V6 invariants from the DAS paper are preserved:
  V1 Stream Independence  — encode_stream_a/b share BioBERT weights (read-only,
                            not mutable state); separate calls, no shared buffers.
  V2 Geometric Divergence — cosine distance ∈ [0, 2] ⊆ ℝ≥0
  V3 Symmetry Breaking    — cosine distance is symmetric; asymmetric extensions allowed
  V4 Content Blindness    — route() receives only (c_a, c_b, D); never sees z_a or z_b
  V5 Routing Completeness — exactly one RouteDecision returned for all inputs
  V6 Safety-Learning      — STRUCTURAL_IMPASSE reachable; both_low branch present
"""

import os
import sys

_HERE     = os.path.dirname(os.path.abspath(__file__))
_LAR_JEPA = os.path.normpath(os.path.join(_HERE, "..", "..", "..", "lar_jepa"))
if _LAR_JEPA not in sys.path:
    sys.path.insert(0, _LAR_JEPA)

import torch
import torch.nn.functional as F

from core import AbstractDivergenceRouter
from core.types import RouteDecision


class CXRTextDivergenceRouter(AbstractDivergenceRouter):
    """
    Option B: both streams are BioBERT text embeddings.

    Routing maps to the four canonical rules (DAS paper §3):
        Execute     → COMMIT_TRAJECTORY   (both confident, low divergence)
        Investigate → TRIGGER_REPLAN      (both confident, high divergence)
        Defer       → COMMIT_TRAJECTORY   (one stream confident)
        Halt        → STRUCTURAL_IMPASSE  (both uncertain)
    """

    def __init__(
        self,
        bert_model,
        tau_high: float = 0.75,
        tau_low:  float = 0.30,
        delta:    float = 0.45,
        device:   str   = "mps",
    ):
        self.bert_model = bert_model
        self.tau_high = tau_high
        self.tau_low  = tau_low
        self.delta    = delta
        self.device   = device

    def encode_stream_a(self, latent_a: torch.Tensor, conf_a: float):
        """V1: Stream A already encoded — pass through."""
        return latent_a, conf_a

    def encode_stream_b(self, latent_b: torch.Tensor, conf_b: float):
        """V1: Stream B already encoded — pass through."""
        return latent_b, conf_b

    def divergence(self, z_a: torch.Tensor, z_b: torch.Tensor) -> float:
        """V2 + V3: Cosine distance ∈ [0, 2]. D(z,z)=0."""
        z_a_norm = F.normalize(z_a.float(), dim=-1)
        z_b_norm = F.normalize(z_b.float(), dim=-1)
        cosine_sim = (z_a_norm * z_b_norm).sum(dim=-1).clamp(-1.0, 1.0)
        return float((1.0 - cosine_sim).mean().item())

    def route(
        self,
        confidence_a: float,
        confidence_b: float,
        divergence:   float,
    ) -> RouteDecision:
        """
        V4 Content Blindness: only scalars — no z_a or z_b.
        V5 Routing Completeness: exactly one RouteDecision.
        V6 Safety-Learning: STRUCTURAL_IMPASSE = max learning signal.
        """
        tau_h, tau_l, delta = self.tau_high, self.tau_low, self.delta
        both_high = confidence_a >= tau_h and confidence_b >= tau_h
        both_low  = confidence_a <  tau_l and confidence_b <  tau_l

        if both_high and divergence < delta:
            return RouteDecision.COMMIT_TRAJECTORY   # Execute

        if both_high and divergence >= delta:
            return RouteDecision.TRIGGER_REPLAN      # Investigate

        if both_low:
            return RouteDecision.STRUCTURAL_IMPASSE  # Halt

        return RouteDecision.COMMIT_TRAJECTORY       # Defer
