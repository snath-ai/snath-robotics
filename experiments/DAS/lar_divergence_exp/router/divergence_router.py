"""
AbstractDivergenceRouter — concrete implementation for the NLM CXR experiment.

Invariants V1--V6 as specified in:
    Sajeev, A.V. (2026). Divergence Is Not Noise.
    DOI: 10.5281/zenodo.20278781

Stream A: Vision encoder (ViT-B/16 on chest X-ray image)
Stream B: Language encoder (BioBERT-base on radiology report)
Divergence: Cosine distance between L2-normalised latent embeddings
"""

import os
import sys

# Path bootstrap — top-level JEPA_Playground/lar_jepa/ (canonical v2.3.0)
_HERE     = os.path.dirname(os.path.abspath(__file__))
_LAR_JEPA = os.path.join(_HERE, "..", "..", "..", "lar_jepa")
_LAR_JEPA = os.path.normpath(_LAR_JEPA)
if _LAR_JEPA not in sys.path:
    sys.path.insert(0, _LAR_JEPA)

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from core import AbstractDivergenceRouter
from core.types import RouteDecision


@dataclass
class RouterOutput:
    decision: RouteDecision
    confidence_a: float
    confidence_b: float
    divergence: float
    # Which stream was deferred to (only set when decision is Defer → COMMIT_TRAJECTORY)
    deferred_to: str = ""


class CXRDivergenceRouter(AbstractDivergenceRouter):
    """
    Concrete router for the NLM CXR biomedical experiment.

    Stream A: ViT-B/16 vision encoder on chest X-ray image.
    Stream B: BioBERT-base language encoder on radiology report.

    Confidence = max-softmax of the classification head output,
    calibrated via temperature scaling (temperature=1.5 default).

    Routing aligns with the canonical four rules from the DAS paper:
        Execute     → COMMIT_TRAJECTORY  (both high confidence, low divergence)
        Investigate → TRIGGER_REPLAN     (both high confidence, high divergence)
        Defer       → COMMIT_TRAJECTORY  (one stream confident — use that stream)
        Halt        → STRUCTURAL_IMPASSE (both uncertain)
    """

    def __init__(
        self,
        vision_encoder,
        language_encoder,
        tau_high: float = 0.8,
        tau_low: float = 0.3,
        delta: float = 0.5,
        temperature: float = 1.5,
        device: str = "mps",
    ):
        self.vision_encoder = vision_encoder
        self.language_encoder = language_encoder
        self.tau_high = tau_high
        self.tau_low = tau_low
        self.delta = delta
        self.temperature = temperature
        self.device = device

    @torch.no_grad()
    def encode_stream_a(self, image: torch.Tensor) -> tuple[torch.Tensor, float]:
        """V1 (Stream Independence): Encode chest X-ray image → (CLS latent, confidence)."""
        outputs = self.vision_encoder(pixel_values=image.to(self.device))
        latent = outputs.last_hidden_state[:, 0, :]
        logits = outputs.last_hidden_state.mean(dim=1)
        probs = F.softmax(logits / self.temperature, dim=-1)
        confidence = float(probs.max(dim=-1).values.mean().item())
        return latent, min(max(confidence, 0.0), 1.0)

    @torch.no_grad()
    def encode_stream_b(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, float]:
        """V1 (Stream Independence): Encode radiology report → (CLS latent, confidence)."""
        outputs = self.language_encoder(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
        )
        latent = outputs.last_hidden_state[:, 0, :]
        logits = outputs.last_hidden_state.mean(dim=1)
        probs = F.softmax(logits / self.temperature, dim=-1)
        confidence = float(probs.max(dim=-1).values.mean().item())
        return latent, min(max(confidence, 0.0), 1.0)

    def divergence(self, z_a: torch.Tensor, z_b: torch.Tensor) -> float:
        """V2 + V3: Cosine distance ∈ [0, 2]. D(z,z)=0. Asymmetry allowed (V3)."""
        z_a_norm = F.normalize(z_a.float(), dim=-1)
        z_b_norm = F.normalize(z_b.float(), dim=-1)
        cosine_sim = (z_a_norm * z_b_norm).sum(dim=-1).clamp(-1.0, 1.0)
        return (1.0 - cosine_sim).mean().item()

    def route(
        self,
        confidence_a: float,
        confidence_b: float,
        divergence: float,
    ) -> RouteDecision:
        """
        V4 (Content Blindness): receives only scalars — no access to z_a or z_b.
        V5 (Routing Completeness): returns exactly one RouteDecision.
        V6 (Safety-Learning Equivalence): STRUCTURAL_IMPASSE = max learning signal.
        """
        tau_h, tau_l, delta = self.tau_high, self.tau_low, self.delta
        both_high = confidence_a >= tau_h and confidence_b >= tau_h
        both_low  = confidence_a < tau_l  and confidence_b < tau_l

        if both_high and divergence < delta:
            return RouteDecision.COMMIT_TRAJECTORY   # Execute

        if both_high and divergence >= delta:
            return RouteDecision.TRIGGER_REPLAN      # Investigate

        if both_low:
            return RouteDecision.STRUCTURAL_IMPASSE  # Halt

        return RouteDecision.COMMIT_TRAJECTORY       # Defer (one stream confident)

    def forward(self, x_a, x_b) -> RouterOutput:
        """Full routing pass. Returns RouterOutput with all scalars for D_hard construction."""
        z_a, c_a = self.encode_stream_a(x_a)
        z_b, c_b = self.encode_stream_b(x_b)
        D = self.divergence(z_a, z_b)
        decision = self.route(c_a, c_b, D)

        deferred = ""
        if decision == RouteDecision.COMMIT_TRAJECTORY:
            # Distinguish Execute (both high) from Defer (one confident)
            both_high = c_a >= self.tau_high and c_b >= self.tau_high
            if not both_high:
                deferred = "A" if c_a >= self.tau_high else "B"

        return RouterOutput(
            decision=decision,
            confidence_a=c_a,
            confidence_b=c_b,
            divergence=D,
            deferred_to=deferred,
        )
