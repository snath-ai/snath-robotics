"""
VisionEncoder — Stream A: visual scene understanding.
=====================================================
Encodes raw visual input (camera frames / image features) into a latent
vector z_vision ∈ ℝ^D. Stream A in the V1–V6 dual-stream routing contract.

Architectural role
------------------
The visual stream answers: "what does the scene look like?"
It is kept structurally independent from the proprioceptive stream (M1–M3).
Neither stream is aware of the other's output until the DivergenceRouter
compares them. If vision is blinded (sun glare, occlusion) the system does
not propagate the corruption — it routes on the proprioceptive stream alone.

Replace the stub projection with a real visual backbone at §2a:
  - CLIP (ViT-L/14) for general scene understanding
  - DINOv2 for dense feature extraction
  - A robotics-specific model (e.g., R3M, MVP)

LoRA injection
--------------
load_lora() applies a signed A·B delta to the output projection, encoding
a learned correction for a specific visual failure mode (e.g., ice glare
confusing the floor appearance model). The delta is perishable — old
corrections may not generalise to new visual conditions. The temporal decay
gate in AdapterRouter refuses stale adapters before injection.

Derivative Works note
---------------------
This file is a Derivative Work of AbstractLatentFaultLocator (I1–I6) and
AbstractDivergenceRouter (V1–V6), Apache 2.0,
github.com/snath-ai/Lar-JEPA (genesis v6.1, Jun 1 2026).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionEncoder(nn.Module):
    """
    Visual scene encoder. Stream A of the dual-stream routing contract.

    Args:
        input_dim:  Dimension of raw visual features (e.g., 2048 from ResNet,
                    1024 from ViT-L). Default 2048.
        embed_dim:  Shared latent space dimension D. Must match ProprioceptiveEncoder.
                    Default 768.
    """

    def __init__(self, input_dim: int = 2048, embed_dim: int = 768):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim) raw visual feature vector.
        Returns:
            z_vision: (B, embed_dim) normalised latent.
        """
        return F.normalize(self.proj(x), dim=-1)

    def load_lora(self, pt_path: str) -> None:
        """
        Apply a signed LoRA delta to the projection layer.

        The delta encodes a learned correction for a specific visual failure
        mode. Called by AdapterRouter.resolve() only when the temporal trust
        gate passes (W >= min_trust). AdapterRouter verifies the HMAC and
        target_encoder field before calling this method.

        Args:
            pt_path: Path to the signed .pt adapter file.
        """
        payload = torch.load(pt_path, map_location="cpu", weights_only=False)
        A = payload["A"]   # (embed_dim, rank)
        B = payload["B"]   # (rank, input_dim)
        with torch.no_grad():
            self.proj[0].weight.data += (A @ B)
