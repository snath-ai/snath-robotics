"""
ProprioceptiveEncoder — Stream B: body physics.
================================================
Encodes IMU + joint sensor + tactile data into z_proprio ∈ ℝ^D.
Stream B in the V1–V6 dual-stream routing contract.

Architectural role
------------------
The proprioceptive stream answers: "what does the body actually feel?"
It is the physics ground truth — what the accelerometers, gyroscopes,
joint torque sensors, and fingertip pressure sensors report, independent
of what the visual scene suggests.

The canonical disagreement scenario (ice slip):
  z_vision  → floor looks flat and safe (visual appearance)
  z_proprio → friction is zero, accelerometer reports slip onset
  D = ||softmax(z_vision) − softmax(z_proprio)||₁ / √G  ≫  τ_high
  → STRUCTURAL_IMPASSE → brace

LoRA injection — motor degradation compensation
-----------------------------------------------
When a motor degrades, z_proprio carries a characteristic asymmetry
(one joint's torque distribution shifts). The overnight DMN cycle trains
a LoRA adapter on historical degradation D_hard events that compensates
the encoder's projection for that specific joint's failure signature.
Load via load_lora() — gated by temporal trust W >= min_trust.

Derivative Works note
---------------------
This file is a Derivative Work of AbstractLatentFaultLocator (I1–I6) and
AbstractDivergenceRouter (V1–V6), Apache 2.0,
github.com/snath-ai/Lar-JEPA (genesis v6.1, Jun 1 2026).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProprioceptiveEncoder(nn.Module):
    """
    Proprioceptive encoder: IMU + joint sensors + tactile. Stream B.

    Args:
        imu_dim:     Dimension of raw IMU + joint data (accelerometer,
                     gyroscope, joint angles, torques). Default 64.
        tactile_dim: Dimension of raw tactile / pressure data. Default 32.
        embed_dim:   Shared latent space dimension D. Must match VisionEncoder.
                     Default 768.
    """

    def __init__(self, imu_dim: int = 64, tactile_dim: int = 32,
                 embed_dim: int = 8):
        super().__init__()
        self.embed_dim   = embed_dim
        input_dim        = imu_dim + tactile_dim

        # Concept projection to low-dimensional concept space — same
        # rationale as VisionEncoder. Two-layer MLP kept for expressivity
        # on the sensor-fusion task; final output is concept_dim=8.
        self.proj = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        nn.init.xavier_uniform_(self.proj[0].weight)
        nn.init.zeros_(self.proj[0].bias)
        nn.init.xavier_uniform_(self.proj[2].weight)
        nn.init.zeros_(self.proj[2].bias)

    def forward(self, imu: torch.Tensor,
                tactile: torch.Tensor) -> torch.Tensor:
        """
        Args:
            imu:     (B, imu_dim) IMU + joint sensor readings.
            tactile: (B, tactile_dim) tactile / pressure readings.
        Returns:
            z_proprio: (B, embed_dim) normalised latent.
        """
        x = torch.cat([imu, tactile], dim=-1)
        return F.normalize(self.proj(x), dim=-1)

    def load_lora(self, pt_path: str) -> None:
        """
        Apply a signed LoRA delta to compensate for a hardware failure.

        Typical use: a motor degradation adapter trained on D_hard events
        from a specific joint failure signature. Perishable — gated by
        temporal trust W >= min_trust before injection.

        Args:
            pt_path: Path to the signed .pt adapter file.
        """
        payload = torch.load(pt_path, map_location="cpu", weights_only=False)
        A = payload["A"]   # (embed_dim, rank)
        B = payload["B"]   # (rank, embed_dim)
        with torch.no_grad():
            # Apply to final linear layer (proj[2]) — concept projection layer
            self.proj[2].weight.data += (A @ B)
