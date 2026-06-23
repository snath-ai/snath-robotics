"""
GRU Proprioceptive Encoder — temporal sequence variant.

Encodes a rolling window of obs into a concept vector.
Gait-phase invariant (GRU aggregates over the full window),
terrain-type sensitive (height drop + velocity variance over 10 steps).

Used by the V1–V6 router as z_proprio Stream B.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUProprioEncoder(nn.Module):
    """
    Args:
        obs_dim:    Raw obs dimension (17 for Walker2d-v5).
        hidden_dim: GRU hidden state size.
        embed_dim:  Output concept dimension (must match VisionEncoder).
        seq_len:    Rolling window length.
    """
    def __init__(
        self,
        obs_dim:    int = 17,
        hidden_dim: int = 32,
        embed_dim:  int = 8,
        seq_len:    int = 10,
    ):
        super().__init__()
        self.obs_dim    = obs_dim
        self.hidden_dim = hidden_dim
        self.embed_dim  = embed_dim
        self.seq_len    = seq_len

        self.gru  = nn.GRU(obs_dim, hidden_dim, num_layers=1, batch_first=True)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        nn.init.xavier_uniform_(self.proj[0].weight)
        nn.init.zeros_(self.proj[0].bias)

    def forward(self, obs_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs_seq: (B, seq_len, obs_dim)
        Returns:
            z: (B, embed_dim) — raw LayerNorm output, no F.normalize
        """
        _, hidden = self.gru(obs_seq)       # hidden: (1, B, hidden_dim)
        return self.proj(hidden.squeeze(0)) # (B, embed_dim)
