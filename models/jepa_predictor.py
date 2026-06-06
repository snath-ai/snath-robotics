"""
Snath Robotics — JEPA Predictor (world model loop closure).
============================================================
Predicts z_proprio (what the body should feel) from z_vision (what the
scene looks like). This closes the annotation loop:

    camera → VisionEncoder → z_vision
                                  ↓
                              JEPAPredictor → ẑ_proprio
                                  ↓
    sensors → ProprioEncoder → z_proprio
                                  ↓
              D_pred = 1 − cos(ẑ_proprio, z_proprio)  [stop-gradient]

Training signal = prediction error. No labels. Physics provides the
ground truth: if the floor looks icy, the body should feel low friction.
When it doesn't predict that correctly, it learns from the discrepancy.

This is LeCun's JEPA claim applied concretely: the robot simulates
physical consequences in latent space before committing to an action.
High prediction error → replan before the slip becomes a fall.

Derivative Works note
---------------------
Extends AbstractDivergenceRouter (V1–V6) and JEPA_DMN_Consolidation_Node,
Apache 2.0, github.com/snath-ai/Lar-JEPA (genesis v6.1, Jun 2026).
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


class JEPAPredictor(nn.Module):
    """
    Vision-to-proprioception predictor: z_vision → ẑ_proprio.

    Learns: given what the scene looks like, what should the body feel?

    Architecture: residual MLP with LayerNorm.
    Loss: 1 − cos(ẑ_proprio, sg(z_proprio))  — stop-gradient prevents collapse.

    High prediction error means the robot's body is not responding the way
    the visual scene suggested it should. That is the safety signal.

    Args:
        embed_dim:   Latent space dimension. Must match both encoder embed_dims.
        hidden_mult: Hidden layer width as multiple of embed_dim.
    """

    def __init__(self, embed_dim: int = 8, hidden_mult: int = 4):
        super().__init__()
        hidden = embed_dim * hidden_mult
        self.embed_dim = embed_dim
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )
        self.skip = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, z_vision: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_vision: (B, embed_dim) normalised vision latent.
        Returns:
            ẑ_proprio: (B, embed_dim) predicted proprioception latent.
        """
        return self.net(z_vision) + self.skip(z_vision)

    def prediction_loss(
        self, z_vision: torch.Tensor, z_proprio: torch.Tensor
    ) -> torch.Tensor:
        """
        JEPA cosine loss to stop-gradient target.

        Stop-gradient on z_proprio is the key: the predictor learns to match
        physics without the physics encoder collapsing to match the predictor.
        """
        z_hat   = self.forward(z_vision)
        target  = z_proprio.detach()                   # stop-gradient
        z_hat_n = F.normalize(z_hat,   dim=-1)
        tgt_n   = F.normalize(target,  dim=-1)
        return 1.0 - (z_hat_n * tgt_n).sum(dim=-1).mean()

    def prediction_error(
        self, z_vision: torch.Tensor, z_proprio: torch.Tensor
    ) -> torch.Tensor:
        """
        Per-sample prediction error ∈ [0, 2] (no grad).

        0.0 = perfectly predicted (scene and body agree)
        2.0 = maximally wrong (scene implies opposite of what body reports)

        Use this as a pre-routing safety signal: high error fires BEFORE
        the divergence router sees the mismatch, giving the system one
        extra inference step to replan.
        """
        with torch.no_grad():
            z_hat   = self.forward(z_vision)
            z_hat_n = F.normalize(z_hat,    dim=-1)
            z_prp_n = F.normalize(z_proprio, dim=-1)
            return 1.0 - (z_hat_n * z_prp_n).sum(dim=-1)

    def train_on_batch(
        self,
        z_vision:  torch.Tensor,
        z_proprio: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        optimizer.zero_grad()
        loss = self.prediction_loss(z_vision, z_proprio)
        loss.backward()
        optimizer.step()
        return float(loss.item())


def train_predictor(
    predictor:  JEPAPredictor,
    z_vision:   torch.Tensor,
    z_proprio:  torch.Tensor,
    n_epochs:   int   = 200,
    lr:         float = 1e-3,
    batch_size: int   = 32,
) -> dict:
    """
    Train predictor on accumulated (z_vision, z_proprio) pairs.

    No labels. Physics is the supervision: prediction error = discrepancy
    between what the scene implied and what the sensors reported.

    In practice, call this after any session where the robot accumulated
    D_hard events (TRIGGER_REPLAN or STRUCTURAL_IMPASSE). Those hard cases
    are exactly the training signal — normal operation generates easy pairs,
    anomalous scenarios generate hard ones that improve the predictor fastest.

    Args:
        predictor:  JEPAPredictor on the correct device.
        z_vision:   (N, embed_dim) accumulated visual latents.
        z_proprio:  (N, embed_dim) corresponding proprioceptive latents.
        n_epochs:   Training epochs.
        lr:         AdamW learning rate.
        batch_size: Mini-batch size.

    Returns:
        dict: error_before, error_after, loss_final.
    """
    device    = next(predictor.parameters()).device
    z_vision  = z_vision.to(device).detach()
    z_proprio = z_proprio.to(device).detach()
    N         = z_vision.size(0)

    optimizer = torch.optim.AdamW(
        predictor.parameters(), lr=lr, weight_decay=1e-4
    )

    with torch.no_grad():
        err_before = float(predictor.prediction_error(z_vision, z_proprio).mean())

    predictor.train()
    loss_val = 0.0
    for epoch in range(n_epochs):
        idx = torch.randperm(N, device=device)
        for start in range(0, N, batch_size):
            b        = idx[start : start + batch_size]
            loss_val = predictor.train_on_batch(z_vision[b], z_proprio[b], optimizer)
        if (epoch + 1) % 50 == 0:
            log.info(f"  predictor epoch {epoch+1:3d}/{n_epochs}  loss={loss_val:.4f}")

    predictor.eval()
    with torch.no_grad():
        err_after = float(predictor.prediction_error(z_vision, z_proprio).mean())

    log.info(f"  Predictor: error {err_before:.4f} → {err_after:.4f}")
    return {
        "error_before": err_before,
        "error_after":  err_after,
        "loss_final":   loss_val,
    }
