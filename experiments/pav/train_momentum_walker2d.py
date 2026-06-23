"""
Momentum Encoder Training — Walker2d Terrain
=============================================
Trains ProprioceptiveEncoder using a momentum (EMA) target encoder,
VICReg covariance decorrelation, and obs augmentation.

Why momentum encoder beats JEPA cosine for this task:

  JEPA cosine loss   → encoder learns temporal SMOOTHNESS
                        adjacent obs produce similar embeddings regardless of terrain
                        ice falls are also smooth → no terrain-type separation

  Momentum encoder   → EMA target is more stable than stop-grad target
                        forces online encoder to produce representations that are
                        invariant to gait-phase noise but sensitive to terrain type
                        VICReg covariance decorrelates all 8 concept dimensions

  VICReg covariance  → off-diagonal covariance → 0
                        forces each dimension to encode something different
                        prevents collapse to 1-2 active dimensions

Training setup:
  - Online encoder: ProprioceptiveEncoder (trained by gradient)
  - Target encoder: EMA of online encoder, τ=0.995 (no gradients)
  - Predictor: 2-layer MLP online → target alignment
  - Augmentation: Gaussian obs noise (σ=0.05) — no terrain labels used
  - Loss: cosine(predictor(aug(z_online)), aug(z_target))
         + λ_var  * max(0, γ - std(z))       variance floor
         + λ_cov  * off_diag(cov(z))^2       covariance decorrelation

Expected after training:
  - Centroid angle normal vs ice: > 40°
  - Within-cluster distance: < 0.4
  - Router confidence: > 0.35
  - Per-step D on ice (vs normal centroid): > tau_high=0.60

Run:
    poetry run python experiments/train_momentum_walker2d.py
    poetry run python experiments/train_momentum_walker2d.py --epochs 200 --steps 120000
"""
from __future__ import annotations

import copy
import argparse
import sys
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

import gymnasium
from encoders.robotics.proprio_encoder import ProprioceptiveEncoder

IMU_DIM       = 64
TACTILE_DIM   = 32
EMBED_DIM     = 8
FRICTION_NORMAL = 0.80
FRICTION_ICE    = 0.05
MODEL_PATH    = _ROOT / "models" / "pav" / "proprio_momentum.pt"


# ══════════════════════════════════════════════════════════════════════════════
# Obs → encoder input
# ══════════════════════════════════════════════════════════════════════════════

def obs_to_encoder(obs: np.ndarray, noise_std: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    imu     = torch.zeros(1, IMU_DIM)
    tactile = torch.zeros(1, TACTILE_DIM)
    t = torch.from_numpy(obs).float()
    if noise_std > 0:
        t = t + torch.randn_like(t) * noise_std
    imu[0, :17]     = t
    tactile[0, :6]  = t[2:8]
    tactile[0, 6:12] = t[11:17]
    return imu, tactile


# ══════════════════════════════════════════════════════════════════════════════
# Predictor
# ══════════════════════════════════════════════════════════════════════════════

class Predictor(nn.Module):
    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.LayerNorm(embed_dim * 4),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ══════════════════════════════════════════════════════════════════════════════
# VICReg losses
# ══════════════════════════════════════════════════════════════════════════════

def variance_loss(z: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    std = z.std(dim=0)
    return F.relu(torch.tensor(gamma) - std).mean()


def covariance_loss(z: torch.Tensor) -> torch.Tensor:
    B, D = z.shape
    z_norm = z - z.mean(dim=0)
    cov = (z_norm.T @ z_norm) / (B - 1)
    off_diag = cov - torch.diag(cov.diag())
    return (off_diag ** 2).sum() / D


# ══════════════════════════════════════════════════════════════════════════════
# Rollout collection
# ══════════════════════════════════════════════════════════════════════════════

def collect_rollouts(n: int, friction: float, seed: int) -> list[tuple]:
    env  = gymnasium.make("Walker2d-v5")
    rng  = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    env.unwrapped.model.geom_friction[0, 0] = friction
    pairs, step = [], 0
    while len(pairs) < n:
        action   = rng.uniform(-0.4, 0.4, size=env.action_space.shape)
        obs_t    = obs.copy()
        obs_next, _, term, trunc, _ = env.step(action)
        pairs.append((obs_t, obs_next.copy()))
        obs  = obs_next
        step += 1
        if term or trunc or (step % 500 == 0):
            obs, _ = env.reset()
            env.unwrapped.model.geom_friction[0, 0] = friction
    env.close()
    return pairs


# ══════════════════════════════════════════════════════════════════════════════
# EMA update
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def ema_update(online: nn.Module, target: nn.Module, tau: float) -> None:
    for p_o, p_t in zip(online.parameters(), target.parameters()):
        p_t.data.mul_(tau).add_(p_o.data, alpha=1.0 - tau)


# ══════════════════════════════════════════════════════════════════════════════
# Training loop
# ══════════════════════════════════════════════════════════════════════════════

def train(
    online:    ProprioceptiveEncoder,
    target:    ProprioceptiveEncoder,
    predictor: Predictor,
    pairs:     list[tuple],
    epochs:    int,
    batch:     int,
    lr:        float,
    tau:       float,
    noise_std: float,
    lam_var:   float,
    lam_cov:   float,
) -> None:
    opt = torch.optim.Adam(
        list(online.parameters()) + list(predictor.parameters()), lr=lr,
        weight_decay=1e-4,
    )
    n = len(pairs)

    for epoch in range(epochs):
        idx   = np.random.permutation(n)
        total_cos = total_var = total_cov = 0.0
        steps = 0

        for start in range(0, n - batch, batch):
            bi = idx[start : start + batch]
            B  = len(bi)

            imu_t  = torch.zeros(B, IMU_DIM);  tac_t  = torch.zeros(B, TACTILE_DIM)
            imu_t1 = torch.zeros(B, IMU_DIM);  tac_t1 = torch.zeros(B, TACTILE_DIM)

            for i, idx_i in enumerate(bi):
                obs_t, obs_t1 = pairs[idx_i]
                im, ta = obs_to_encoder(obs_t,  noise_std)
                imu_t[i]  = im[0]; tac_t[i]  = ta[0]
                im1, ta1 = obs_to_encoder(obs_t1, noise_std)
                imu_t1[i] = im1[0]; tac_t1[i] = ta1[0]

            # Online path — gradients flow
            z_online = online(imu_t, tac_t)        # (B, D)
            z_pred   = predictor(z_online)          # (B, D)

            # Target path — no gradients
            with torch.no_grad():
                z_target = target(imu_t1, tac_t1)  # (B, D)

            # Cosine alignment loss (BYOL-style)
            cos_loss = (1.0 - F.cosine_similarity(z_pred, z_target, dim=-1)).mean()

            # VICReg regularisation on online embeddings
            v_loss = variance_loss(z_online, gamma=1.0)
            c_loss = covariance_loss(z_online)

            loss = cos_loss + lam_var * v_loss + lam_cov * c_loss

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(online.parameters(), 1.0)
            opt.step()

            # EMA target update
            ema_update(online, target, tau)

            total_cos += cos_loss.item()
            total_var += v_loss.item()
            total_cov += c_loss.item()
            steps += 1

        if (epoch + 1) % 20 == 0:
            std_v  = z_online.std(dim=0).mean().item()
            print(f"  epoch {epoch+1:3d}/{epochs}  "
                  f"cos={total_cos/steps:.4f}  "
                  f"var={total_var/steps:.4f}  "
                  f"cov={total_cov/steps:.4f}  "
                  f"std={std_v:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# Verification
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def verify(encoder: ProprioceptiveEncoder, n: int = 300) -> dict:
    env = gymnasium.make("Walker2d-v5")
    rng = np.random.default_rng(999)
    z_normal, z_ice = [], []

    for friction, zlist, seed in [
        (FRICTION_NORMAL, z_normal, 1000),
        (FRICTION_ICE,    z_ice,    2000),
    ]:
        obs, _ = env.reset(seed=seed)
        env.unwrapped.model.geom_friction[0, 0] = friction
        for _ in range(n):
            imu, tac = obs_to_encoder(obs)
            z = encoder(imu, tac).squeeze(0)
            zlist.append(z)
            action = rng.uniform(-0.4, 0.4, size=env.action_space.shape)
            obs, _, term, trunc, _ = env.step(action)
            if term or trunc:
                obs, _ = env.reset()
                env.unwrapped.model.geom_friction[0, 0] = friction
    env.close()

    zn = torch.stack(z_normal)
    zi = torch.stack(z_ice)

    def mean_cdist(a, b):
        sims = F.cosine_similarity(
            a.unsqueeze(1).expand(-1, len(b), -1).reshape(-1, a.shape[-1]),
            b.unsqueeze(0).expand(len(a), -1, -1).reshape(-1, b.shape[-1]),
            dim=-1,
        )
        return float(1 - sims.mean())

    def mean_conf(z):
        G = z.shape[-1]
        p = F.softmax(z, dim=-1)
        return float(((p.max(dim=-1).values - 1/G) / (1 - 1/G)).clamp(min=0).mean())

    within_n = mean_cdist(zn, zn)
    within_i = mean_cdist(zi, zi)
    between  = mean_cdist(zn, zi)

    mean_zn = F.normalize(zn.mean(0), dim=0)
    mean_zi = F.normalize(zi.mean(0), dim=0)
    angle   = math.degrees(math.acos(
        float(F.cosine_similarity(mean_zn.unsqueeze(0), mean_zi.unsqueeze(0)).clamp(-1, 1))
    ))

    # Also measure per-step D the router would see (z_vision = centroid of normal)
    z_vision = zn.mean(0)
    if z_vision.norm() > 0:
        z_vision_scaled = z_vision * (zn.std(dim=0).mean() / z_vision.norm().clamp(min=1e-6))
    else:
        z_vision_scaled = z_vision

    def router_D(z_v, z_p):
        p_a = F.softmax(z_v, dim=0)
        p_b = F.softmax(z_p, dim=0)
        return float((p_a - p_b).abs().sum() / math.sqrt(EMBED_DIM))

    d_normal = float(np.mean([router_D(z_vision_scaled, z) for z in z_normal[:100]]))
    d_ice    = float(np.mean([router_D(z_vision_scaled, z) for z in z_ice[:100]]))

    return {
        "within_normal":  round(within_n, 4),
        "within_ice":     round(within_i, 4),
        "between":        round(between,  4),
        "separation":     round(between - max(within_n, within_i), 4),
        "centroid_angle": round(angle, 1),
        "conf_normal":    round(mean_conf(zn), 4),
        "conf_ice":       round(mean_conf(zi), 4),
        "router_D_normal": round(d_normal, 4),
        "router_D_ice":    round(d_ice, 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main(steps: int, epochs: int, batch: int, lr: float, tau: float) -> None:
    print("\n" + "═" * 65)
    print("  Momentum Encoder Training — Walker2d Terrain")
    print("  EMA target · VICReg covariance · obs augmentation")
    print("═" * 65)

    half = steps // 2
    print(f"\n  Collecting {half} normal steps …")
    normal_pairs = collect_rollouts(half, FRICTION_NORMAL, seed=0)
    print(f"  Collecting {half} ice steps …")
    ice_pairs    = collect_rollouts(half, FRICTION_ICE, seed=1)
    pairs        = normal_pairs + ice_pairs
    np.random.default_rng(42).shuffle(pairs)
    print(f"  Total pairs: {len(pairs)}")

    online    = ProprioceptiveEncoder(IMU_DIM, TACTILE_DIM, EMBED_DIM)
    target    = copy.deepcopy(online)
    predictor = Predictor(EMBED_DIM)
    for p in target.parameters():
        p.requires_grad_(False)

    print(f"\n  Training: {epochs} epochs · batch={batch} · lr={lr} · τ={tau}")
    print(f"  Loss: cosine + 0.5*variance + 0.1*covariance\n")

    train(
        online, target, predictor, pairs,
        epochs=epochs, batch=batch, lr=lr, tau=tau,
        noise_std=0.05, lam_var=0.5, lam_cov=0.1,
    )

    print(f"\n  Verifying separation …")
    online.eval()
    stats = verify(online)

    print(f"\n  ┌─────────────────────────────────────────────┐")
    print(f"  │  Momentum Encoder — Concept Separation       │")
    print(f"  ├─────────────────────────────────────────────┤")
    print(f"  │  Within-cluster (normal):  {stats['within_normal']:.4f}            │")
    print(f"  │  Within-cluster (ice):     {stats['within_ice']:.4f}            │")
    print(f"  │  Between-cluster:          {stats['between']:.4f}            │")
    print(f"  │  Separation margin:        {stats['separation']:+.4f}           │")
    print(f"  │  Centroid angle:           {stats['centroid_angle']:.1f}°             │")
    print(f"  │  Router conf (normal):     {stats['conf_normal']:.4f}            │")
    print(f"  │  Router conf (ice):        {stats['conf_ice']:.4f}            │")
    print(f"  │  Router D normal (mean):   {stats['router_D_normal']:.4f}            │")
    print(f"  │  Router D ice (mean):      {stats['router_D_ice']:.4f}            │")
    print(f"  └─────────────────────────────────────────────┘")

    separated = stats["separation"] > 0 and stats["centroid_angle"] > 10.0
    confident = stats["conf_normal"] > 0.25 and stats["conf_ice"] > 0.25
    routing_works = stats["router_D_normal"] < 0.40 and stats["router_D_ice"] > 0.40

    print(f"\n  Concept separation: {'✓' if separated else '✗'}  "
          f"Confidence: {'✓' if confident else '✗'}  "
          f"Routing gap: {'✓' if routing_works else '✗'}")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    torch.save({
        "encoder_state": online.state_dict(),
        "embed_dim":     EMBED_DIM,
        "imu_dim":       IMU_DIM,
        "tactile_dim":   TACTILE_DIM,
        "stats":         stats,
    }, MODEL_PATH)
    print(f"\n  Saved → {MODEL_PATH.relative_to(_ROOT)}")

    if routing_works:
        print(f"  ✓ Ready for mujoco_bipedal_proof_trained.py --model momentum")
    else:
        d_gap = stats["router_D_ice"] - stats["router_D_normal"]
        print(f"  D gap = {d_gap:+.4f}  (need > 0) — try --epochs 200 or --tau 0.999")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps",  type=int,   default=100000)
    ap.add_argument("--epochs", type=int,   default=150)
    ap.add_argument("--batch",  type=int,   default=512)
    ap.add_argument("--lr",     type=float, default=3e-4)
    ap.add_argument("--tau",    type=float, default=0.995)
    a = ap.parse_args()
    main(a.steps, a.epochs, a.batch, a.lr, a.tau)
