"""
JEPA Encoder Training — Walker2d Terrain
==========================================
Trains ProprioceptiveEncoder on real Walker2d rollouts using JEPA
(Joint-Embedding Predictive Architecture). No terrain labels used.

The encoder learns temporal structure: z_t → predict z_{t+1}.
Normal and ice terrain have fundamentally different temporal statistics:
  - Normal: stable height, smooth velocity → predictable → tight cluster
  - Ice:    falling height, erratic velocity → hard to predict → different cluster

After training, we verify concept separation by measuring:
  - Within-class cosine distance (normal-to-normal, ice-to-ice)
  - Between-class cosine distance (normal-to-ice)
  - Router confidence (must exceed tau_low=0.25 with raw LayerNorm output)

Loss: JEPA cosine loss + VICReg variance term (prevents collapse)
    L = (1 - cos(ẑ_{t+1}, z_{t+1})) + λ * max(0, γ - std(z_t))

Run:
    poetry run python experiments/train_jepa_walker2d.py
    poetry run python experiments/train_jepa_walker2d.py --epochs 80 --steps 60000
"""
from __future__ import annotations

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

# ── constants ─────────────────────────────────────────────────────────────────
OBS_DIM       = 17   # Walker2d-v5
IMU_DIM       = 64
TACTILE_DIM   = 32
EMBED_DIM     = 8
FRICTION_NORMAL = 0.80
FRICTION_ICE    = 0.05
MODEL_PATH    = _ROOT / "models" / "pav" / "proprio_jepa.pt"


# ══════════════════════════════════════════════════════════════════════════════
# Obs → encoder input mapping
# ══════════════════════════════════════════════════════════════════════════════

def obs_to_encoder(obs: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Map Walker2d 17-dim obs to ProprioceptiveEncoder inputs.

    Walker2d obs layout:
      [0]    torso z-height
      [1]    torso angle
      [2:8]  joint angles (thigh_l, leg_l, foot_l, thigh_r, leg_r, foot_r)
      [8]    torso x-velocity
      [9]    torso z-velocity
      [10]   torso angle velocity
      [11:17] joint velocities

    IMU (64-dim): positions + velocities in [0:17], rest zeros
    Tactile (32-dim): foot joints + foot velocities in [0:9], rest zeros
                      proxy for foot-ground contact pressure
    """
    imu     = torch.zeros(1, IMU_DIM)
    tactile = torch.zeros(1, TACTILE_DIM)

    t = torch.from_numpy(obs).float()
    imu[0, :17]    = t              # full obs → IMU channels
    tactile[0, :6] = t[2:8]        # foot joint angles → pressure proxy
    tactile[0, 6:12] = t[11:17]   # foot velocities (6 elements into slots 6-11)

    return imu, tactile


# ══════════════════════════════════════════════════════════════════════════════
# JEPA predictor
# ══════════════════════════════════════════════════════════════════════════════

class JEPAPredictor(nn.Module):
    """
    Predicts z_{t+1} from z_t.
    Kept small — the encoder does the heavy lifting.
    """
    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ══════════════════════════════════════════════════════════════════════════════
# Rollout collection
# ══════════════════════════════════════════════════════════════════════════════

def collect_rollouts(
    n_steps:    int,
    friction:   float,
    seed:       int,
    reset_every: int = 500,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Collect (obs_t, obs_{t+1}) pairs from Walker2d with given friction.
    Uses random actions — we only need physics observations, not a policy.
    """
    env  = gymnasium.make("Walker2d-v5")
    rng  = np.random.default_rng(seed)
    pairs: list[tuple[np.ndarray, np.ndarray]] = []

    obs, _ = env.reset(seed=seed)
    env.unwrapped.model.geom_friction[0, 0] = friction
    step = 0

    while len(pairs) < n_steps:
        action = rng.uniform(-0.4, 0.4, size=env.action_space.shape)
        obs_t  = obs.copy()
        obs_next, _, terminated, truncated, _ = env.step(action)

        pairs.append((obs_t, obs_next.copy()))
        obs   = obs_next
        step += 1

        if terminated or truncated or (step % reset_every == 0):
            obs, _ = env.reset()
            env.unwrapped.model.geom_friction[0, 0] = friction

    env.close()
    return pairs


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

def train(
    encoder:   ProprioceptiveEncoder,
    predictor: JEPAPredictor,
    pairs:     list[tuple[np.ndarray, np.ndarray]],
    epochs:    int,
    batch:     int,
    lr:        float,
    lambda_var: float,
    gamma:     float,
) -> list[float]:
    """
    JEPA training loop with VICReg variance term.

    L = cosine_loss(ẑ_{t+1}, stop_grad(z_{t+1})) + λ * max(0, γ - std(z_t))
    """
    opt = torch.optim.Adam(
        list(encoder.parameters()) + list(predictor.parameters()), lr=lr
    )
    losses: list[float] = []
    n = len(pairs)

    for epoch in range(epochs):
        idx   = np.random.permutation(n)
        total = 0.0
        steps = 0

        for start in range(0, n - batch, batch):
            batch_idx = idx[start : start + batch]

            imu_t  = torch.zeros(batch, IMU_DIM)
            tac_t  = torch.zeros(batch, TACTILE_DIM)
            imu_t1 = torch.zeros(batch, IMU_DIM)
            tac_t1 = torch.zeros(batch, TACTILE_DIM)

            for i, bi in enumerate(batch_idx):
                obs_t, obs_t1   = pairs[bi]
                im, ta          = obs_to_encoder(obs_t)
                imu_t[i]        = im[0]
                tac_t[i]        = ta[0]
                im1, ta1        = obs_to_encoder(obs_t1)
                imu_t1[i]       = im1[0]
                tac_t1[i]       = ta1[0]

            z_t  = encoder(imu_t, tac_t)               # (B, D) — gradients flow
            z_hat = predictor(z_t)                      # (B, D) — prediction

            with torch.no_grad():
                z_t1 = encoder(imu_t1, tac_t1)         # (B, D) — stop grad target

            # JEPA cosine loss
            jepa_loss = (1.0 - F.cosine_similarity(z_hat, z_t1, dim=-1)).mean()

            # VICReg variance: prevent embedding collapse
            std = z_t.std(dim=0).mean()
            var_loss = F.relu(torch.tensor(gamma) - std)

            loss = jepa_loss + lambda_var * var_loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            opt.step()

            total += loss.item()
            steps += 1

        avg = total / max(steps, 1)
        losses.append(avg)
        if (epoch + 1) % 10 == 0:
            std_val = z_t.std(dim=0).mean().item()
            print(f"  epoch {epoch+1:3d}/{epochs}  loss={avg:.4f}  embed_std={std_val:.3f}")

    return losses


# ══════════════════════════════════════════════════════════════════════════════
# Verification — concept separation
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def verify_separation(
    encoder: ProprioceptiveEncoder,
    n_verify: int = 300,
) -> dict:
    """
    Measure normal vs ice cluster separation in concept space.

    Reports:
      - mean cosine distance within class (lower = tighter cluster)
      - mean cosine distance between classes (higher = better separation)
      - router softmax confidence (must be > tau_low=0.25)
    """
    env = gymnasium.make("Walker2d-v5")
    rng = np.random.default_rng(999)
    z_normal, z_ice = [], []

    for friction, zlist, s in [
        (FRICTION_NORMAL, z_normal, 1000),
        (FRICTION_ICE,    z_ice,    2000),
    ]:
        obs, _ = env.reset(seed=s)
        env.unwrapped.model.geom_friction[0, 0] = friction
        for _ in range(n_verify):
            action = rng.uniform(-0.4, 0.4, size=env.action_space.shape)
            imu, tac = obs_to_encoder(obs)
            z = encoder(imu, tac).squeeze(0)
            zlist.append(z)
            obs, _, term, trunc, _ = env.step(action)
            if term or trunc:
                obs, _ = env.reset()
                env.unwrapped.model.geom_friction[0, 0] = friction
    env.close()

    zn = torch.stack(z_normal)  # (N, D)
    zi = torch.stack(z_ice)     # (N, D)

    # Cosine distance = 1 - cosine_similarity
    def mean_cos_dist(a: torch.Tensor, b: torch.Tensor) -> float:
        sims = F.cosine_similarity(
            a.unsqueeze(1).expand(-1, len(b), -1).reshape(-1, a.shape[-1]),
            b.unsqueeze(0).expand(len(a), -1, -1).reshape(-1, b.shape[-1]),
            dim=-1,
        )
        return float(1.0 - sims.mean())

    within_n  = mean_cos_dist(zn, zn)
    within_i  = mean_cos_dist(zi, zi)
    between   = mean_cos_dist(zn, zi)

    # Router confidence for normal and ice
    def mean_conf(zs: torch.Tensor) -> float:
        G   = zs.shape[-1]
        p   = F.softmax(zs, dim=-1)
        max_p = p.max(dim=-1).values
        return float(((max_p - 1.0/G) / (1.0 - 1.0/G)).clamp(min=0).mean())

    conf_normal = mean_conf(zn)
    conf_ice    = mean_conf(zi)

    mean_zn = F.normalize(zn.mean(0), dim=0)
    mean_zi = F.normalize(zi.mean(0), dim=0)
    centroid_angle = math.degrees(
        math.acos(float(F.cosine_similarity(mean_zn.unsqueeze(0), mean_zi.unsqueeze(0)).clamp(-1, 1)))
    )

    return {
        "within_normal":  round(within_n, 4),
        "within_ice":     round(within_i, 4),
        "between":        round(between,  4),
        "separation":     round(between - max(within_n, within_i), 4),
        "centroid_angle": round(centroid_angle, 1),
        "conf_normal":    round(conf_normal, 4),
        "conf_ice":       round(conf_ice, 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main(steps: int, epochs: int, batch: int, lr: float) -> None:
    print("\n" + "═" * 65)
    print("  JEPA Encoder Training — Walker2d Terrain")
    print("  Label-free · real MuJoCo physics · trains on your laptop")
    print("═" * 65)

    # ── collect rollouts ──────────────────────────────────────────────────────
    half = steps // 2
    print(f"\n  Collecting {half} normal steps  (friction={FRICTION_NORMAL}) …")
    normal_pairs = collect_rollouts(half, FRICTION_NORMAL, seed=0)
    print(f"  Collecting {half} ice steps     (friction={FRICTION_ICE}) …")
    ice_pairs    = collect_rollouts(half, FRICTION_ICE, seed=1)
    pairs        = normal_pairs + ice_pairs
    np.random.default_rng(42).shuffle(pairs)
    print(f"  Total pairs: {len(pairs)}")

    # ── init models ───────────────────────────────────────────────────────────
    encoder   = ProprioceptiveEncoder(IMU_DIM, TACTILE_DIM, EMBED_DIM)
    predictor = JEPAPredictor(EMBED_DIM)

    # ── train ─────────────────────────────────────────────────────────────────
    print(f"\n  Training JEPA: {epochs} epochs · batch={batch} · lr={lr}")
    print(f"  Loss = cosine_loss + 0.5 * max(0, 1.0 - std(z))\n")
    train(
        encoder, predictor, pairs,
        epochs=epochs, batch=batch, lr=lr,
        lambda_var=0.5, gamma=1.0,
    )

    # ── verify ────────────────────────────────────────────────────────────────
    print(f"\n  Verifying concept separation (300 obs per class) …")
    encoder.eval()
    stats = verify_separation(encoder)
    print(f"\n  ┌─────────────────────────────────────────┐")
    print(f"  │  Concept Separation Report               │")
    print(f"  ├─────────────────────────────────────────┤")
    print(f"  │  Within-cluster (normal):  {stats['within_normal']:.4f}          │")
    print(f"  │  Within-cluster (ice):     {stats['within_ice']:.4f}          │")
    print(f"  │  Between-cluster:          {stats['between']:.4f}          │")
    print(f"  │  Separation margin:        {stats['separation']:+.4f}         │")
    print(f"  │  Centroid angle:           {stats['centroid_angle']:.1f}°           │")
    print(f"  │  Router conf (normal):     {stats['conf_normal']:.4f}          │")
    print(f"  │  Router conf (ice):        {stats['conf_ice']:.4f}          │")
    print(f"  └─────────────────────────────────────────┘")

    separated = stats["separation"] > 0 and stats["centroid_angle"] > 5.0
    confident = stats["conf_normal"] > 0.25 and stats["conf_ice"] > 0.25
    print(f"\n  Concept separation: {'✓ YES' if separated else '✗ NO — need more training'}")
    print(f"  Router confidence:  {'✓ YES' if confident else '✗ NO — below tau_low=0.25'}")

    # ── save ──────────────────────────────────────────────────────────────────
    MODEL_PATH.parent.mkdir(exist_ok=True)
    torch.save({
        "encoder_state": encoder.state_dict(),
        "embed_dim":     EMBED_DIM,
        "imu_dim":       IMU_DIM,
        "tactile_dim":   TACTILE_DIM,
        "stats":         stats,
    }, MODEL_PATH)
    print(f"\n  Encoder saved → {MODEL_PATH.relative_to(_ROOT)}")

    if separated and confident:
        print(f"\n  ✓ Proof ready — run mujoco_bipedal_proof.py --trained to use real encoder")
    else:
        print(f"\n  ↺ Run with --epochs 150 --steps 120000 for stronger separation")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps",  type=int,   default=80000)
    ap.add_argument("--epochs", type=int,   default=60)
    ap.add_argument("--batch",  type=int,   default=512)
    ap.add_argument("--lr",     type=float, default=1e-3)
    a = ap.parse_args()
    main(a.steps, a.epochs, a.batch, a.lr)
