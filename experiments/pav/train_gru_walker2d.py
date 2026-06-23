"""
GRU Encoder Training — Walker2d Terrain
=========================================
Trains GRUProprioEncoder on rolling windows of Walker2d obs using
a momentum (EMA) target encoder + VICReg covariance.

Why GRU fixes the per-frame encoder's failure:

  Per-frame encoder: obs_t → z_t
    Walker2d gait produces highly variable per-frame obs.
    Normal step 5 looks different from normal step 12 (different gait phase).
    The within-cluster scatter is so high that normal and ice D values overlap.

  GRU sequence encoder: [obs_{t-9}, ..., obs_t] → z_t
    Over 10 steps, the GRU accumulates:
      - Height trend: stable at 1.25 (normal) vs dropping (ice)
      - Velocity variance: regular (normal) vs erratic (ice)
      - Joint angle pattern: periodic gait (normal) vs collapsing (ice)
    The representation is gait-phase invariant and terrain-type sensitive.

Expected after training:
  - Centroid angle > 60°
  - Within-cluster distance < 0.4
  - Router D ice vs normal gap > 0.4

Run:
    poetry run python experiments/train_gru_walker2d.py
"""
from __future__ import annotations

import copy
import argparse
import sys
import math
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

import gymnasium
from encoders.robotics.gru_proprio_encoder import GRUProprioEncoder

OBS_DIM    = 17
SEQ_LEN    = 10
EMBED_DIM  = 8
HIDDEN_DIM = 32
FRICTION_NORMAL = 0.80
FRICTION_ICE    = 0.05
MODEL_PATH = _ROOT / "models" / "pav" / "gru_proprio.pt"


# ══════════════════════════════════════════════════════════════════════════════
# Predictor + VICReg
# ══════════════════════════════════════════════════════════════════════════════

class Predictor(nn.Module):
    def __init__(self, d: int = EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d * 4), nn.GELU(),
            nn.LayerNorm(d * 4),
            nn.Linear(d * 4, d),
        )
    def forward(self, z): return self.net(z)


def variance_loss(z: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    return F.relu(torch.tensor(gamma) - z.std(dim=0)).mean()

def covariance_loss(z: torch.Tensor) -> torch.Tensor:
    B, D = z.shape
    z_c  = z - z.mean(dim=0)
    cov  = (z_c.T @ z_c) / (B - 1)
    return ((cov - torch.diag(cov.diag())) ** 2).sum() / D


# ══════════════════════════════════════════════════════════════════════════════
# Rollout collection → rolling windows
# ══════════════════════════════════════════════════════════════════════════════

def collect_windows(
    n_windows: int,
    friction:  float,
    seed:      int,
    noise_std: float = 0.0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Collect (window_t, window_{t+1}) pairs.
    window_t  = obs[t-SEQ_LEN+1 : t+1]   shape (SEQ_LEN, OBS_DIM)
    window_t1 = obs[t-SEQ_LEN+2 : t+2]   shape (SEQ_LEN, OBS_DIM) — one step ahead
    """
    env  = gymnasium.make("Walker2d-v5")
    rng  = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    env.unwrapped.model.geom_friction[0, 0] = friction

    buf   = deque(maxlen=SEQ_LEN + 1)   # keep one extra for the +1 window
    pairs = []

    while len(pairs) < n_windows:
        buf.append(obs.copy())
        action = rng.uniform(-0.4, 0.4, size=env.action_space.shape)
        obs, _, term, trunc, _ = env.step(action)

        if len(buf) == SEQ_LEN + 1:
            win_t  = np.array(list(buf)[:SEQ_LEN])   # oldest SEQ_LEN
            win_t1 = np.array(list(buf)[1:])          # shifted by 1
            if noise_std > 0:
                win_t  = win_t  + rng.normal(0, noise_std, win_t.shape)
                win_t1 = win_t1 + rng.normal(0, noise_std, win_t1.shape)
            pairs.append((win_t, win_t1))

        if term or trunc:
            obs, _ = env.reset()
            env.unwrapped.model.geom_friction[0, 0] = friction
            buf.clear()

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
# Training
# ══════════════════════════════════════════════════════════════════════════════

def train(
    online:    GRUProprioEncoder,
    target:    GRUProprioEncoder,
    predictor: Predictor,
    pairs:     list,
    epochs:    int,
    batch:     int,
    lr:        float,
    tau:       float,
) -> None:
    opt = torch.optim.Adam(
        list(online.parameters()) + list(predictor.parameters()),
        lr=lr, weight_decay=1e-4,
    )
    n = len(pairs)

    for epoch in range(epochs):
        idx   = np.random.permutation(n)
        t_cos = t_var = t_cov = 0.0
        steps = 0

        for start in range(0, n - batch, batch):
            bi   = idx[start : start + batch]
            B    = len(bi)
            wt   = torch.zeros(B, SEQ_LEN, OBS_DIM)
            wt1  = torch.zeros(B, SEQ_LEN, OBS_DIM)
            for i, k in enumerate(bi):
                wt[i]  = torch.from_numpy(pairs[k][0]).float()
                wt1[i] = torch.from_numpy(pairs[k][1]).float()

            z_online = online(wt)                # (B, D)
            z_pred   = predictor(z_online)       # (B, D)
            with torch.no_grad():
                z_target = target(wt1)           # (B, D)

            cos_loss = (1 - F.cosine_similarity(z_pred, z_target, dim=-1)).mean()
            v_loss   = variance_loss(z_online)
            c_loss   = covariance_loss(z_online)
            loss     = cos_loss + 0.5 * v_loss + 0.1 * c_loss

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(online.parameters(), 1.0)
            opt.step()
            ema_update(online, target, tau)

            t_cos += cos_loss.item(); t_var += v_loss.item()
            t_cov += c_loss.item();   steps += 1

        if (epoch + 1) % 20 == 0:
            print(f"  epoch {epoch+1:3d}/{epochs}  "
                  f"cos={t_cos/steps:.4f}  var={t_var/steps:.4f}  "
                  f"cov={t_cov/steps:.4f}  std={z_online.std(dim=0).mean().item():.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# Verification
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def verify(encoder: GRUProprioEncoder, n: int = 300) -> dict:
    env = gymnasium.make("Walker2d-v5")
    rng = np.random.default_rng(999)
    z_normal, z_ice = [], []

    for friction, zlist, seed in [
        (FRICTION_NORMAL, z_normal, 1000),
        (FRICTION_ICE,    z_ice,    2000),
    ]:
        obs, _ = env.reset(seed=seed)
        env.unwrapped.model.geom_friction[0, 0] = friction
        buf = deque(maxlen=SEQ_LEN)
        for _ in range(n + SEQ_LEN):
            buf.append(obs.copy())
            if len(buf) == SEQ_LEN:
                win = torch.from_numpy(np.array(buf)).float().unsqueeze(0)
                z   = encoder(win).squeeze(0)
                zlist.append(z)
            action = rng.uniform(-0.4, 0.4, size=env.action_space.shape)
            obs, _, term, trunc, _ = env.step(action)
            if term or trunc:
                obs, _ = env.reset()
                env.unwrapped.model.geom_friction[0, 0] = friction
                buf.clear()
    env.close()

    zn = torch.stack(z_normal[:n])
    zi = torch.stack(z_ice[:n])

    def mean_cdist(a, b):
        cos = F.cosine_similarity(
            a.unsqueeze(1).expand(-1, len(b), -1).reshape(-1, EMBED_DIM),
            b.unsqueeze(0).expand(len(a), -1, -1).reshape(-1, EMBED_DIM), dim=-1,
        )
        return float(1 - cos.mean())

    def mean_conf(z):
        G = EMBED_DIM
        p = F.softmax(z, dim=-1)
        return float(((p.max(dim=-1).values - 1/G) / (1 - 1/G)).clamp(0).mean())

    def router_D(z_v, zs):
        ds = []
        for z_p in zs:
            p_a = F.softmax(z_v, dim=0)
            p_b = F.softmax(z_p, dim=0)
            ds.append(float((p_a - p_b).abs().sum() / math.sqrt(EMBED_DIM)))
        return float(np.mean(ds))

    wn, wi, bw = mean_cdist(zn, zn), mean_cdist(zi, zi), mean_cdist(zn, zi)
    mn = F.normalize(zn.mean(0), dim=0)
    mi = F.normalize(zi.mean(0), dim=0)
    angle = math.degrees(math.acos(
        float(F.cosine_similarity(mn.unsqueeze(0), mi.unsqueeze(0)).clamp(-1, 1))
    ))

    # z_vision = centroid of normal — scaled to match typical embedding magnitude
    z_v = zn.mean(0)
    scale = zn.std(dim=0).mean() / z_v.norm().clamp(min=1e-6)
    z_v_scaled = z_v * scale

    return {
        "within_normal":   round(wn, 4),
        "within_ice":      round(wi, 4),
        "between":         round(bw, 4),
        "separation":      round(bw - max(wn, wi), 4),
        "centroid_angle":  round(angle, 1),
        "conf_normal":     round(mean_conf(zn), 4),
        "conf_ice":        round(mean_conf(zi), 4),
        "router_D_normal": round(router_D(z_v_scaled, zn[:100]), 4),
        "router_D_ice":    round(router_D(z_v_scaled, zi[:100]), 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main(windows: int, epochs: int, batch: int, lr: float, tau: float) -> None:
    print("\n" + "═" * 65)
    print("  GRU Encoder Training — Walker2d Terrain")
    print(f"  seq_len={SEQ_LEN} · EMA τ={tau} · momentum + VICReg")
    print("═" * 65)

    half = windows // 2
    print(f"\n  Collecting {half} normal windows …")
    normal_w = collect_windows(half, FRICTION_NORMAL, seed=0, noise_std=0.03)
    print(f"  Collecting {half} ice windows …")
    ice_w    = collect_windows(half, FRICTION_ICE,    seed=1, noise_std=0.03)
    pairs = normal_w + ice_w
    np.random.default_rng(42).shuffle(pairs)
    print(f"  Total window pairs: {len(pairs)}")

    online    = GRUProprioEncoder(OBS_DIM, HIDDEN_DIM, EMBED_DIM, SEQ_LEN)
    target    = copy.deepcopy(online)
    predictor = Predictor(EMBED_DIM)
    for p in target.parameters():
        p.requires_grad_(False)

    print(f"\n  Training: {epochs} epochs · batch={batch} · lr={lr}")
    print(f"  Loss: cosine + 0.5*var + 0.1*cov\n")
    train(online, target, predictor, pairs, epochs, batch, lr, tau)

    print(f"\n  Verifying separation …")
    online.eval()
    s = verify(online)

    print(f"\n  ┌───────────────────────────────────────────────┐")
    print(f"  │  GRU Encoder — Concept Separation              │")
    print(f"  ├───────────────────────────────────────────────┤")
    print(f"  │  Within-cluster normal:  {s['within_normal']:.4f}               │")
    print(f"  │  Within-cluster ice:     {s['within_ice']:.4f}               │")
    print(f"  │  Between-cluster:        {s['between']:.4f}               │")
    print(f"  │  Separation margin:      {s['separation']:+.4f}              │")
    print(f"  │  Centroid angle:         {s['centroid_angle']:.1f}°                 │")
    print(f"  │  Router conf normal:     {s['conf_normal']:.4f}               │")
    print(f"  │  Router conf ice:        {s['conf_ice']:.4f}               │")
    print(f"  │  Router D normal (mean): {s['router_D_normal']:.4f}               │")
    print(f"  │  Router D ice (mean):    {s['router_D_ice']:.4f}               │")
    print(f"  │  D gap (ice - normal):   {s['router_D_ice']-s['router_D_normal']:+.4f}              │")
    print(f"  └───────────────────────────────────────────────┘")

    sep  = s["separation"] > 0 and s["centroid_angle"] > 20
    conf = s["conf_normal"] > 0.25 and s["conf_ice"] > 0.25
    gap  = (s["router_D_ice"] - s["router_D_normal"]) > 0.10

    print(f"\n  Separation: {'✓' if sep else '✗'}  "
          f"Confidence: {'✓' if conf else '✗'}  "
          f"D gap: {'✓' if gap else '✗'}")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    torch.save({
        "encoder_state": online.state_dict(),
        "obs_dim": OBS_DIM, "hidden_dim": HIDDEN_DIM,
        "embed_dim": EMBED_DIM, "seq_len": SEQ_LEN,
        "stats": s,
    }, MODEL_PATH)
    print(f"\n  Saved → {MODEL_PATH.relative_to(_ROOT)}")
    if sep and conf and gap:
        print(f"  ✓ Ready — run mujoco_bipedal_proof_gru.py")
    else:
        print(f"  ↺ Try --epochs 200 --windows 120000 for stronger separation")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int,   default=100000)
    ap.add_argument("--epochs",  type=int,   default=150)
    ap.add_argument("--batch",   type=int,   default=512)
    ap.add_argument("--lr",      type=float, default=3e-4)
    ap.add_argument("--tau",     type=float, default=0.995)
    a = ap.parse_args()
    main(a.windows, a.epochs, a.batch, a.lr, a.tau)
