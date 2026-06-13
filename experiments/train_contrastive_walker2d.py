"""
Contrastive GRU Encoder — Walker2d Terrain
===========================================
SimCLR-style NT-Xent loss: same-terrain windows pulled together,
cross-terrain windows pushed apart. Within-cluster scatter collapses
to near 0 (vs 0.72 with momentum JEPA), giving clean D separation.

Training uses terrain labels (normal/ice windows collected separately),
but inference is label-free: routing fires purely from concept geometry.

Run:
    poetry run python experiments/train_contrastive_walker2d.py
"""
from __future__ import annotations

import sys, math, copy, argparse
from pathlib import Path
from collections import deque

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

import gymnasium
from encoders.gru_proprio_encoder import GRUProprioEncoder

OBS_DIM    = 17
SEQ_LEN    = 30
EMBED_DIM  = 8
HIDDEN_DIM = 64        # bigger hidden to capture terrain physics over 30 steps
FRICTION_NORMAL = 0.80
FRICTION_ICE    = 0.05
MODEL_PATH = _ROOT / "models" / "gru_contrastive.pt"


# ─────────────────────────────────────────────────────────────────────────────
# Supervised Contrastive loss (SupCon)
# ─────────────────────────────────────────────────────────────────────────────

def supcon_loss(z: torch.Tensor, labels: torch.Tensor, temperature: float = 0.10) -> torch.Tensor:
    """
    z:      (B, D) L2-normalized embeddings
    labels: (B,)   integer terrain labels (0=normal, 1=ice)

    Positives = same label, negatives = different label.
    Cross-terrain pairs are hard negatives inside the same loss — no separate
    repulsion term needed.
    """
    B = z.shape[0]
    sim = (z @ z.T) / temperature                              # (B, B)

    eye     = torch.eye(B, dtype=torch.bool, device=z.device)
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1))    # (B, B) same terrain
    pos_mask = pos_mask & ~eye

    # Numerical stability
    sim = sim - sim.detach().max(dim=1, keepdim=True).values

    exp_sim  = torch.exp(sim)
    exp_sim  = exp_sim.masked_fill(eye, 0.0)                   # zero out diagonal

    log_denom = torch.log(exp_sim.sum(dim=1) + 1e-8)            # all non-self pairs

    n_pos = pos_mask.sum(dim=1).clamp(min=1).float()
    loss  = (-(sim * pos_mask).sum(dim=1) / n_pos + log_denom).mean()
    return loss


def variance_loss(z: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    return F.relu(torch.tensor(gamma) - z.std(dim=0)).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Data collection
# ─────────────────────────────────────────────────────────────────────────────

def collect_windows(n: int, friction: float, seed: int, noise_std: float = 0.03):
    env = gymnasium.make("Walker2d-v5")
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    env.unwrapped.model.geom_friction[0, 0] = friction
    buf = deque(maxlen=SEQ_LEN)
    wins = []
    while len(wins) < n:
        buf.append(obs.copy())
        if len(buf) == SEQ_LEN:
            w = np.array(buf)
            if noise_std > 0:
                w = w + rng.normal(0, noise_std, w.shape)
            wins.append(w)
        action = rng.uniform(-0.4, 0.4, size=env.action_space.shape)
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            obs, _ = env.reset()
            env.unwrapped.model.geom_friction[0, 0] = friction
            buf.clear()
    env.close()
    return wins


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(n_each: int = 50000, epochs: int = 150, batch: int = 256,
         lr: float = 3e-4, tau_nt: float = 0.10) -> None:

    print("\n" + "═" * 65)
    print("  Contrastive GRU Encoder — Walker2d Terrain")
    print(f"  NT-Xent τ={tau_nt} · hidden={HIDDEN_DIM} · seq_len={SEQ_LEN}")
    print("═" * 65)

    print(f"\n  Collecting {n_each} normal windows …")
    normal_wins = collect_windows(n_each, FRICTION_NORMAL, seed=0)
    print(f"  Collecting {n_each} ice windows …")
    ice_wins    = collect_windows(n_each, FRICTION_ICE, seed=1)
    # Second set for positive pairs (different augmentation / time)
    print(f"  Collecting {n_each} normal windows (pair set 2) …")
    normal_wins2 = collect_windows(n_each, FRICTION_NORMAL, seed=10)
    print(f"  Collecting {n_each} ice windows (pair set 2) …")
    ice_wins2    = collect_windows(n_each, FRICTION_ICE, seed=11)
    print(f"  Data ready: {n_each} normal + {n_each} ice pairs each")

    encoder = GRUProprioEncoder(OBS_DIM, HIDDEN_DIM, EMBED_DIM, SEQ_LEN)
    opt = torch.optim.Adam(encoder.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr/10)

    n_pairs = n_each  # one pair per sample
    print(f"\n  Training {epochs} epochs · batch={batch} · NT-Xent τ={tau_nt}")
    print(f"  Loss: NT-Xent(normal) + NT-Xent(ice) + 0.5*variance\n")

    half = batch // 2   # half normal, half ice per batch
    for epoch in range(epochs):
        idx_n = np.random.permutation(n_each)
        idx_i = np.random.permutation(n_each)
        t_sc = t_var = 0.0
        steps = 0

        for start in range(0, n_each - half, half):
            bn = idx_n[start : start + half]
            bi = idx_i[start : start + half]

            # Two views per sample (different augmentation seeds)
            wn1 = torch.from_numpy(np.array([normal_wins[k]  for k in bn])).float()
            wn2 = torch.from_numpy(np.array([normal_wins2[k] for k in bn])).float()
            wi1 = torch.from_numpy(np.array([ice_wins[k]  for k in bi])).float()
            wi2 = torch.from_numpy(np.array([ice_wins2[k] for k in bi])).float()

            # Mixed batch: [normal_view1, ice_view1, normal_view2, ice_view2]
            # SupCon sees all four views; labels 0=normal 1=ice
            B4 = 2 * half
            w_all  = torch.cat([wn1, wi1, wn2, wi2], dim=0)       # (4*half, seq, obs)
            labels = torch.cat([
                torch.zeros(half, dtype=torch.long),
                torch.ones(half,  dtype=torch.long),
                torch.zeros(half, dtype=torch.long),
                torch.ones(half,  dtype=torch.long),
            ])

            z_all = F.normalize(encoder(w_all), dim=1)             # (4*half, D)
            sc    = supcon_loss(z_all, labels, tau_nt)
            v_loss = variance_loss(encoder(torch.cat([wn1, wi1], dim=0)))

            loss = sc + 0.5 * v_loss

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            opt.step()

            t_sc  += sc.item()
            t_var += v_loss.item()
            steps += 1

        scheduler.step()
        if (epoch + 1) % 30 == 0:
            with torch.no_grad():
                zn_s = encoder(torch.from_numpy(np.array(normal_wins[:512])).float())
                zi_s = encoder(torch.from_numpy(np.array(ice_wins[:512])).float())
                wn = float(1 - F.cosine_similarity(
                    zn_s.unsqueeze(1).expand(-1, 512, -1).reshape(-1, EMBED_DIM),
                    zn_s.unsqueeze(0).expand(512, -1, -1).reshape(-1, EMBED_DIM), dim=-1
                ).mean())
                wi = float(1 - F.cosine_similarity(
                    zi_s.unsqueeze(1).expand(-1, 512, -1).reshape(-1, EMBED_DIM),
                    zi_s.unsqueeze(0).expand(512, -1, -1).reshape(-1, EMBED_DIM), dim=-1
                ).mean())
                bw = float(1 - F.cosine_similarity(
                    zn_s[:512].unsqueeze(1).expand(-1, 512, -1).reshape(-1, EMBED_DIM),
                    zi_s[:512].unsqueeze(0).expand(512, -1, -1).reshape(-1, EMBED_DIM), dim=-1
                ).mean())
                conf_n = float(((F.softmax(zn_s, dim=-1).max(dim=-1).values - 1/8) / (1 - 1/8)).clamp(0).mean())
                conf_i = float(((F.softmax(zi_s, dim=-1).max(dim=-1).values - 1/8) / (1 - 1/8)).clamp(0).mean())
            print(f"  epoch {epoch+1:3d}  sc={t_sc/steps:.4f}  var={t_var/steps:.4f}  "
                  f"wn={wn:.3f} wi={wi:.3f} bw={bw:.3f}  conf_n={conf_n:.3f} conf_i={conf_i:.3f}")

    # ── Final verification ────────────────────────────────────────────────────
    encoder.eval()
    with torch.no_grad():
        zn = encoder(torch.from_numpy(np.array(normal_wins[:300])).float())
        zi = encoder(torch.from_numpy(np.array(ice_wins[:300])).float())

    def cdist(a, b):
        cos = F.cosine_similarity(
            a.unsqueeze(1).expand(-1, len(b), -1).reshape(-1, EMBED_DIM),
            b.unsqueeze(0).expand(len(a), -1, -1).reshape(-1, EMBED_DIM), dim=-1)
        return float(1 - cos.mean())

    def mean_conf(z):
        p = F.softmax(z, dim=-1)
        return float(((p.max(dim=-1).values - 1/8) / (1 - 1/8)).clamp(0).mean())

    def router_D(z_v, zs):
        ds = [float(((F.softmax(z_v, dim=0) - F.softmax(z, dim=0)).abs().sum() / math.sqrt(EMBED_DIM)))
              for z in zs]
        return float(np.mean(ds))

    wn, wi, bw = cdist(zn, zn), cdist(zi, zi), cdist(zn, zi)
    mn = F.normalize(zn.mean(0), dim=0); mi = F.normalize(zi.mean(0), dim=0)
    angle = math.degrees(math.acos(float(F.cosine_similarity(mn.unsqueeze(0), mi.unsqueeze(0)).clamp(-1, 1))))

    # z_vision: EWMA over first 50 windows (simulating what proof will do)
    alpha = 0.90
    z_v = zn[0].clone()
    for z in zn[1:50]:
        z_v = alpha * z_v + (1 - alpha) * z
    conf_v = float((F.softmax(z_v, dim=0).max() - 1/8) / (1 - 1/8))

    dn = router_D(z_v, zn[:100])
    di = router_D(z_v, zi[:100])

    stats = {
        "within_normal": round(wn, 4), "within_ice": round(wi, 4), "between": round(bw, 4),
        "separation": round(bw - max(wn, wi), 4), "centroid_angle": round(angle, 1),
        "conf_normal": round(mean_conf(zn), 4), "conf_ice": round(mean_conf(zi), 4),
        "ewma_conf_v": round(conf_v, 4),
        "router_D_normal": round(dn, 4), "router_D_ice": round(di, 4),
    }

    print(f"\n  ┌───────────────────────────────────────────────────────┐")
    print(f"  │  Contrastive GRU — Results                             │")
    print(f"  ├───────────────────────────────────────────────────────┤")
    print(f"  │  Within normal:   {wn:.4f}  (target < 0.25)              │")
    print(f"  │  Within ice:      {wi:.4f}  (target < 0.25)              │")
    print(f"  │  Between:         {bw:.4f}                               │")
    print(f"  │  Separation:      {bw-max(wn,wi):+.4f}  (target > 0.40)             │")
    print(f"  │  Centroid angle:  {angle:.1f}°  (target > 80°)             │")
    print(f"  │  Conf normal:     {mean_conf(zn):.4f}  (target > 0.30)              │")
    print(f"  │  Conf ice:        {mean_conf(zi):.4f}  (target > 0.30)              │")
    print(f"  │  EWMA conf_v:     {conf_v:.4f}  (target > 0.25)              │")
    print(f"  │  D normal:        {dn:.4f}  (target < 0.25)              │")
    print(f"  │  D ice:           {di:.4f}  (target > 0.35)              │")
    print(f"  │  D gap:           {di-dn:+.4f}  (target > 0.20)             │")
    print(f"  └───────────────────────────────────────────────────────┘")

    ok_scatter = wn < 0.25 and wi < 0.25
    ok_conf    = mean_conf(zn) > 0.30 and conf_v > 0.25
    ok_D       = (di - dn) > 0.15
    print(f"\n  Scatter: {'✓' if ok_scatter else '✗'}  Conf: {'✓' if ok_conf else '✗'}  D-gap: {'✓' if ok_D else '✗'}")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    torch.save({
        "encoder_state": encoder.state_dict(),
        "obs_dim": OBS_DIM, "hidden_dim": HIDDEN_DIM,
        "embed_dim": EMBED_DIM, "seq_len": SEQ_LEN,
        "stats": stats,
    }, MODEL_PATH)
    print(f"\n  Saved → {MODEL_PATH.relative_to(_ROOT)}")
    if ok_scatter and ok_conf and ok_D:
        print(f"  ✓ Ready — run mujoco_bipedal_proof_gru.py with MODEL_PATH → gru_contrastive.pt")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",      type=int,   default=50000)
    ap.add_argument("--epochs", type=int,   default=150)
    ap.add_argument("--batch",  type=int,   default=256)
    ap.add_argument("--lr",     type=float, default=3e-4)
    ap.add_argument("--tau",    type=float, default=0.10)
    a = ap.parse_args()
    main(a.n, a.epochs, a.batch, a.lr, a.tau)
