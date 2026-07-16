"""
Multi-seed CLS-GRU retrain — fixes single-trajectory overfitting.
==================================================================
The published gru_cls.pt was trained on 50k windows per class collected
from ONE seed per class (train_cls_walker2d.py: seed=0 normal, seed=1 ice).
Overlapping windows of a single trajectory -> the encoder memorises that
trajectory. On fresh live rollouts it is near chance (ice windows argmax
class 'normal' ~60% of the time; live D gap 0.04-0.09 vs the 0.50
suggested by training-eval stats).

Identical architecture, loss, and hyperparameters — the only change is
data diversity: windows are collected across many seeds per class, and
verification runs on HELD-OUT seeds never used in training.

Saves to models/pav/gru_cls_multiseed.pt (does not touch the published
gru_cls.pt artifact).

Run:
    .venv/bin/python experiments/fleet/train_cls_multiseed.py
"""
from __future__ import annotations

import sys, math, argparse, time
from pathlib import Path
from collections import deque, Counter

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

import gymnasium
from encoders.robotics.gru_proprio_encoder import GRUProprioEncoder
from experiments.fleet.policies import make_policy

OBS_DIM    = 17
SEQ_LEN    = 30
EMBED_DIM  = 8
HIDDEN_DIM = 64
FRICTION_NORMAL = 0.80
FRICTION_ICE    = 0.05
POLICY = "random"   # overridden by --policy; checkpoint name follows it

TRAIN_SEEDS_NORMAL = list(range(0, 40, 2))        # 20 seeds
TRAIN_SEEDS_ICE    = list(range(1, 41, 2))        # 20 seeds, disjoint
EVAL_SEEDS_NORMAL  = [500, 502, 504, 506]         # held out
EVAL_SEEDS_ICE     = [501, 503, 505, 507]         # held out


def set_real_friction(env, f: float) -> None:
    """
    Set sliding friction on ALL geoms (PERSIST ice_world.py mechanism).

    The original PAV set_friction touched only geom 0 (the floor). MuJoCo
    resolves contact friction as the elementwise max of the two geoms'
    coefficients (equal priority), and Walker2d-v5 feet have friction 1.9
    -- so floor-only changes below 1.9 never alter the dynamics. Verified:
    floor mu=0.80 and mu=0.05 produce bit-identical trajectories.
    """
    env.unwrapped.model.geom_friction[:, 0] = f


def collect_windows(n: int, friction: float, seed: int, noise_std: float = 0.02):
    """train_cls_walker2d.collect_windows, generalised over the policy."""
    env = gymnasium.make("Walker2d-v5")
    rng = np.random.default_rng(seed)
    policy = make_policy(POLICY, rng)
    obs, _ = env.reset(seed=seed)
    set_real_friction(env, friction)
    buf = deque(maxlen=SEQ_LEN)
    wins = []
    while len(wins) < n:
        buf.append(obs.copy())
        if len(buf) == SEQ_LEN:
            w = np.array(buf)
            if noise_std > 0:
                w = w + rng.normal(0, noise_std, w.shape)
            wins.append(w)
        obs, _, term, trunc, _ = env.step(policy(obs))
        if term or trunc:
            obs, _ = env.reset()
            set_real_friction(env, friction)
            buf.clear()
            policy.reset()
    env.close()
    return wins


def collect_multiseed(n_total: int, friction: float, seeds: list[int],
                      noise_std: float = 0.02):
    per = n_total // len(seeds)
    wins = []
    for s in seeds:
        wins.extend(collect_windows(per, friction, s, noise_std))
    return wins


def variance_loss(z: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    return F.relu(torch.tensor(gamma) - z.std(dim=0)).mean()


def live_eval(encoder: GRUProprioEncoder) -> dict:
    """Held-out-seed live verification: no obs noise, fresh trajectories."""
    encoder.eval()

    def embed_seeds(friction, seeds, n_per=150):
        zs = []
        for s in seeds:
            for w in collect_windows(n_per, friction, s, noise_std=0.0):
                with torch.no_grad():
                    zs.append(encoder(
                        torch.from_numpy(np.asarray(w, dtype=np.float32)).unsqueeze(0)
                    ).squeeze(0))
        return torch.stack(zs)

    zn = embed_seeds(FRICTION_NORMAL, EVAL_SEEDS_NORMAL)
    zi = embed_seeds(FRICTION_ICE, EVAL_SEEDS_ICE)
    pn, pi = F.softmax(zn, dim=1), F.softmax(zi, dim=1)

    acc_n = float((pn.argmax(1) == 0).float().mean())
    acc_i = float((pi.argmax(1) == 1).float().mean())

    # EWMA-style anchor from the first held-out normal seed's stream
    anchor = None
    for z in zn[:150]:
        anchor = z.clone() if anchor is None else 0.90 * anchor + 0.10 * z
    pa = F.softmax(anchor, dim=0)
    Dn = float(((pa[None, :] - pn).abs().sum(1) / math.sqrt(EMBED_DIM)).mean())
    Di = float(((pa[None, :] - pi).abs().sum(1) / math.sqrt(EMBED_DIM)).mean())
    conf_a = float((pa.max() - 1/EMBED_DIM) / (1 - 1/EMBED_DIM))

    return {
        "live_acc_normal": round(acc_n, 4), "live_acc_ice": round(acc_i, 4),
        "live_D_normal": round(Dn, 4), "live_D_ice": round(Di, 4),
        "live_D_gap": round(Di - Dn, 4), "anchor_conf": round(conf_a, 4),
    }


def main(n_each: int = 50000, epochs: int = 100, batch: int = 512,
         lr: float = 3e-4) -> None:
    t0 = time.time()
    torch.manual_seed(0)
    np.random.seed(0)

    print("═" * 65)
    print(f"  Multi-seed CLS GRU — Walker2d Terrain · policy={POLICY}")
    print(f"  {len(TRAIN_SEEDS_NORMAL)} train seeds/class · held-out eval seeds")
    print("═" * 65, flush=True)

    print(f"\n  Collecting {n_each} normal windows over {len(TRAIN_SEEDS_NORMAL)} seeds …", flush=True)
    normal_wins = collect_multiseed(n_each, FRICTION_NORMAL, TRAIN_SEEDS_NORMAL)
    print(f"  Collecting {n_each} ice windows over {len(TRAIN_SEEDS_ICE)} seeds …", flush=True)
    ice_wins = collect_multiseed(n_each, FRICTION_ICE, TRAIN_SEEDS_ICE)
    print(f"  Collection done in {time.time()-t0:.0f}s", flush=True)

    windows = np.array(normal_wins + ice_wins, dtype=np.float32)
    labels  = np.array([0] * len(normal_wins) + [1] * len(ice_wins), dtype=np.int64)
    N = len(windows)
    print(f"  Total: {N} windows\n", flush=True)

    encoder = GRUProprioEncoder(OBS_DIM, HIDDEN_DIM, EMBED_DIM, SEQ_LEN)
    opt = torch.optim.Adam(encoder.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr / 20)

    for epoch in range(epochs):
        idx = np.random.permutation(N)
        t_ce = t_acc = 0.0
        steps = 0
        for start in range(0, N - batch, batch):
            bi = idx[start:start + batch]
            w = torch.from_numpy(windows[bi])
            y = torch.from_numpy(labels[bi])
            z = encoder(w)
            ce = F.cross_entropy(z, y)
            loss = ce + 0.5 * variance_loss(z)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            opt.step()
            t_ce += ce.item()
            t_acc += (z.argmax(dim=1) == y).float().mean().item()
            steps += 1
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1:3d}  ce={t_ce/steps:.4f}  acc={t_acc/steps:.3f}  "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    print("\n  Held-out live verification …", flush=True)
    stats = live_eval(encoder)
    for k, v in stats.items():
        print(f"    {k:<18} {v}")

    ok = (stats["live_acc_normal"] > 0.8 and stats["live_acc_ice"] > 0.8
          and stats["live_D_gap"] > 0.20 and stats["live_D_normal"] < 0.25)
    print(f"\n  {'✓ Ready for fleet experiment' if ok else '✗ Live separation insufficient'}")

    model_path = _ROOT / "models" / "pav" / f"gru_cls_multiseed_{POLICY}.pt"
    model_path.parent.mkdir(exist_ok=True)
    torch.save({
        "encoder_state": encoder.state_dict(),
        "obs_dim": OBS_DIM, "hidden_dim": HIDDEN_DIM,
        "embed_dim": EMBED_DIM, "seq_len": SEQ_LEN,
        "stats": stats, "policy": POLICY,
        "train_seeds": {"normal": TRAIN_SEEDS_NORMAL, "ice": TRAIN_SEEDS_ICE},
        "eval_seeds": {"normal": EVAL_SEEDS_NORMAL, "ice": EVAL_SEEDS_ICE},
    }, model_path)
    print(f"  Saved → {model_path.relative_to(_ROOT)}  [{time.time()-t0:.0f}s total]\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--policy", choices=["random", "cpg"], default="random")
    a = ap.parse_args()
    POLICY = a.policy
    main(a.n, a.epochs, a.batch, a.lr)
