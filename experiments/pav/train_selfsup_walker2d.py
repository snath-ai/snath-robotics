"""
Self-Supervised GRU Encoder — Gap 1 Closure Experiment
=======================================================
Can the routing contract bootstrap its own concept space?

The question: can a random GRU encoder become a good proprioceptive
concept encoder using ONLY routing events as supervision — no terrain
labels, no cross-entropy loss, no human annotation of any kind?

Two bugs in the naive attempt revealed the path forward:

  Bug 1 — gradient starvation: storing embeddings as numpy constants
    means the encoder never receives a gradient. Fix: store raw obs
    windows and re-encode them fresh at training time.

  Bug 2 — routing circularity: a random encoder produces D ≈ 0.11
    uniformly (well below delta=0.35), so routing never fires REPLAN.
    Almost zero negative pairs → no signal to distinguish physics.
    Fix: bootstrap the initial signal from raw obs statistics (not
    encoder geometry), then switch to routing-based pairs as the
    encoder improves.

Approach — two-phase bootstrap:

  Phase A (rounds 1-3): Raw obs bootstrap
    The obs distribution changes dramatically between normal (stable
    height, periodic joints) and ice (dropping height, chaotic joints).
    A raw obs distance metric detects this without any encoder and
    without any terrain label:
        raw_dist = ||mean(window_t) - mean(window_ref)||_2
    If raw_dist > raw_threshold → likely physics change → negative pair
    If raw_dist < raw_threshold/3 → likely stable → positive pair
    Train encoder on these pairs. Gradient flows through fresh encodings.

  Phase B (rounds 4+): Routing-guided pairs
    Once the encoder has some separation, switch to routing-based pairs:
    COMMIT → positive, REPLAN/IMPASSE → negative.
    The routing signal now has real content because the encoder has been
    shaped toward physics separability.

If within-cluster scatter drops below 0.20 and D gap exceeds 0.25
after Phase B → Gap 1 is closed.

No terrain label is used at any point. The teacher is:
  Phase A: the raw physics of the obs stream
  Phase B: routing disagreement from a shaped concept space

Run:
    poetry run python experiments/train_selfsup_walker2d.py
    poetry run python experiments/train_selfsup_walker2d.py --rounds 12
"""
from __future__ import annotations

import sys, math, argparse, json
from pathlib import Path
from collections import deque
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

import gymnasium
from encoders.robotics.gru_proprio_encoder import GRUProprioEncoder

# ── Constants ──────────────────────────────────────────────────────────────────
OBS_DIM         = 17
SEQ_LEN         = 30
EMBED_DIM       = 8
HIDDEN_DIM      = 64
FRICTION_NORMAL = 0.80
FRICTION_ICE    = 0.05
MODEL_PATH      = _ROOT / "models" / "pav" / "gru_selfsup.pt"

TAU_H  = 0.60
TAU_L  = 0.25
DELTA  = 0.35
EWMA_A = 0.90

MARGIN      = 1.0    # push pairs must be at least this far apart (MSE space)
LAMBDA_VAR  = 0.50
RAW_THRESH  = 0.40   # raw obs distance threshold for Phase A


# ── Routing helpers ────────────────────────────────────────────────────────────

def divergence_t(z_a: torch.Tensor, z_b: torch.Tensor) -> float:
    return float(((F.softmax(z_a, dim=0) - F.softmax(z_b, dim=0)).abs().sum()
                  / math.sqrt(EMBED_DIM)))

def confidence_t(z: torch.Tensor) -> float:
    return float(((F.softmax(z, dim=0).max() - 1/EMBED_DIM)
                  / (1 - 1/EMBED_DIM)).clamp(0))


# ── Raw obs distance (no encoder needed) ──────────────────────────────────────

def raw_obs_dist(win_a: np.ndarray, win_b: np.ndarray) -> float:
    """L2 distance between mean obs vectors of two windows."""
    return float(np.linalg.norm(win_a.mean(axis=0) - win_b.mean(axis=0)))


# ── Rollout collection ─────────────────────────────────────────────────────────

def collect_pairs_raw(
    n_steps: int,
    terrain_period: int,
    seed: int,
    warmup: int = 80,
) -> dict:
    """
    Phase A: use raw obs distance to generate positive/negative pairs.
    No encoder. No terrain label. Physics provides the signal.
    """
    env = gymnasium.make("Walker2d-v5")
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    env.unwrapped.model.geom_friction[0, 0] = FRICTION_NORMAL

    buf = deque(maxlen=SEQ_LEN)
    pos_windows, pos_refs = [], []   # raw obs windows for positive pairs
    neg_windows, neg_refs = [], []   # raw obs windows for negative pairs

    # EWMA reference in raw obs space
    ref_win = None

    # True terrain for diagnostics (never used in loss or pairing decision)
    step_terrains = []

    for step in range(n_steps):
        period_idx    = step // terrain_period
        is_normal     = (period_idx % 2 == 0)
        friction      = FRICTION_NORMAL if is_normal else FRICTION_ICE
        env.unwrapped.model.geom_friction[0, 0] = friction
        step_terrains.append('normal' if is_normal else 'ice')

        buf.append(obs.copy())
        action = rng.uniform(-0.4, 0.4, size=env.action_space.shape)
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            obs, _ = env.reset()
            env.unwrapped.model.geom_friction[0, 0] = friction
            buf.clear()
            continue

        if len(buf) < SEQ_LEN:
            continue

        win = np.array(buf)

        # Build raw EWMA reference during warm-up
        if step < warmup:
            ref_win = win.copy() if ref_win is None else (
                0.90 * ref_win + 0.10 * win
            )
            continue

        if ref_win is None:
            ref_win = win.copy()
            continue

        d = raw_obs_dist(win, ref_win)

        if d < RAW_THRESH / 3:
            # Stable physics → positive pair (pull together)
            pos_windows.append(win.copy())
            pos_refs.append(ref_win.copy())
            # Update EWMA ref (stable)
            ref_win = 0.90 * ref_win + 0.10 * win
        elif d > RAW_THRESH:
            # Physics changed → negative pair (push apart)
            neg_windows.append(win.copy())
            neg_refs.append(ref_win.copy())
            # Don't update ref — it should stay anchored to stable physics

    env.close()

    # Diagnostics: check if pairing aligned with true terrain
    correct = 0
    n_judged = len(pos_windows) + len(neg_windows)
    # (rough: positive pairs should be mostly normal, negative pairs mostly ice)
    # We can't compute this accurately without tracking per-window terrain
    # but we can check that we have pairs at all

    return {
        'pos_windows': np.array(pos_windows, dtype=np.float32) if pos_windows else np.zeros((0, SEQ_LEN, OBS_DIM), dtype=np.float32),
        'pos_refs':    np.array(pos_refs,    dtype=np.float32) if pos_refs    else np.zeros((0, SEQ_LEN, OBS_DIM), dtype=np.float32),
        'neg_windows': np.array(neg_windows, dtype=np.float32) if neg_windows else np.zeros((0, SEQ_LEN, OBS_DIM), dtype=np.float32),
        'neg_refs':    np.array(neg_refs,    dtype=np.float32) if neg_refs    else np.zeros((0, SEQ_LEN, OBS_DIM), dtype=np.float32),
        'mode': 'raw',
    }


def collect_pairs_routing(
    encoder: GRUProprioEncoder,
    n_steps: int,
    terrain_period: int,
    seed: int,
    warmup: int = 80,
) -> dict:
    """
    Phase B: use routing contract to generate pairs.
    Encoder must already have some separation for this to work.
    Stores RAW OBS WINDOWS so gradient flows through encoder at training.
    """
    env = gymnasium.make("Walker2d-v5")
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    env.unwrapped.model.geom_friction[0, 0] = FRICTION_NORMAL

    buf   = deque(maxlen=SEQ_LEN)
    z_ref = None
    encoder.eval()

    pos_windows, pos_refs = [], []
    neg_windows, neg_refs = [], []
    ref_win = None   # raw obs EWMA for the reference window
    decisions, D_vals = [], []

    for step in range(n_steps):
        period_idx = step // terrain_period
        is_normal  = (period_idx % 2 == 0)
        friction   = FRICTION_NORMAL if is_normal else FRICTION_ICE
        env.unwrapped.model.geom_friction[0, 0] = friction

        buf.append(obs.copy())
        action = rng.uniform(-0.4, 0.4, size=env.action_space.shape)
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            obs, _ = env.reset()
            env.unwrapped.model.geom_friction[0, 0] = friction
            buf.clear()
            z_ref = None
            ref_win = None
            continue

        if len(buf) < SEQ_LEN:
            continue

        win = np.array(buf)
        win_t = torch.from_numpy(win).float().unsqueeze(0)
        with torch.no_grad():
            z_live = encoder(win_t).squeeze(0)

        if step < warmup:
            z_ref = z_live.clone() if z_ref is None else (
                EWMA_A * z_ref + (1 - EWMA_A) * z_live
            )
            ref_win = win.copy() if ref_win is None else (
                0.90 * ref_win + 0.10 * win
            )
            continue

        if z_ref is None:
            z_ref = z_live.clone()
            ref_win = win.copy()
            continue

        c_ref = confidence_t(z_ref)
        if c_ref < TAU_L:
            z_ref   = 0.5 * z_ref + 0.5 * z_live
            ref_win = 0.5 * ref_win + 0.5 * win
            decisions.append('DEFER')
            D_vals.append(0.0)
            continue

        D = divergence_t(z_ref, z_live)
        D_vals.append(D)

        if D > TAU_H:
            decisions.append('IMPASSE')
            neg_windows.append(win.copy())
            neg_refs.append(ref_win.copy())
        elif D > DELTA:
            decisions.append('REPLAN')
            neg_windows.append(win.copy())
            neg_refs.append(ref_win.copy())
            z_ref   = 0.98 * z_ref   + 0.02 * z_live
            ref_win = 0.98 * ref_win  + 0.02 * win
        else:
            decisions.append('COMMIT')
            pos_windows.append(win.copy())
            pos_refs.append(ref_win.copy())
            z_ref   = EWMA_A * z_ref   + (1 - EWMA_A) * z_live
            ref_win = 0.90 * ref_win + 0.10 * win

    env.close()

    n_neg = len(neg_windows)
    n_pos = len(pos_windows)
    n_def = decisions.count('DEFER')
    routing_acc = (decisions.count('COMMIT') + n_neg) / max(len(decisions), 1)

    return {
        'pos_windows': np.array(pos_windows, dtype=np.float32) if pos_windows else np.zeros((0, SEQ_LEN, OBS_DIM), dtype=np.float32),
        'pos_refs':    np.array(pos_refs,    dtype=np.float32) if pos_refs    else np.zeros((0, SEQ_LEN, OBS_DIM), dtype=np.float32),
        'neg_windows': np.array(neg_windows, dtype=np.float32) if neg_windows else np.zeros((0, SEQ_LEN, OBS_DIM), dtype=np.float32),
        'neg_refs':    np.array(neg_refs,    dtype=np.float32) if neg_refs    else np.zeros((0, SEQ_LEN, OBS_DIM), dtype=np.float32),
        'n_pos': n_pos, 'n_neg': n_neg, 'n_defer': n_def,
        'mean_D': float(np.mean(D_vals)) if D_vals else 0.0,
        'routing_acc': routing_acc,
        'mode': 'routing',
    }


# ── Training ───────────────────────────────────────────────────────────────────

def variance_loss(z: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    return F.relu(torch.tensor(gamma) - z.std(dim=0)).mean()


def train_round(
    encoder: GRUProprioEncoder,
    data: dict,
    epochs: int,
    batch: int,
    lr: float,
) -> float:
    """
    Train encoder on raw obs windows.
    Gradient flows through the encoder because we re-encode at every step.
    """
    pos_wins = data['pos_windows']
    pos_refs = data['pos_refs']
    neg_wins = data['neg_windows']
    neg_refs = data['neg_refs']

    n_pos = len(pos_wins)
    n_neg = len(neg_wins)
    if n_pos + n_neg < 4:
        return 0.0

    opt = torch.optim.Adam(encoder.parameters(), lr=lr, weight_decay=1e-4)
    encoder.train()
    total_loss = 0.0

    for epoch in range(epochs):
        # Sample batch
        n_each = min(batch // 2, n_pos, n_neg) if n_pos > 0 and n_neg > 0 else batch // 2
        n_pos_batch = min(n_each, n_pos)
        n_neg_batch = min(n_each, n_neg)

        loss = torch.tensor(0.0)
        all_z = []

        # Positive batch (pull: z_live → z_ref)
        if n_pos_batch > 0:
            idx = np.random.choice(n_pos, n_pos_batch, replace=False)
            w_live = torch.from_numpy(pos_wins[idx]).float()  # (B, SEQ, OBS)
            w_ref  = torch.from_numpy(pos_refs[idx]).float()

            z_live = encoder(w_live)          # (B, G) — gradient flows here
            with torch.no_grad():
                z_ref  = encoder(w_ref)       # stop-gradient on reference

            pos_loss = F.mse_loss(z_live, z_ref)
            loss = loss + pos_loss
            all_z.append(z_live)

        # Negative batch (push: z_live away from z_ref)
        if n_neg_batch > 0:
            idx = np.random.choice(n_neg, n_neg_batch, replace=False)
            w_live = torch.from_numpy(neg_wins[idx]).float()
            w_ref  = torch.from_numpy(neg_refs[idx]).float()

            z_live = encoder(w_live)
            with torch.no_grad():
                z_ref  = encoder(w_ref)

            dist_sq  = ((z_live - z_ref) ** 2).sum(dim=1)
            neg_loss = F.relu(MARGIN - dist_sq).mean()
            loss = loss + neg_loss
            all_z.append(z_live)

        # Variance (prevent collapse)
        if all_z:
            z_cat = torch.cat(all_z, dim=0)
            if len(z_cat) > 1:
                loss = loss + LAMBDA_VAR * variance_loss(z_cat)

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        opt.step()
        total_loss += loss.item()

    return total_loss / max(epochs, 1)


# ── Concept space quality ──────────────────────────────────────────────────────

@torch.no_grad()
def measure_quality(encoder: GRUProprioEncoder, n: int = 300) -> dict:
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
                zlist.append(encoder(win).squeeze(0))
            action = rng.uniform(-0.4, 0.4, size=env.action_space.shape)
            obs, _, term, trunc, _ = env.step(action)
            if term or trunc:
                obs, _ = env.reset()
                env.unwrapped.model.geom_friction[0, 0] = friction
                buf.clear()
    env.close()

    zn = torch.stack(z_normal[:n])
    zi = torch.stack(z_ice[:n])

    def cdist(a, b):
        s = F.cosine_similarity(
            a.unsqueeze(1).expand(-1, len(b), -1).reshape(-1, EMBED_DIM),
            b.unsqueeze(0).expand(len(a), -1, -1).reshape(-1, EMBED_DIM), dim=-1)
        return float(1 - s.mean())

    def conf(z):
        p = F.softmax(z, dim=-1)
        return float(((p.max(dim=-1).values - 1/EMBED_DIM) / (1 - 1/EMBED_DIM)).clamp(0).mean())

    wn, wi, bw = cdist(zn, zn), cdist(zi, zi), cdist(zn, zi)
    mn = F.normalize(zn.mean(0), dim=0)
    mi = F.normalize(zi.mean(0), dim=0)
    angle = math.degrees(math.acos(
        float(F.cosine_similarity(mn.unsqueeze(0), mi.unsqueeze(0)).clamp(-1, 1))
    ))

    z_v = zn[:80].mean(0)
    scale = zn[:80].std(dim=0).mean() / z_v.norm().clamp(min=1e-6)
    z_v = z_v * scale

    def router_D(ref, zs):
        return float(np.mean([
            float((F.softmax(ref, dim=0) - F.softmax(z, dim=0)).abs().sum()
                  / math.sqrt(EMBED_DIM))
            for z in zs[:100]
        ]))

    dn, di = router_D(z_v, zn), router_D(z_v, zi)

    return {
        'within_normal':  round(wn, 4),
        'within_ice':     round(wi, 4),
        'between':        round(bw, 4),
        'centroid_angle': round(angle, 1),
        'conf_normal':    round(conf(zn), 4),
        'conf_ice':       round(conf(zi), 4),
        'router_D_normal': round(dn, 4),
        'router_D_ice':    round(di, 4),
        'D_gap':           round(di - dn, 4),
    }


def print_quality(r: int, q: dict, q0: dict) -> None:
    def d(k): return f" ({q[k]-q0.get(k, q[k]):+.4f})" if q0 else ""
    print(f"     within normal: {q['within_normal']:.4f}{d('within_normal')}  "
          f"within ice: {q['within_ice']:.4f}{d('within_ice')}  "
          f"angle: {q['centroid_angle']:.1f}°")
    print(f"     conf normal: {q['conf_normal']:.4f}  "
          f"conf ice: {q['conf_ice']:.4f}  "
          f"D gap: {q['D_gap']:+.4f}  "
          f"D normal: {q['router_D_normal']:.4f}  D ice: {q['router_D_ice']:.4f}")


# ── Bootstrap ──────────────────────────────────────────────────────────────────

def bootstrap(
    n_rounds: int,
    steps: int,
    terrain_period: int,
    epochs: int,
    batch: int,
    lr: float,
    raw_rounds: int,
) -> tuple[GRUProprioEncoder, list[dict]]:

    print("\n" + "═" * 72)
    print("  Self-Supervised GRU Encoder — Two-Phase Routing Bootstrap")
    print(f"  No terrain labels. Phase A: raw obs signal. Phase B: routing signal.")
    print(f"  Rounds: {n_rounds}  (first {raw_rounds} raw, rest routing)")
    print("═" * 72)

    encoder = GRUProprioEncoder(OBS_DIM, HIDDEN_DIM, EMBED_DIM, SEQ_LEN)

    print(f"\n  Round 0 (random encoder) — baseline")
    encoder.eval()
    q0 = measure_quality(encoder)
    print_quality(0, q0, {})
    history = [{'round': 0, 'quality': q0}]

    for r in range(1, n_rounds + 1):
        use_raw = (r <= raw_rounds)
        mode    = "raw-obs" if use_raw else "routing"
        print(f"\n  ── Round {r}/{n_rounds}  [{mode}] ──────────────────────────────")

        if use_raw:
            data = collect_pairs_raw(steps, terrain_period, seed=r * 100)
            n_pos = len(data['pos_windows'])
            n_neg = len(data['neg_windows'])
            print(f"     Raw pairs: {n_pos} positive · {n_neg} negative")
        else:
            data = collect_pairs_routing(encoder, steps, terrain_period, seed=r * 100)
            n_pos = data['n_pos']
            n_neg = data['n_neg']
            print(f"     Routing pairs: {n_pos} positive · {n_neg} negative  "
                  f"(defers: {data['n_defer']}  mean D: {data['mean_D']:.3f})")

        if n_pos + n_neg < 4:
            print(f"     Too few pairs — skipping")
            continue

        avg_loss = train_round(encoder, data, epochs, batch, lr)
        print(f"     Training loss: {avg_loss:.4f}")

        encoder.eval()
        q = measure_quality(encoder)
        print_quality(r, q, q0)
        history.append({'round': r, 'quality': q, 'mode': mode,
                        'n_pos': n_pos, 'n_neg': n_neg})

        if (q['within_normal'] < 0.20 and q['within_ice'] < 0.20
                and q['D_gap'] > 0.25 and q['conf_normal'] > 0.40):
            print(f"\n  ✓ Concept space converged at round {r}.")
            break

    return encoder, history


# ── Main ───────────────────────────────────────────────────────────────────────

def main(rounds, steps, terrain_period, epochs, batch, lr, raw_rounds):
    encoder, history = bootstrap(
        n_rounds=rounds,
        steps=steps,
        terrain_period=terrain_period,
        epochs=epochs,
        batch=batch,
        lr=lr,
        raw_rounds=raw_rounds,
    )

    encoder.eval()
    final_q = history[-1]['quality']

    closed = (
        final_q['within_normal'] < 0.20
        and final_q['within_ice'] < 0.20
        and final_q['D_gap'] > 0.25
        and final_q['conf_normal'] > 0.40
    )

    print(f"\n{'═'*72}")
    print(f"  Final quality:")
    print(f"    Within normal:  {final_q['within_normal']:.4f}  (need < 0.20)")
    print(f"    Within ice:     {final_q['within_ice']:.4f}  (need < 0.20)")
    print(f"    D gap:          {final_q['D_gap']:+.4f}  (need > 0.25)")
    print(f"    Conf normal:    {final_q['conf_normal']:.4f}  (need > 0.40)")
    if closed:
        print(f"\n  ✓ GAP 1 CLOSED — routing disagreement bootstrapped concept space.")
    else:
        print(f"\n  ✗ Gap 1 not yet closed.")
        print(f"  Try: --rounds 15 --raw-rounds 5 --steps 3000")
    print(f"{'═'*72}\n")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    torch.save({
        'encoder_state': encoder.state_dict(),
        'obs_dim': OBS_DIM, 'hidden_dim': HIDDEN_DIM,
        'embed_dim': EMBED_DIM, 'seq_len': SEQ_LEN,
        'gap1_closed': closed, 'final_quality': final_q,
    }, MODEL_PATH)
    print(f"  Saved → {MODEL_PATH.relative_to(_ROOT)}")

    ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    hp = Path(__file__).parent / f"selfsup_proof_{ts}.json"
    with open(hp, 'w') as f:
        json.dump({'gap1_closed': closed, 'final_quality': final_q,
                   'n_rounds': len(history)}, f, indent=2)
    print(f"  Results → {hp.name}")
    if closed:
        print(f"  Next: poetry run python experiments/mujoco_selfsup_proof.py")
    print()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--rounds',          type=int,   default=10)
    ap.add_argument('--raw-rounds',      type=int,   default=4,
                    help='Rounds using raw obs signal before switching to routing')
    ap.add_argument('--steps',           type=int,   default=2000)
    ap.add_argument('--terrain-period',  type=int,   default=150)
    ap.add_argument('--epochs',          type=int,   default=60)
    ap.add_argument('--batch',           type=int,   default=256)
    ap.add_argument('--lr',              type=float, default=3e-4)
    a = ap.parse_args()
    main(a.rounds, a.steps, a.terrain_period, a.epochs, a.batch, a.lr,
         a.raw_rounds)
