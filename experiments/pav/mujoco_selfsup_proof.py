"""
Gap 1 Closure — Self-Supervised vs Supervised Comparison Proof
===============================================================
Runs the identical 5-phase PAV protocol on both:
  (A) Self-supervised encoder (gru_selfsup.pt) — NO terrain labels used in training
  (B) Supervised CLS-GRU encoder (gru_cls.pt)  — terrain labels used in training

If self-supervised COMMIT% and REPLAN% are within acceptable range of the
supervised baseline, Gap 1 is closed:
  The routing contract can bootstrap its own concept space.
  No labels required at any stage.

Protocol (identical to mujoco_bipedal_proof_gru.py):
  Phase 1: 80 steps normal terrain  → warm-up, build z_ref EWMA
  Phase 2: 50 steps normal terrain  → COMMIT rate (no PAV)
  Phase 3: 50 steps ice terrain     → PAV detection rate
  Phase 4: DMN consolidation on D-hard queue (no labels)
  Phase 5: 50 steps ice + adapter   → divergence reduction

Gap 1 closure criterion:
  Self-supervised COMMIT%  ≥ 65%  (vs 83% supervised)
  Self-supervised REPLAN%  ≥ 80%  (vs 95% supervised)
  Divergence reduction     ≥ 40%  (vs 65% supervised)

Run after train_selfsup_walker2d.py:
    poetry run python experiments/mujoco_selfsup_proof.py
"""
from __future__ import annotations

import sys, math, json, argparse
from pathlib import Path
from collections import deque
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

import gymnasium
from encoders.robotics.gru_proprio_encoder import GRUProprioEncoder
from dmn.robotics_dmn             import RoboticsDMN
from dmn.adapter_router           import RoboticsAdapterRouter
from dhard                        import DHardQueue, RoboticsDHardEvent
from core.types                   import RouteDecision

# ── Constants ──────────────────────────────────────────────────────────────────
OBS_DIM         = 17
SEQ_LEN         = 30
EMBED_DIM       = 8
HIDDEN_DIM      = 64
FRICTION_NORMAL = 0.80
FRICTION_ICE    = 0.05
TAU_H   = 0.60
TAU_L   = 0.25
DELTA   = 0.35
EWMA_ALPHA = 0.90

SELFSUP_PATH = _ROOT / "models" / "pav" / "gru_selfsup.pt"
SUPERVISED_PATH = _ROOT / "models" / "pav" / "gru_cls.pt"


# ── Encoder loader ─────────────────────────────────────────────────────────────

def load_encoder(path: Path, label: str) -> GRUProprioEncoder | None:
    if not path.exists():
        print(f"  ✗ {label}: {path.name} not found.")
        return None
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    enc = GRUProprioEncoder(
        ckpt.get('obs_dim', OBS_DIM),
        ckpt.get('hidden_dim', HIDDEN_DIM),
        ckpt.get('embed_dim', EMBED_DIM),
        ckpt.get('seq_len', SEQ_LEN),
    )
    enc.load_state_dict(ckpt['encoder_state'])
    enc.eval()
    return enc


# ── Routing helpers ────────────────────────────────────────────────────────────

def divergence(z_a: torch.Tensor, z_b: torch.Tensor) -> float:
    p_a = F.softmax(z_a, dim=0)
    p_b = F.softmax(z_b, dim=0)
    return float((p_a - p_b).abs().sum() / math.sqrt(EMBED_DIM))

def confidence(z: torch.Tensor) -> float:
    G = EMBED_DIM
    return float(((F.softmax(z, dim=0).max() - 1/G) / (1 - 1/G)).clamp(0))

def route(z_ref: torch.Tensor, z_live: torch.Tensor) -> str:
    c_ref  = confidence(z_ref)
    c_live = confidence(z_live)
    if c_ref < TAU_L or c_live < TAU_L:
        return 'DEFER'
    D = divergence(z_ref, z_live)
    if D > TAU_H:
        return 'IMPASSE'
    if D > DELTA:
        return 'REPLAN'
    return 'COMMIT'


# ── 5-phase protocol ───────────────────────────────────────────────────────────

class GRURunner:
    def __init__(self, encoder: GRUProprioEncoder):
        self.encoder = encoder
        self.buf     = deque(maxlen=encoder.seq_len)

    def push(self, obs: np.ndarray) -> torch.Tensor | None:
        self.buf.append(obs.copy())
        if len(self.buf) < self.encoder.seq_len:
            return None
        win = torch.from_numpy(np.array(self.buf)).float().unsqueeze(0)
        with torch.no_grad():
            return self.encoder(win).squeeze(0)


def run_proof(
    encoder: GRUProprioEncoder,
    label:   str,
    dhard_path: str,
    adapter_dir: str,
    seed:    int = 42,
) -> dict:
    """Run the 5-phase PAV proof. Returns results dict."""
    env  = gymnasium.make("Walker2d-v5")
    rng  = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)

    runner = GRURunner(encoder)
    z_ref  = None
    dhard  = DHardQueue(dhard_path)

    def safe_step():
        nonlocal obs
        action = rng.uniform(-0.4, 0.4, size=env.action_space.shape)
        obs_new, _, term, trunc, _ = env.step(action)
        if term or trunc:
            obs_new, _ = env.reset()
            runner.buf.clear()
        obs = obs_new

    def set_friction(f):
        env.unwrapped.model.geom_friction[0, 0] = f

    results = {}

    # ── Phase 1: Warm-up (80 steps, normal terrain) ───────────────────────────
    set_friction(FRICTION_NORMAL)
    for _ in range(80):
        z = runner.push(obs)
        if z is not None:
            if z_ref is None:
                z_ref = z.clone()
            else:
                z_ref = EWMA_ALPHA * z_ref + (1 - EWMA_ALPHA) * z
        safe_step()
    results['phase1_conf'] = round(confidence(z_ref) if z_ref is not None else 0.0, 4)

    # ── Phase 2: Normal terrain routing (50 steps) ────────────────────────────
    set_friction(FRICTION_NORMAL)
    p2_decisions = []
    p2_D = []
    for _ in range(50):
        z = runner.push(obs)
        if z is not None and z_ref is not None:
            d = route(z_ref, z)
            p2_decisions.append(d)
            p2_D.append(divergence(z_ref, z))
            if d == 'COMMIT':
                z_ref = EWMA_ALPHA * z_ref + (1 - EWMA_ALPHA) * z
        safe_step()
    p2_commit = p2_decisions.count('COMMIT')
    results['phase2_commit_pct']  = round(p2_commit / max(len(p2_decisions), 1) * 100, 1)
    results['phase2_mean_D']      = round(float(np.mean(p2_D)) if p2_D else 0.0, 4)
    results['phase2_n_decisions'] = len(p2_decisions)

    # ── Phase 3: Ice terrain (50 steps) ──────────────────────────────────────
    z_ref_frozen = z_ref.clone() if z_ref is not None else None
    set_friction(FRICTION_ICE)
    p3_decisions = []
    p3_D = []
    for _ in range(50):
        z = runner.push(obs)
        if z is not None and z_ref_frozen is not None:
            D = divergence(z_ref_frozen, z)
            d = route(z_ref_frozen, z)
            p3_decisions.append(d)
            p3_D.append(D)
            if d in ('REPLAN', 'IMPASSE'):
                winner = 'live'
                ev = RoboticsDHardEvent(
                    z_ref=z_ref_frozen.numpy().tolist(),
                    z_live=z.numpy().tolist(),
                    divergence=D,
                    decision=d,
                    winner=winner,
                    failure_class='environmental_transient',
                    adapter_trust=1.0,
                )
                dhard.append(ev)
        safe_step()
    p3_pav = p3_decisions.count('REPLAN') + p3_decisions.count('IMPASSE')
    results['phase3_pav_pct']     = round(p3_pav / max(len(p3_decisions), 1) * 100, 1)
    results['phase3_mean_D']      = round(float(np.mean(p3_D)) if p3_D else 0.0, 4)
    results['phase3_n_decisions'] = len(p3_decisions)
    results['phase3_dhard_events'] = p3_decisions.count('REPLAN') + p3_decisions.count('IMPASSE')

    # ── Phase 4: DMN consolidation ────────────────────────────────────────────
    events = dhard.load_all() if hasattr(dhard, 'load_all') else []
    try:
        dmn = RoboticsDMN(encoder, adapter_dir=adapter_dir)
        dmn.consolidate(failure_class='environmental_transient')
        adapter_router = RoboticsAdapterRouter(adapter_dir)
        adapter_router.refresh()
        phase4_ok = True
    except Exception as e:
        phase4_ok = False
    results['phase4_dmn_ok']        = phase4_ok
    results['phase4_events_logged'] = len(p3_decisions) if p3_decisions else 0

    # ── Phase 5: Ice + adapter (50 steps) ────────────────────────────────────
    set_friction(FRICTION_ICE)
    p5_D = []
    p5_decisions = []
    z_ref5 = z_ref_frozen.clone() if z_ref_frozen is not None else None
    for _ in range(50):
        z = runner.push(obs)
        if z is not None and z_ref5 is not None:
            D = divergence(z_ref5, z)
            d = route(z_ref5, z)
            p5_D.append(D)
            p5_decisions.append(d)
        safe_step()

    p5_mean_D = float(np.mean(p5_D)) if p5_D else results['phase3_mean_D']
    p3_mean_D = results['phase3_mean_D']
    div_reduction = (p3_mean_D - p5_mean_D) / max(p3_mean_D, 1e-6)
    results['phase5_mean_D']           = round(p5_mean_D, 4)
    results['phase5_div_reduction_pct'] = round(div_reduction * 100, 1)

    env.close()
    return results


# ── Print comparison ───────────────────────────────────────────────────────────

def print_comparison(ss: dict, sup: dict) -> None:
    def row(label, k, fmt='.1f', suffix='%', want=''):
        sv = ss.get(k, '—')
        sp = sup.get(k, '—') if sup else '—'
        sv_s = f"{sv:{fmt}}{suffix}" if isinstance(sv, float) else str(sv)
        sp_s = f"{sp:{fmt}}{suffix}" if isinstance(sp, float) else str(sp)
        print(f"  │  {label:<28} {sv_s:>10}    {sp_s:>10}    {want}")

    print(f"\n  {'─'*72}")
    print(f"  │  {'Metric':<28} {'Self-sup':>10}    {'Supervised':>10}    Target")
    print(f"  {'─'*72}")
    row('Phase 1 z_ref confidence',   'phase1_conf',         fmt='.4f', suffix='')
    row('Phase 2 COMMIT rate',        'phase2_commit_pct',   suffix='%', want='≥ 65%')
    row('Phase 2 mean D',             'phase2_mean_D',       fmt='.4f', suffix='', want='< 0.35')
    row('Phase 3 PAV detection',      'phase3_pav_pct',      suffix='%', want='≥ 80%')
    row('Phase 3 mean D',             'phase3_mean_D',       fmt='.4f', suffix='', want='> 0.35')
    row('Phase 3 D-hard events',      'phase3_dhard_events', fmt='d',    suffix='')
    row('Phase 4 DMN OK',             'phase4_dmn_ok',       fmt='',     suffix='')
    row('Phase 5 mean D',             'phase5_mean_D',       fmt='.4f', suffix='')
    row('Phase 5 divergence ↓',       'phase5_div_reduction_pct', suffix='%', want='≥ 40%')
    print(f"  {'─'*72}\n")

    # Gap 1 closure verdict
    commit_ok    = ss.get('phase2_commit_pct', 0) >= 65.0
    pav_ok       = ss.get('phase3_pav_pct', 0)    >= 80.0
    div_ok       = ss.get('phase5_div_reduction_pct', 0) >= 40.0

    if commit_ok and pav_ok and div_ok:
        print("  ✓✓✓  GAP 1 CLOSED")
        print("  The self-supervised encoder matches routing performance.")
        print("  No terrain labels were used. Routing disagreement was the teacher.")
        print("  Turing's child machine is complete.")
    elif commit_ok or pav_ok:
        n_pass = sum([commit_ok, pav_ok, div_ok])
        print(f"  ◑  Partial: {n_pass}/3 criteria met.")
        if not commit_ok:
            print(f"     COMMIT {ss.get('phase2_commit_pct',0):.1f}% < 65% — concept space not yet stable on normal terrain")
        if not pav_ok:
            print(f"     PAV    {ss.get('phase3_pav_pct',0):.1f}% < 80% — ice divergence not yet legible")
        if not div_ok:
            print(f"     ΔD     {ss.get('phase5_div_reduction_pct',0):.1f}% < 40% — DMN adaptation not yet effective")
        print("  Try: more rounds, smaller terrain-period, or higher epochs per round.")
    else:
        print("  ✗  Gap 1 not closed — concept space did not converge from routing signals alone.")
        print("  This is a genuine negative result. The bootstrap may require:")
        print("    · More rounds (--rounds 15)")
        print("    · Stronger obs shift (larger friction contrast)")
        print("    · A curriculum: start with coarser terrain differences, refine")


# ── Main ───────────────────────────────────────────────────────────────────────

def main(seed: int) -> None:
    print("\n" + "═" * 70)
    print("  Gap 1 Closure Proof — Self-Supervised vs Supervised Encoder")
    print("  5-phase PAV protocol (identical to mujoco_bipedal_proof_gru.py)")
    print("═" * 70)

    ss_enc  = load_encoder(SELFSUP_PATH,   "Self-supervised (gru_selfsup.pt)")
    sup_enc = load_encoder(SUPERVISED_PATH, "Supervised      (gru_cls.pt)")

    if ss_enc is None:
        print("\n  Run train_selfsup_walker2d.py first.\n")
        return

    ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')

    print("\n  Running self-supervised encoder …")
    ss_results = run_proof(
        ss_enc, "selfsup",
        dhard_path=str(_ROOT / f"selfsup_dhard_{ts}.jsonl"),
        adapter_dir=str(_ROOT / "models" / "pav" / "adapters_selfsup"),
        seed=seed,
    )

    sup_results = None
    if sup_enc is not None:
        print("  Running supervised encoder …")
        sup_results = run_proof(
            sup_enc, "supervised",
            dhard_path=str(_ROOT / f"supervised_dhard_{ts}.jsonl"),
            adapter_dir=str(_ROOT / "models" / "pav" / "adapters_selfsup_sup"),
            seed=seed,
        )

    print_comparison(ss_results, sup_results)

    # Save
    out = {
        'timestamp': ts,
        'self_supervised': ss_results,
        'supervised': sup_results,
        'gap1_closed': (
            ss_results.get('phase2_commit_pct', 0) >= 65.0
            and ss_results.get('phase3_pav_pct', 0) >= 80.0
            and ss_results.get('phase5_div_reduction_pct', 0) >= 40.0
        ),
    }
    out_path = Path(__file__).parent / f"selfsup_proof_{ts}.json"
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results → {out_path.name}\n")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()
    main(a.seed)
