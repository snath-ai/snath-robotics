"""
Foot/Leg Effort Ratio -- the real positive finding of the 2026-10-20
investigation (FLEET STATUS.md).
=============================================================================
action_effort_check.py found total effort inconclusive at n=5 seeds, but a
structured per-joint pattern underneath it: every thigh/leg joint's effort
rises on ice, both feet's effort falls. This file tests that reallocation
directly, as a normalised ratio (foot effort / leg effort) rather than raw
magnitude, with proper statistical power -- 10 seeds, 500 real steps each,
paired by seed.

Result: normal mean ratio 0.622, ice mean ratio 0.479. Paired t-statistic
-20.69 (9 df), all 10 seeds the same sign, difference ten times its own
standard error. The strongest, cleanest signal found in this whole
investigation -- see wind_confound_check.py for the follow-up that checks
whether it's actually ice-specific rather than a generic disturbance
detector (it is: wind moves this ratio in the OPPOSITE direction).

Run:
    .venv/bin/python experiments/fleet/foot_leg_ratio_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

import gymnasium
from stable_baselines3 import PPO
from experiments.fleet.fleet_n2_transfer import set_friction, FRICTION_NORMAL, FRICTION_ICE

PPO_CHECKPOINT = "/Users/aadithya/Desktop/Lar_Main/lar_jepa/models/walker2d_ppo.zip"
# Walker2d-v5 action layout: [thigh_r, leg_r, foot_r, thigh_l, leg_l, foot_l]
FOOT_IDX = [2, 5]
LEG_IDX = [0, 1, 3, 4]
SEEDS = list(range(1, 11))
N_STEPS = 500


def run_episode(env, model, friction, seed, n_steps=N_STEPS):
    obs, _ = env.reset(seed=seed)
    set_friction(env, friction)
    ratios = []
    falls = 0
    for i in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        a = np.abs(action)
        foot_eff = a[FOOT_IDX].sum()
        leg_eff = a[LEG_IDX].sum()
        ratios.append(foot_eff / (leg_eff + 1e-6))
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            falls += 1
            obs, _ = env.reset(seed=seed + i)
            set_friction(env, friction)
    return np.array(ratios), falls


def main():
    model = PPO.load(PPO_CHECKPOINT)
    env = gymnasium.make("Walker2d-v5")

    print("=" * 78)
    print("FOOT/LEG EFFORT RATIO -- a normalised allocation signal, not raw magnitude")
    print("=" * 78)

    all_normal_ratios, all_ice_ratios = [], []
    per_seed_normal_mean, per_seed_ice_mean = [], []

    for s in SEEDS:
        n_ratios, n_falls = run_episode(env, model, FRICTION_NORMAL, s)
        i_ratios, i_falls = run_episode(env, model, FRICTION_ICE, s)
        all_normal_ratios.append(n_ratios)
        all_ice_ratios.append(i_ratios)
        per_seed_normal_mean.append(n_ratios.mean())
        per_seed_ice_mean.append(i_ratios.mean())
        print(f"seed {s:2d}: normal ratio mean={n_ratios.mean():.4f} std={n_ratios.std():.4f} "
              f"(falls={n_falls})  ice ratio mean={i_ratios.mean():.4f} "
              f"std={i_ratios.std():.4f} (falls={i_falls})")

    all_normal = np.concatenate(all_normal_ratios)
    all_ice = np.concatenate(all_ice_ratios)
    per_seed_normal_mean = np.array(per_seed_normal_mean)
    per_seed_ice_mean = np.array(per_seed_ice_mean)

    print(f"\nPOOLED over all {len(all_normal)} real steps per condition "
          f"({len(SEEDS)} seeds x {N_STEPS} steps):")
    print(f"  NORMAL: mean={all_normal.mean():.5f} std={all_normal.std():.5f}")
    print(f"  ICE:    mean={all_ice.mean():.5f} std={all_ice.std():.5f}")
    print(f"  Pooled-step difference: {all_ice.mean()-all_normal.mean():+.5f}")

    print(f"\nPER-SEED MEANS (the statistically honest comparison -- "
          f"{len(SEEDS)} independent samples per condition):")
    print(f"  NORMAL: mean={per_seed_normal_mean.mean():.5f} std={per_seed_normal_mean.std():.5f}")
    print(f"  ICE:    mean={per_seed_ice_mean.mean():.5f} std={per_seed_ice_mean.std():.5f}")
    paired_diffs = per_seed_ice_mean - per_seed_normal_mean
    se = paired_diffs.std(ddof=1) / np.sqrt(len(SEEDS))
    t_stat = paired_diffs.mean() / se if se > 0 else float("inf")
    print(f"  Mean paired difference (ice - normal), per seed: {paired_diffs.mean():+.5f}")
    print(f"  Per-seed paired diffs: {[round(float(x), 5) for x in paired_diffs]}")
    print(f"  Standard error of the paired difference: {se:.5f}")
    print(f"  t-statistic (paired, {len(SEEDS)-1} df): {t_stat:.2f}")
    print(f"  Clears 2-SE band: {'YES' if abs(paired_diffs.mean()) > 2*se else 'NO'}")
    same_sign = all(d > 0 for d in paired_diffs) or all(d < 0 for d in paired_diffs)
    print(f"  All {len(SEEDS)} seeds same sign: {'YES' if same_sign else 'NO'}")

    env.close()


if __name__ == "__main__":
    main()
