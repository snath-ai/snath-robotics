"""
Wind Confound Check -- does the foot/leg effort ratio also shift under wind,
the same way FLEET's existing divergence signal does (its own published
class-capture failure)?
=============================================================================
foot_leg_ratio_check.py found a strong, clean ice signal: the foot/leg
effort ratio drops from 0.622 (normal) to 0.479 (ice), t=-20.69. Before
trusting that as an ice-specific signal, it has to clear the exact confound
that already broke FLEET's existing detector once: wind (ood_guard.py's own
10/20/30/40 N sweep) pushing divergence up in the same direction as ice,
so the two get mistaken for each other ("class capture").

Result: wind does NOT confound this signal -- it discriminates. Wind pushes
the ratio UP (0.653-0.664 across all four levels, all statistically solid),
the opposite direction from ice's drop to 0.479. Mechanistically sensible:
bracing against a lateral push means leaning harder on the feet (ratio up);
losing foot traction on ice means compensating through the legs instead
(ratio down). Two different causes, two different compensations, two
different directions -- not the same failure mode at different magnitudes.

Run:
    .venv/bin/python experiments/fleet/wind_confound_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

import gymnasium
from stable_baselines3 import PPO
from experiments.fleet.fleet_n2_transfer import set_friction, apply_wind, FRICTION_NORMAL

PPO_CHECKPOINT = "/Users/aadithya/Desktop/Lar_Main/lar_jepa/models/walker2d_ppo.zip"
FOOT_IDX = [2, 5]
LEG_IDX = [0, 1, 3, 4]
SEEDS = list(range(1, 11))
N_STEPS = 500
WIND_LEVELS = [10.0, 20.0, 30.0, 40.0]  # same sweep as ood_guard.py


def run_episode(env, model, friction, wind, seed, n_steps=N_STEPS):
    obs, _ = env.reset(seed=seed)
    set_friction(env, friction)
    ratios = []
    falls = 0
    for i in range(n_steps):
        if wind != 0.0:
            apply_wind(env, wind)  # re-assert every step -- xfrc_applied can reset
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
    print("WIND CONFOUND CHECK -- does the foot/leg ratio also drop under wind,")
    print("same as it does under ice? (FLEET's own known confound, re-tested here)")
    print("=" * 78)
    print("\nReference (foot_leg_ratio_check.py): normal=0.622, ice=0.479, diff=-0.143\n")

    normal_means = []
    for s in SEEDS:
        ratios, _ = run_episode(env, model, FRICTION_NORMAL, 0.0, s)
        normal_means.append(ratios.mean())
    normal_means = np.array(normal_means)
    print(f"normal (recomputed, same seeds): mean={normal_means.mean():.5f} "
          f"std={normal_means.std():.5f}")

    for wind in WIND_LEVELS:
        per_seed_means = []
        falls_total = 0
        for s in SEEDS:
            ratios, falls = run_episode(env, model, FRICTION_NORMAL, wind, s)
            per_seed_means.append(ratios.mean())
            falls_total += falls
        per_seed_means = np.array(per_seed_means)

        diff_vec = per_seed_means - normal_means
        se = diff_vec.std(ddof=1) / np.sqrt(len(SEEDS))
        t = diff_vec.mean() / se if se > 0 else float("inf")
        same_sign = all(d > 0 for d in diff_vec) or all(d < 0 for d in diff_vec)
        print(f"\nwind={wind:4.0f}N: mean={per_seed_means.mean():.5f} "
              f"std={per_seed_means.std():.5f} (falls over {len(SEEDS)} seeds: {falls_total})")
        print(f"  diff from normal={diff_vec.mean():+.5f}  "
              f"diff from ice(0.47946)={per_seed_means.mean()-0.47946:+.5f}")
        print(f"  paired t={t:+.2f}  all-seeds-same-sign={same_sign}  "
              f"clears-2SE={abs(diff_vec.mean()) > 2*se}")

    env.close()


if __name__ == "__main__":
    main()
