"""
Action-Effort Check -- does the trained PPO policy work harder on ice, even
though correction_search_ppo.py showed its proprioceptive OUTCOME looks
indistinguishable from normal walking?
=============================================================================
correction_search_ppo.py found that PPO's divergence-on-ice reads as
confidently normal to FLEET's own detector (0.00038, vs. tau_low=0.25) --
the policy copes well enough that the outcome it produces carries no
visible trace of struggle. This file asks the next question: even if the
OUTCOME looks the same, is the EFFORT required to produce it different?
Total |action| magnitude, not the embedding, is the thing measured here.

Result (5 seeds, 200 real steps each, real corrected friction): ice effort
is higher than normal in 4/5 seeds, never lower, but the aggregate
difference (+0.058) does not clear its own pooled noise band (+/-0.094) at
this sample size -- inconclusive on its own. The per-joint breakdown is
the real finding: every thigh/leg joint's effort rises on ice, both feet's
effort falls -- a structured reallocation, not uniform extra effort. That
observation is what motivated foot_leg_ratio_check.py, which tests the
ratio directly with proper statistical power and finds it real
(t=-20.69, 9 df; see FLEET STATUS.md, 2026-10-20).

Run:
    .venv/bin/python experiments/fleet/action_effort_check.py
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
JOINTS = ["thigh_r", "leg_r", "foot_r", "thigh_l", "leg_l", "foot_l"]
SEEDS = [1, 2, 3, 4, 5]
N_STEPS = 200


def run_episode(env, model, friction, seed, n_steps=N_STEPS):
    obs, _ = env.reset(seed=seed)
    set_friction(env, friction)
    action_norms = []
    action_abs_sums = np.zeros(6)
    falls = 0
    for i in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        action_norms.append(float(np.linalg.norm(action)))
        action_abs_sums += np.abs(action)
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            falls += 1
            obs, _ = env.reset(seed=seed + i)
            set_friction(env, friction)
    return np.array(action_norms), action_abs_sums / n_steps, falls


def main():
    model = PPO.load(PPO_CHECKPOINT)
    env = gymnasium.make("Walker2d-v5")

    print("=" * 78)
    print("ACTION-EFFORT CHECK -- does PPO work harder on ice even though it doesn't fall?")
    print("=" * 78)

    normal_means, ice_means = [], []
    normal_per_joint, ice_per_joint = [], []

    for s in SEEDS:
        n_norms, n_joint, n_falls = run_episode(env, model, FRICTION_NORMAL, s)
        i_norms, i_joint, i_falls = run_episode(env, model, FRICTION_ICE, s)
        normal_means.append(n_norms.mean())
        ice_means.append(i_norms.mean())
        normal_per_joint.append(n_joint)
        ice_per_joint.append(i_joint)
        print(f"seed {s}: normal mean|action|={n_norms.mean():.4f} (falls={n_falls})  "
              f"ice mean|action|={i_norms.mean():.4f} (falls={i_falls})  "
              f"diff={i_norms.mean()-n_norms.mean():+.4f}")

    normal_means = np.array(normal_means)
    ice_means = np.array(ice_means)
    print(f"\nNORMAL: mean={normal_means.mean():.4f} std={normal_means.std():.4f} "
          f"across {len(SEEDS)} seeds")
    print(f"ICE:    mean={ice_means.mean():.4f} std={ice_means.std():.4f} "
          f"across {len(SEEDS)} seeds")
    diff = ice_means.mean() - normal_means.mean()
    pooled_std = np.sqrt(normal_means.std() ** 2 + ice_means.std() ** 2)
    print(f"Difference: {diff:+.4f}  (pooled std band: +/-{2*pooled_std:.4f})")
    print(f"Clears noise band: {'YES' if abs(diff) > 2 * pooled_std else 'NO'}")

    print("\nPer-joint mean |action| (normal vs ice, averaged over seeds):")
    n_pj = np.array(normal_per_joint).mean(axis=0)
    i_pj = np.array(ice_per_joint).mean(axis=0)
    for j, nj, ij in zip(JOINTS, n_pj, i_pj):
        print(f"  {j:8s}: normal={nj:.4f}  ice={ij:.4f}  diff={ij-nj:+.4f}")

    env.close()


if __name__ == "__main__":
    main()
