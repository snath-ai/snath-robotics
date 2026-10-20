"""
Long Horizon Check -- does any correction actually make the open-loop CPG
walk stably, or does every one of them just delay the same inevitable
fall past whatever window happened to be measured?
=============================================================================
STATUS.md 2026-10-20 (VII): every "falls=0/8" verdict in this investigation
(fixed_scale_check.py, persist_magnitude_search.py, the original manual
result) was measured in a window that completes around real step ~50 --
right before the SAME trajectory, continued a few more steps, actually
falls. No test in this entire arc has run long enough to find out whether
a correction genuinely stabilizes the walk or merely postpones the fall.

This file answers that directly, with no FLEET/PERSIST machinery at all --
no encoder, no divergence router, no z_ref. None of that is needed to ask
"does this policy, with this fixed action offset, ever actually fall." Pure
physics: step the env, watch `terminated` (a real fall -- unhealthy height
or angle), and correctly do NOT count `truncated` (hitting Walker2d-v5's
own registered 1000-step time limit) as a fall -- the two were conflated
as "term or trunc" in every earlier probe this session, which didn't matter
when everything fell in the first 300 steps, but matters now.

Tests, both frictions, all 8 seeds used throughout this investigation, up
to the environment's own 1000-step limit:
  - no correction (the open-loop baseline, already known to fall ~36-47
    regardless of friction)
  - the expert-derived direction (extract_correction(), normalized to a
    unit vector) at several raw magnitudes
  - the randomly-discovered direction (discover_direction.py's +d6,
    already unit-norm) at the SAME raw magnitudes, for a fair comparison

Run:
    .venv/bin/python experiments/fleet/long_horizon_check.py
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
from experiments.fleet.policies import CPGPolicy
from experiments.fleet.extraction import extract_correction

PPO_CHECKPOINT = "/Users/aadithya/Desktop/Lar_Main/lar_jepa/models/walker2d_ppo.zip"
SEEDS = [1000 + s * 13 for s in range(1, 9)]  # same seeds used throughout this investigation
MAX_STEPS = 1000  # Walker2d-v5's own registered TimeLimit -- confirmed via gymnasium.spec()
MAGNITUDES = [0.17, 0.35, 0.69, 1.4, 2.8, 5.5]  # raw offset norm, same scale for both directions
                                                  # (1x,2x,4x,8x,16x,32x of the expert's own extracted norm)

# the discovered direction from discover_direction.py's random search (K=8, seed 777,
# probe_scale=0.6) -- fixed here rather than re-running the search, since this script's
# job is to check the MAGNITUDE question for a direction already found, not re-discover it
DISCOVERED_DIR = np.array([0.4878, -0.1155, 0.6561, 0.5003, 0.1945, -0.1735], dtype=np.float32)
DISCOVERED_DIR = DISCOVERED_DIR / np.linalg.norm(DISCOVERED_DIR)  # confirm/enforce unit norm


def run_one(seed, friction, delta, max_steps=MAX_STEPS):
    """Fresh env, fixed offset active from step 1. Returns (fell: bool,
    step: int) -- step is the fall step if fell, else the step reached
    (max_steps if it survived the full horizon)."""
    env = gymnasium.make("Walker2d-v5")
    policy = CPGPolicy(np.random.default_rng(seed))
    obs, _ = env.reset(seed=seed)
    set_friction(env, friction)
    policy.reset()
    for i in range(max_steps):
        action = np.clip(policy(obs) + delta, -1.0, 1.0).astype(np.float32)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated:
            env.close()
            return True, i + 1
        if truncated:
            # hit Walker2d-v5's own TimeLimit -- NOT a fall, genuinely survived
            env.close()
            return False, i + 1
    env.close()
    return False, max_steps


def sweep(label, direction, magnitudes):
    print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")
    for friction, fname in [(FRICTION_NORMAL, "normal"), (FRICTION_ICE, "ice")]:
        for mag in magnitudes:
            delta = direction * mag if direction is not None else np.zeros(6)
            falls, fall_steps, survived = 0, [], 0
            for seed in SEEDS:
                fell, step = run_one(seed, friction, delta)
                if fell:
                    falls += 1
                    fall_steps.append(step)
                else:
                    survived += 1
            print(f"  friction={fname:6s} magnitude={mag:5.2f}: "
                  f"fell={falls}/{len(SEEDS)}  survived_to_{MAX_STEPS}={survived}/{len(SEEDS)}  "
                  f"fall_steps={fall_steps}")
            if direction is None:
                break  # no-correction baseline doesn't depend on magnitude


def main():
    print("=" * 78)
    print("LONG HORIZON CHECK -- does any correction actually stabilize the walk,")
    print(f"or does every one just delay the fall? Real env.terminated only (NOT")
    print(f"truncated -- Walker2d-v5's own {MAX_STEPS}-step TimeLimit is not a fall),")
    print(f"up to {MAX_STEPS} real steps, {len(SEEDS)} seeds, both frictions.")
    print("=" * 78)

    sweep("NO CORRECTION (open-loop CPG baseline)", None, [0.0])

    print("\nExtracting the expert direction (same verified function used throughout).")
    env = gymnasium.make("Walker2d-v5")
    model = PPO.load(PPO_CHECKPOINT)

    def ppo_action(o):
        a, _ = model.predict(o, deterministic=True)
        return np.asarray(a, dtype=np.float32)

    def ppo_reset():
        pass

    expert = extract_correction(
        ppo_action, ppo_reset, env,
        lambda e: set_friction(e, FRICTION_ICE),
        lambda e: set_friction(e, FRICTION_NORMAL),
        n_seeds=8, n_steps=200, seed_base=1,
    )
    env.close()
    expert_raw = np.where(expert.trustworthy_dims(), expert.delta, 0.0).astype(np.float32)
    expert_unit = expert_raw / np.linalg.norm(expert_raw)
    print(f"Expert direction (unit-norm): {[round(float(x), 4) for x in expert_unit]}")

    sweep("EXPERT-DERIVED DIRECTION (unit-norm) x raw magnitude", expert_unit, MAGNITUDES)
    sweep("RANDOMLY-DISCOVERED DIRECTION (unit-norm, +d6) x raw magnitude", DISCOVERED_DIR, MAGNITUDES)

    print(f"\n{'=' * 78}\nSUMMARY\n{'=' * 78}")
    print(f"If 'survived_to_{MAX_STEPS}' is ever the full 8/8 at some magnitude on ICE: "
          f"a real, indefinitely-stable correction exists, and every earlier short-window "
          f"'0 falls' result, while measured over too short a window to prove it, happened "
          f"to be gesturing at something real.")
    print(f"If falls==8/8 at every magnitude, just later and later: no correction tested "
          f"stabilizes the open-loop CPG indefinitely -- every result in this arc has been "
          f"measuring delay, not stability, and that needs to be the honest conclusion.")


if __name__ == "__main__":
    main()
