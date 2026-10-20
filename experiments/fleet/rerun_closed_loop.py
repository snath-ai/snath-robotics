"""
Rerun, Closed-Loop -- the correction-mechanism investigation redone with
the two foundational fixes this session surfaced: a weak policy that
actually has feedback (ReactiveCPGPolicy, not CPGPolicy), and a real
"normal" friction baseline (the environment's true default, not the
uniform FRICTION_NORMAL=0.80 that cut foot friction from 1.9 to 0.80
before any hazard was applied -- STATUS.md 2026-10-20 IX).
=============================================================================
Also drops the FLEET divergence signal as the scoring metric for discovery
and magnitude search. STATUS.md VII showed that proxy's short-window
measurement (driven by the GRU's sliding buffer filling its quota) was
itself a source of false "success" -- it isn't needed to answer "does
this correction help the robot not fall," which is the actual question.
This file scores everything by real survival time (env.terminated,
correctly distinct from env.truncated's own 1000-step TimeLimit), nothing
else, start to finish.

Steps:
  1. Baseline: does ReactiveCPGPolicy's survival gap between true-normal
     and ice actually exist and matter? (prerequisite -- if not, there's
     nothing for a correction to fix)
  2. Re-extract the expert (PPO) direction using extract_correction() --
     unchanged, real, already-verified function -- but with
     restore_default_friction in place of the old uniform "normal."
  3. Discover a direction for ReactiveCPGPolicy from scratch (random
     search, no expert), scored by real paired survival-time improvement
     on ice, not divergence.
  4. Long-horizon magnitude sweep for BOTH directions, both frictions
     (true-normal this time), all 8 seeds, up to the real 1000-step
     limit -- does any correction get ice-survival anywhere near
     ReactiveCPGPolicy's own true-normal survival time?

Run:
    .venv/bin/python experiments/fleet/rerun_closed_loop.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

import gymnasium
from stable_baselines3 import PPO
from experiments.fleet.fleet_n2_transfer import set_friction, FRICTION_ICE
from experiments.fleet.friction_fix import restore_default_friction
from experiments.fleet.policies import ReactiveCPGPolicy
from experiments.fleet.extraction import extract_correction

PPO_CHECKPOINT = "/Users/aadithya/Desktop/Lar_Main/lar_jepa/models/walker2d_ppo.zip"
SEEDS = [1000 + s * 13 for s in range(1, 9)]
ACT_DIM = 6
MAX_STEPS = 1000
SCORING_CAP = 500          # cap for the discovery/scoring phase -- faster iteration,
                            # still far beyond ReactiveCPGPolicy's typical ice fall point
K_DIRECTIONS = 8
PROBE_SCALE = 0.3           # ~1.5x ReactiveCPGPolicy's own base_amp=0.2 -- policy-relative,
                            # not reverse-engineered from any known-good answer
DIRECTION_RNG_SEED = 777
MAGNITUDES = [0.15, 0.3, 0.6, 1.0, 1.5, 2.2]


def run_survival(seed, friction_fn, delta, max_steps):
    """Fresh env, ReactiveCPGPolicy + fixed offset active from step 1.
    Returns (fell: bool, step: int) -- step reached, real terminated only."""
    env = gymnasium.make("Walker2d-v5")
    policy = ReactiveCPGPolicy(np.random.default_rng(seed))
    obs, _ = env.reset(seed=seed)
    friction_fn(env)
    policy.reset()
    for i in range(max_steps):
        action = np.clip(policy(obs) + delta, -1.0, 1.0).astype(np.float32)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated:
            env.close()
            return True, i + 1
        if truncated:
            env.close()
            return False, i + 1
    env.close()
    return False, max_steps


def sample_random_directions(rng, k, dim):
    v = rng.normal(size=(k, dim))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def main():
    print("=" * 78)
    print("RERUN, CLOSED-LOOP -- ReactiveCPGPolicy, real friction baseline,")
    print("survival-time scoring instead of the divergence proxy")
    print("=" * 78)

    print(f"\nStep 1: baseline survival, no correction, {len(SEEDS)} seeds, "
          f"up to {MAX_STEPS} steps, true-normal vs ice.")
    normal_fall, ice_fall = {}, {}
    for seed in SEEDS:
        fell_n, step_n = run_survival(seed, restore_default_friction, np.zeros(ACT_DIM), MAX_STEPS)
        fell_i, step_i = run_survival(seed, lambda e: set_friction(e, FRICTION_ICE), np.zeros(ACT_DIM), MAX_STEPS)
        normal_fall[seed] = step_n
        ice_fall[seed] = step_i
        print(f"  seed {seed}: normal={'FELL@'+str(step_n) if fell_n else 'survived '+str(step_n)}  "
              f"ice={'FELL@'+str(step_i) if fell_i else 'survived '+str(step_i)}")

    print("\nStep 2: re-extract the expert direction with the TRUE friction baseline "
          "(restore_default_friction instead of the old uniform 'normal').")
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
        restore_default_friction,
        n_seeds=8, n_steps=200, seed_base=1,
    )
    env.close()
    print(expert.summary())
    expert_raw = np.where(expert.trustworthy_dims(), expert.delta, 0.0).astype(np.float32)
    if np.linalg.norm(expert_raw) > 1e-9:
        expert_unit = expert_raw / np.linalg.norm(expert_raw)
    else:
        expert_unit = expert_raw
        print("  WARNING: no trustworthy dims survived -- expert direction is all zero.")
    print(f"Expert direction (unit-norm): {[round(float(x), 4) for x in expert_unit]}")

    print(f"\nStep 3: discover a direction for ReactiveCPGPolicy from scratch -- "
          f"K={K_DIRECTIONS} random unit directions + antithetic negatives, "
          f"probe_scale={PROBE_SCALE}, scored by REAL paired survival-time "
          f"improvement on ice (capped at {SCORING_CAP} steps for speed), not divergence.")
    direction_rng = np.random.default_rng(DIRECTION_RNG_SEED)
    base_dirs = sample_random_directions(direction_rng, K_DIRECTIONS, ACT_DIM)
    candidates = list(base_dirs) + [-d for d in base_dirs]
    labels = [f"+d{i}" for i in range(K_DIRECTIONS)] + [f"-d{i}" for i in range(K_DIRECTIONS)]

    baseline_ice_capped = {s: min(ice_fall[s], SCORING_CAP) for s in SEEDS}
    stats = {}
    for label, cand in zip(labels, candidates):
        diffs = []
        for seed in SEEDS:
            _, step = run_survival(seed, lambda e: set_friction(e, FRICTION_ICE),
                                    cand * PROBE_SCALE, SCORING_CAP)
            diffs.append(step - baseline_ice_capped[seed])
        diffs = np.array(diffs)
        mean = diffs.mean()
        se = diffs.std(ddof=1) / np.sqrt(len(diffs))
        t = mean / se if se > 1e-12 else float("inf") * np.sign(mean) if mean != 0 else 0.0
        no_harm = bool(np.all(diffs >= -1e-9))
        stats[label] = (mean, t, no_harm)
        print(f"  {label:4s}: mean_extra_survival={mean:+7.1f}  t={t:+.2f}  "
              f"no_harm={no_harm}  per_seed={list(diffs)}")

    qualifying = [(l, m) for l, (m, t, nh) in stats.items() if nh and abs(t) > 2.0 and m > 0]
    if qualifying:
        winner_label, winner_mean = max(qualifying, key=lambda x: x[1])
        winner_dir = candidates[labels.index(winner_label)]
        print(f"\nFOUND: {winner_label}, mean_extra_survival={winner_mean:+.1f} steps, "
              f"t={stats[winner_label][1]:+.2f}, never harmful.")
    else:
        winner_dir = None
        print(f"\nHONEST NEGATIVE: no candidate cleared no_harm AND |t|>2 AND mean>0.")

    print(f"\nStep 4: long-horizon magnitude sweep, both frictions (true-normal "
          f"this time), all {len(SEEDS)} seeds, up to {MAX_STEPS} real steps.")
    for name, direction in [("EXPERT", expert_unit), ("DISCOVERED", winner_dir)]:
        if direction is None or np.linalg.norm(direction) < 1e-9:
            print(f"\n  [{name}] skipped -- no usable direction.")
            continue
        print(f"\n  --- {name} direction ---")
        for mag in MAGNITUDES:
            delta = direction * mag
            for label, friction_fn in [("normal", restore_default_friction),
                                        ("ice", lambda e: set_friction(e, FRICTION_ICE))]:
                falls, steps = 0, []
                for seed in SEEDS:
                    fell, step = run_survival(seed, friction_fn, delta, MAX_STEPS)
                    if fell:
                        falls += 1
                    steps.append(step)
                print(f"    magnitude={mag:4.2f} friction={label:6s}: "
                      f"fell={falls}/{len(SEEDS)}  steps={steps}")

    print(f"\n{'=' * 78}\nSUMMARY\n{'=' * 78}")
    print("Compare each magnitude's ICE step counts above against Step 1's own "
          "true-normal baseline survival times. If a correction gets ice survival "
          "into the same range as (or close to) no-correction true-normal survival, "
          "that is the first real evidence of something resembling ice-specific "
          "competence transfer in this investigation. If ice always falls much "
          "sooner than normal regardless of correction or magnitude, the honest "
          "conclusion is the same as before: delay, not stability, this time on a "
          "policy that at least has a chance of the alternative being true.")


if __name__ == "__main__":
    main()
