"""
Correction Search, PPO policy -- does a reactive policy change the picture?
=============================================================================
v1/v2 (correction_search.py / correction_search_v2.py) found that under the
open-loop CPG gait, divergence-on-ice is saturated (0.706, std 0.00004 over
20 windows) and nothing tested -- constant offsets, direct amplitude scaling
-- moves it beyond noise. The suspected root cause, named at the time: CPG
never reads its own observation, so neither baseline behaviour nor any
correction on top of it can be state-dependent. This file swaps in the real
trained PPO policy (lar_jepa/models/walker2d_ppo.zip, 1.5M real timesteps,
reward 4056 isolated eval) in place of CPG -- a policy that is reactive by
construction -- and reruns the same measurement, honestly, in two stages:

  1. Does divergence-on-ice even look different (less saturated, more
     variable) under a policy that responds to what it senses, with NO
     correction at all yet? That's the first real test of the hypothesis,
     before adding any correction.
  2. If so, try corrections on top: a constant offset (same class as
     before) and a multiplicative gain on the policy's own output (the
     PPO-equivalent of CPG's amplitude scaling -- "be more/less
     aggressive" rather than "walk differently").

Same methodology as v2: settle period before measuring, N_WINDOWS averaged
per condition, judged against the baseline's own noise floor, not a bare
rate from one trial.

Run:
    .venv/bin/python experiments/fleet/correction_search_ppo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

import gymnasium
from stable_baselines3 import PPO
from divergence_router import DivergenceRouter
from experiments.fleet.fleet_n2_transfer import (
    GRURunner, load_base_encoder, set_friction,
    FRICTION_NORMAL, FRICTION_ICE, DEFAULT_MODEL,
    EWMA_ALPHA, WARMUP_STEPS,
)
from experiments.fleet.policies import CPGPolicy
from experiments.fleet.correction_search_v2 import collect_windows, N_WINDOWS, SETTLE_STEPS

PPO_CHECKPOINT = "/Users/aadithya/Desktop/Lar_Main/lar_jepa/models/walker2d_ppo.zip"
ACT_DIM = 6


class PPOPolicyWrapper:
    """Matches CPGPolicy's calling convention (policy(obs) -> action,
    policy.reset() no-op) so it drops straight into collect_windows()."""
    def __init__(self, model, gain=1.0, offset=None):
        self.model = model
        self.gain = gain
        self.offset = offset if offset is not None else np.zeros(ACT_DIM, dtype=np.float32)

    def reset(self):
        pass  # PPO is stateless across steps (MlpPolicy, no recurrent state)

    def __call__(self, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        action = np.asarray(action, dtype=np.float32) * self.gain + self.offset
        return np.clip(action, -1.0, 1.0).astype(np.float32)


def main():
    print("=" * 78)
    print("CORRECTION SEARCH, PPO POLICY -- is a reactive policy different at all?")
    print("=" * 78)

    encoder = load_base_encoder(DEFAULT_MODEL)
    runner = GRURunner(encoder)
    env = gymnasium.make("Walker2d-v5")
    router = DivergenceRouter(tau_high=0.60, tau_low=0.25, delta=0.35)
    ppo_model = PPO.load(PPO_CHECKPOINT)
    print(f"\nLoaded real PPO checkpoint: {PPO_CHECKPOINT}")

    print(f"\nStep 1: warmup on normal friction ({WARMUP_STEPS} steps) -> EWMA anchor. "
          f"Using the SAME CPG warmup as v1/v2 for a comparable anchor -- only the "
          f"ice-phase policy changes.")
    warmup_policy = CPGPolicy(np.random.default_rng(42))
    obs, _ = env.reset(seed=42)
    set_friction(env, FRICTION_NORMAL)
    z_ref = None
    for _ in range(WARMUP_STEPS):
        z = runner.push(obs)
        if z is not None:
            z = z.detach()
            z_ref = z if z_ref is None else EWMA_ALPHA * z_ref + (1 - EWMA_ALPHA) * z
        action = warmup_policy(obs)
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            obs, _ = env.reset()
            set_friction(env, FRICTION_NORMAL)
            warmup_policy.reset()
    print("  z_ref ready.")

    print(f"\nStep 2: PPO baseline on ice, NO correction, {N_WINDOWS} windows, "
          f"{SETTLE_STEPS}-step settle -- the real first test of the hypothesis.")
    ppo_base = PPOPolicyWrapper(ppo_model)
    base_divs, base_resets, base_steps = collect_windows(
        env, runner, ppo_base, router, z_ref, ppo_base,
        N_WINDOWS, reset_seed_base=100, settle_steps=SETTLE_STEPS,
    )
    base_mean = float(np.mean(base_divs))
    base_std = float(np.std(base_divs))
    print(f"  PPO baseline: mean={base_mean:.5f} std={base_std:.5f} "
          f"({base_steps} steps, {base_resets} falls)")
    print(f"  For comparison, CPG's baseline was mean=0.70635 std=0.00004 "
          f"(correction_search_v2.py)")
    print(f"  All {len(base_divs)} raw readings: {[round(d,5) for d in base_divs]}")

    results = {}

    print(f"\nStep 3a: constant offsets on top of PPO (scale 0.1 / 0.2 / 0.4), "
          f"same candidate class as before, now on a reactive base policy.")
    for scale in [0.1, 0.2, 0.4]:
        d_rng = np.random.default_rng(7)
        direction = d_rng.normal(0, 1, size=ACT_DIM)
        direction = (direction / (np.linalg.norm(direction) + 1e-8) * scale).astype(np.float32)
        trial = PPOPolicyWrapper(ppo_model, offset=direction)
        divs, resets, steps = collect_windows(
            env, runner, trial, router, z_ref, trial,
            N_WINDOWS, reset_seed_base=300 + int(scale * 100), settle_steps=SETTLE_STEPS,
        )
        name = f"ppo_offset{scale}"
        if divs:
            mean_d = float(np.mean(divs))
            results[name] = (base_mean - mean_d, mean_d, steps, resets)
            print(f"  {name}: mean={mean_d:.5f} delta={base_mean-mean_d:+.5f} "
                  f"({steps} steps, {resets} falls)")
        else:
            results[name] = (float("-inf"), None, steps, resets)
            print(f"  {name}: NEVER produced a window in {steps} steps ({resets} falls)")

    print(f"\nStep 3b: multiplicative gain on PPO's own action (0.5x / 0.7x / 1.3x / 1.5x) "
          f"-- 'be more/less aggressive', the PPO-equivalent of amplitude scaling.")
    for gain in [0.5, 0.7, 1.3, 1.5]:
        trial = PPOPolicyWrapper(ppo_model, gain=gain)
        divs, resets, steps = collect_windows(
            env, runner, trial, router, z_ref, trial,
            N_WINDOWS, reset_seed_base=400 + int(gain * 100), settle_steps=SETTLE_STEPS,
        )
        name = f"ppo_gain{gain}"
        if divs:
            mean_d = float(np.mean(divs))
            results[name] = (base_mean - mean_d, mean_d, steps, resets)
            print(f"  {name}: mean={mean_d:.5f} delta={base_mean-mean_d:+.5f} "
                  f"({steps} steps, {resets} falls)")
        else:
            results[name] = (float("-inf"), None, steps, resets)
            print(f"  {name}: NEVER produced a window in {steps} steps ({resets} falls)")

    print(f"\n{'=' * 78}")
    print("RESULT")
    print("=" * 78)
    print(f"  PPO baseline mean={base_mean:.5f} std={base_std:.5f} "
          f"(a real reduction must clear roughly +/-{2*base_std:.5f})")
    for name, (reduction, mean_d, steps, resets) in sorted(
        results.items(), key=lambda x: -x[1][0]
    ):
        tag = "destabilising" if mean_d is None else f"mean={mean_d:.5f}"
        print(f"  {name}: reduction={reduction:+.5f}  {tag}")

    if results:
        best_name = max(results, key=lambda n: results[n][0])
        best_reduction = results[best_name][0]
    else:
        best_name, best_reduction = None, float("-inf")

    print()
    if base_std > 0.01:
        print(f"FINDING: the PPO baseline's own std ({base_std:.5f}) is already far "
              f"wider than CPG's saturated std (0.00004) -- a reactive policy produces "
              f"a genuinely more variable, less saturated divergence signal on ice, "
              f"even before any correction. That confirms the hypothesis about WHY "
              f"nothing moved CPG's signal, independent of whether any correction wins.")
    else:
        print(f"FINDING: the PPO baseline's own std ({base_std:.5f}) is still tight, "
              f"similar in character to CPG's saturation -- the hypothesis that a "
              f"reactive policy alone would unstick the signal is not supported by "
              f"this reading.")

    if best_reduction > 2 * base_std:
        print(f"\nPASS -- '{best_name}' reduces divergence by {best_reduction:+.5f}, "
              f"clearing the PPO baseline's own noise band.")
    else:
        print(f"\nHONEST NEGATIVE on corrections -- best reduction ({best_name}, "
              f"{best_reduction:+.5f}) does not clear the PPO baseline's noise band "
              f"({2*base_std:.5f}).")

    env.close()


if __name__ == "__main__":
    main()
