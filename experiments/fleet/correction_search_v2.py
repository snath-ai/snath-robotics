"""
Correction Search v2 -- can ANYTHING move a saturated divergence signal?
=============================================================================
v1 (correction_search.py) found two real bugs (wrong seq_len assumption,
RNG-pollution between warmup and ice phases) and, after fixing them, found
something more fundamental: divergence on real ice under the CPG gait is
SATURATED. Over 103 real consecutive windows with zero correction, D ranges
from 0.7055 to 0.7064 -- a spread of ~0.001. Every v1 candidate (constant
action offsets, scale 0.1-0.4, versus the CPG's own amplitude of 0.55) was
trying to move a signal inside its own noise floor. That is not a dead end,
it is a real finding: small constant-offset nudges are structurally the
wrong class of intervention to test against this metric.

This file tests two things properly, each averaged over enough windows to
clear the ~0.001 noise floor established above (20 windows/candidate, not
3), not just one short trial:

  1. Large constant offsets (scale 0.6, 0.8, 1.0 -- up to the actuator
     limit), in case magnitude alone was the missing ingredient.
  2. A structured correction: directly scaling the CPG's own amplitude
     parameter (policy.amp *= k), a genuinely different intervention class
     from adding noise on top of the existing gait -- actually changing the
     gait, not perturbing it.

Honest framing, unchanged: this may still fail for both classes. If even a
full-range offset and a direct gait-amplitude change can't move a saturated
metric, that is itself the real, structural answer -- not a search problem,
a sign the TV-distance-against-a-single-anchor metric itself may be too
coarse an instrument for this, which would be worth knowing before
investing further in finding "the right delta."

Run:
    .venv/bin/python experiments/fleet/correction_search_v2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

import gymnasium
from divergence_router import DivergenceRouter
from experiments.fleet.fleet_n2_transfer import (
    GRURunner, load_base_encoder, set_friction,
    FRICTION_NORMAL, FRICTION_ICE, DEFAULT_MODEL,
    EWMA_ALPHA, WARMUP_STEPS,
)
from experiments.fleet.policies import CPGPolicy

ACT_DIM = 6
N_WINDOWS = 20          # per candidate -- enough to clear the ~0.001 noise floor
SETTLE_STEPS = 30        # let the correction's dynamical effect establish before measuring
MAX_STEPS_PER_TRIAL = 3000


def collect_windows(env, runner, policy, router, z_ref, action_fn,
                     n_windows, reset_seed_base, settle_steps):
    """Run real ice physics, discard `settle_steps` real steps after each
    reset (let the correction's effect actually take hold before measuring,
    not just capture the pre-correction transient), then collect n_windows
    real divergence readings. Resets transparently through falls."""
    divs = []
    resets = 0
    steps = 0
    obs, _ = env.reset(seed=reset_seed_base)
    set_friction(env, FRICTION_ICE)
    runner.reset()
    policy.reset()
    settled = 0
    while len(divs) < n_windows and steps < MAX_STEPS_PER_TRIAL:
        action = action_fn(obs)
        obs, _, term, trunc, _ = env.step(action)
        steps += 1
        settled += 1
        z = runner.push(obs)
        if z is not None and settled > settle_steps:
            d = router.route(z_ref, z.detach()).divergence
            divs.append(d)
        if term or trunc:
            resets += 1
            obs, _ = env.reset(seed=reset_seed_base + resets)
            set_friction(env, FRICTION_ICE)
            runner.reset()
            policy.reset()
            settled = 0
    return divs, resets, steps


def main():
    print("=" * 78)
    print("CORRECTION SEARCH v2 -- large offsets + structured amplitude correction")
    print("=" * 78)

    encoder = load_base_encoder(DEFAULT_MODEL)
    runner = GRURunner(encoder)
    env = gymnasium.make("Walker2d-v5")
    router = DivergenceRouter(tau_high=0.60, tau_low=0.25, delta=0.35)

    print(f"\nStep 1: warmup on normal friction ({WARMUP_STEPS} steps) -> EWMA anchor.")
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

    print(f"\nStep 2: real baseline, {N_WINDOWS} windows, {SETTLE_STEPS}-step "
          f"settle, no correction.")
    base_policy = CPGPolicy(np.random.default_rng(1042))
    base_divs, base_resets, base_steps = collect_windows(
        env, runner, base_policy, router, z_ref, base_policy,
        N_WINDOWS, reset_seed_base=100, settle_steps=SETTLE_STEPS,
    )
    base_mean = float(np.mean(base_divs))
    base_std = float(np.std(base_divs))
    print(f"  baseline: mean={base_mean:.5f} std={base_std:.5f} "
          f"({base_steps} steps, {base_resets} falls)")

    results = {}

    print(f"\nStep 3a: LARGE constant offsets (scale 0.6 / 0.8 / 1.0).")
    for scale in [0.6, 0.8, 1.0]:
        d_rng = np.random.default_rng(7)
        direction = d_rng.normal(0, 1, size=ACT_DIM)
        direction = (direction / (np.linalg.norm(direction) + 1e-8) * scale).astype(np.float32)
        trial_policy = CPGPolicy(np.random.default_rng(2042))

        def action_fn(o, _d=direction):
            return np.clip(trial_policy(o) + _d, -1.0, 1.0).astype(np.float32)

        divs, resets, steps = collect_windows(
            env, runner, trial_policy, router, z_ref, action_fn,
            N_WINDOWS, reset_seed_base=300 + int(scale * 100), settle_steps=SETTLE_STEPS,
        )
        name = f"offset_scale{scale}"
        if divs:
            mean_d = float(np.mean(divs))
            results[name] = (base_mean - mean_d, mean_d, len(divs), steps, resets)
            print(f"  {name}: mean={mean_d:.5f} (baseline {base_mean:.5f}) "
                  f"delta={base_mean - mean_d:+.5f}  ({steps} steps, {resets} falls, "
                  f"{len(divs)}/{N_WINDOWS} windows)")
        else:
            results[name] = (float("-inf"), None, 0, steps, resets)
            print(f"  {name}: NEVER produced a window in {steps} steps ({resets} falls) "
                  f"-- destabilising.")

    print(f"\nStep 3b: STRUCTURED correction -- directly scale the CPG's own "
          f"amplitude (policy.amp *= k), not a constant offset added on top.")
    for k in [0.5, 0.7, 1.3, 1.6]:
        trial_policy = CPGPolicy(np.random.default_rng(2042))
        orig_reset = trial_policy.reset
        def scaled_reset(_policy=trial_policy, _k=k, _orig=orig_reset):
            _orig()
            _policy.amp *= _k

        def action_fn(o, _p=trial_policy):
            return _p(o)

        # monkey-patch this instance's reset so collect_windows' internal
        # policy.reset() calls (on every fall) keep reapplying the scaled
        # amplitude, not just the first call.
        trial_policy.reset = scaled_reset
        trial_policy.reset()  # apply once before the first rollout too

        divs, resets, steps = collect_windows(
            env, runner, trial_policy, router, z_ref, action_fn,
            N_WINDOWS, reset_seed_base=400 + int(k * 100), settle_steps=SETTLE_STEPS,
        )
        name = f"amp_scale{k}"
        if divs:
            mean_d = float(np.mean(divs))
            results[name] = (base_mean - mean_d, mean_d, len(divs), steps, resets)
            print(f"  {name}: mean={mean_d:.5f} (baseline {base_mean:.5f}) "
                  f"delta={base_mean - mean_d:+.5f}  ({steps} steps, {resets} falls, "
                  f"{len(divs)}/{N_WINDOWS} windows)")
        else:
            results[name] = (float("-inf"), None, 0, steps, resets)
            print(f"  {name}: NEVER produced a window in {steps} steps ({resets} falls) "
                  f"-- destabilising.")

    print(f"\n{'=' * 78}")
    print("RESULT")
    print("=" * 78)
    print(f"  baseline mean={base_mean:.5f} std={base_std:.5f} "
          f"(noise floor -- a real reduction must clear roughly +/-{2*base_std:.5f} "
          f"to be distinguishable from baseline variance)")
    for name, (reduction, mean_d, n, steps, resets) in sorted(
        results.items(), key=lambda x: -x[1][0]
    ):
        tag = "destabilising" if mean_d is None else f"mean={mean_d:.5f}"
        print(f"  {name}: reduction={reduction:+.5f}  {tag}")

    best_name = max(results, key=lambda n: results[n][0])
    best_reduction = results[best_name][0]
    if best_reduction > 2 * base_std:
        print(f"\nPASS -- '{best_name}' reduces divergence by {best_reduction:+.5f}, "
              f"clearing the baseline's own {2*base_std:.5f} noise band. A real "
              f"corrective effect, not noise.")
    else:
        print(f"\nHONEST NEGATIVE -- best reduction ({best_name}, {best_reduction:+.5f}) "
              f"does not clear the baseline's own noise band ({2*base_std:.5f}). "
              f"Neither large constant offsets nor direct amplitude scaling moved "
              f"this saturated divergence metric in a way distinguishable from "
              f"ordinary run-to-run variance.")

    env.close()


if __name__ == "__main__":
    main()
