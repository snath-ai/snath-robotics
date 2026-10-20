"""
Fixed Scale Check -- is the correction's effect on divergence monotonic in
scale, or does it genuinely get worse before it gets better?
=============================================================================
persist_magnitude_search.py's real PersistenceLoop run (after fixing the
frozen-action, abrupt-onset, and env-history-dependence bugs) escalated on
0/8 seeds: delta_mag~0.173 (1x) caused an immediate fall on most seeds,
delta_mag~0.346 (2x) caused a fall on the rest, and the loop's own policy
(escalate immediately on any attempt exceeding escalation_threshold=1.2)
never let it climb to delta_mag~0.692 (4x) -- the scale earlier manually
verified tonight to take 16 falls to 0 (t=+4.10).

Two different explanations fit that outcome, and they have opposite
implications: (a) the correction's effect is genuinely NON-monotonic --
small magnitude destabilizes a gait it only partially compensates, large
magnitude fully compensates and stabilizes -- in which case no incremental
ladder that bails on the first bad attempt could ever reach 4x safely, or
(b) PersistenceLoop's escalation policy is simply too trigger-happy for a
noisy, real-physics divergence signal and the correction is fine (or even
monotonically helpful) at every scale tested manually -- in which case the
loop's own configuration, not the correction, is what's broken.

This script decides between them directly: run a FIXED scale from step 1
(no ladder, no escalation) at 0.5x/1x/2x/4x/8x, 8 seeds each, same
env-per-seed isolation as the just-fixed Bug 3, and report fall counts and
final divergence per scale.

Run:
    .venv/bin/python experiments/fleet/fixed_scale_check.py
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
    FRICTION_NORMAL, FRICTION_ICE, DEFAULT_MODEL, EWMA_ALPHA, WARMUP_STEPS,
)
from experiments.fleet.policies import CPGPolicy
from experiments.fleet.extraction import extract_correction

PPO_CHECKPOINT = "/Users/aadithya/Desktop/Lar_Main/lar_jepa/models/walker2d_ppo.zip"
SETTLE_STEPS = 30
N_WINDOWS = 20  # real GRU windows observed per trial, after settling
SCALES = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
SEEDS = [1000 + s * 13 for s in range(1, 9)]  # same seeds persist_magnitude_search.py used


def run_fixed_scale(runner, router, z_ref, encoder_seq_len, direction, scale, seed):
    """One fresh-env trial, correction = scale*direction active from step 1,
    never changing -- no ladder, no escalation, just a direct measurement."""
    env = gymnasium.make("Walker2d-v5")
    policy = CPGPolicy(np.random.default_rng(seed))
    obs, _ = env.reset(seed=seed)
    set_friction(env, FRICTION_ICE)
    runner.reset()
    policy.reset()
    delta = direction * scale

    divs, fell, fell_at = [], False, None
    step = 0
    settled = 0
    while len(divs) < N_WINDOWS and step < 2000:
        action = np.clip(policy(obs) + delta, -1.0, 1.0).astype(np.float32)
        obs, _, term, trunc, _ = env.step(action)
        step += 1
        settled += 1
        if term or trunc:
            fell = True
            fell_at = step
            break
        z = runner.push(obs)
        if z is not None and settled > SETTLE_STEPS:
            divs.append(router.route(z_ref, z.detach()).divergence)
    env.close()
    return divs, fell, fell_at


def main():
    print("=" * 78)
    print("FIXED SCALE CHECK -- monotonic in scale, or worse-before-better?")
    print("=" * 78)

    env = gymnasium.make("Walker2d-v5")
    model = PPO.load(PPO_CHECKPOINT)

    def ppo_action(o):
        a, _ = model.predict(o, deterministic=True)
        return np.asarray(a, dtype=np.float32)

    def ppo_reset():
        pass

    print("\nStep 1: extract_correction() -- same verified function/direction "
          "persist_magnitude_search.py used.")
    extracted = extract_correction(
        ppo_action, ppo_reset, env,
        lambda e: set_friction(e, FRICTION_ICE),
        lambda e: set_friction(e, FRICTION_NORMAL),
        n_seeds=8, n_steps=200, seed_base=1,
    )
    direction = np.where(extracted.trustworthy_dims(), extracted.delta, 0.0).astype(np.float32)
    print(f"Direction: {direction.tolist()}")

    encoder = load_base_encoder(DEFAULT_MODEL)
    runner = GRURunner(encoder)
    router = DivergenceRouter(tau_high=0.60, tau_low=0.25, delta=0.35)

    print(f"\nWarmup on normal friction ({WARMUP_STEPS} steps) -> EWMA anchor.")
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
    env.close()

    print(f"\nStep 2: fixed-scale trials, no ladder, correction active from "
          f"step 1, {len(SEEDS)} seeds per scale, fresh env per trial.")
    for scale in SCALES:
        falls = 0
        fall_steps = []
        final_divs = []
        for seed in SEEDS:
            divs, fell, fell_at = run_fixed_scale(
                runner, router, z_ref, encoder.seq_len, direction, scale, seed
            )
            if fell:
                falls += 1
                fall_steps.append(fell_at)
            else:
                final_divs.append(np.mean(divs) if divs else np.nan)
        mean_div = np.nanmean(final_divs) if final_divs else float("nan")
        print(f"  scale={scale:4.1f}: falls={falls}/{len(SEEDS)}  "
              f"fall_steps={fall_steps}  "
              f"mean_divergence(non-fallers)={mean_div:.4f}")

    print(f"\n{'=' * 78}")
    print("If falls DROP as scale increases (worst at 0.5x/1x, best at 4x/8x): "
          "the correction is genuinely non-monotonic -- PersistenceLoop's "
          "escalate-on-first-bad-attempt ladder could never have reached the "
          "scale that works, this is a loop-configuration problem, not a "
          "dead end for the correction itself.")
    print("If falls are high across ALL scales including 4x/8x: the earlier "
          "manual 16->0 result does not replicate under env-isolated, "
          "settle-from-step-1 conditions, and that result needs re-checking.")


if __name__ == "__main__":
    main()
