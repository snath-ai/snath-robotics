"""
Correction Search -- does a real divergence-reducing action-delta exist for
Walker2d on real (corrected) ice, under the real CPG policy?
=============================================================================
Step 1 of building verification-gated recognition for FLEET (session
discussion, 2026-10-20). FLEET currently has no corrective-action mechanism
anywhere -- neither robot in the fleet pipeline ever tries to walk
differently on ice, only to recognise it. PERSIST's AdaptorTournament looks
like the natural tool to add that capability, but checking its own code
first surfaces a real gap: PERSIST's smoke test hand-specifies the delta
DIRECTION as a comment ("reduce forward torque") and trials it against a
SYNTHETIC scripted divergence-decay function, not real physics. The
tournament has only ever been proven to refine a MAGNITUDE given an
already-correct direction -- never to discover a working direction from
nothing. That is the actual open question here, tested for the first time
against real MuJoCo physics.

Two real bugs found and fixed getting this running, documented here rather
than silently patched:

v1 hardcoded TRIAL_STEPS=10, assuming GRUProprioEncoder's class default
seq_len. The actual checkpoint (gru_cls_multiseed_cpg.pt) was trained with
seq_len=30 -- the window never filled in 10 steps, every trial silently hit
the "window not full" fallback, and the reported "PASS" was built entirely
on a hardcoded default divergence value, not a real reading. Fixed by
reading encoder.seq_len directly from the loaded checkpoint.

v1 also reused the SAME CPGPolicy (and its RNG) across a 120-step normal
warmup and then the ice phase. policy.reset() re-rolls freq/amp/phase0 from
whatever RNG state the warmup left behind, not from a clean, independent
draw -- so the ice-phase gait parameters were an uncontrolled side effect of
how much randomness the warmup had already consumed, not a fair, reproducible
sample. Fixed by giving the ice phase its own fresh policy and RNG.

Also: a single continuous episode is not guaranteed to survive the 30 steps
a window needs, and does not need to -- FLEET's own real Robot.phase() never
assumes this either, it accumulates routed windows across as many resets as
it takes. This file does the same: run_until_n_windows() keeps stepping
(and transparently resetting through falls) until the requested number of
REAL windows has been observed, exactly matching that convention rather
than assuming one lucky surviving episode.

Honest framing, unchanged from v1: this may fail. If no candidate reduces
divergence, that is the real answer, not a bug to fix until it looks better.

Run:
    .venv/bin/python experiments/fleet/correction_search.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

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
from experiments.persist.persistence_loop import Adaptor, AdaptorTournament

ACT_DIM = 6
N_CANDIDATES = 8
N_WINDOWS_ENTRY = 5    # real divergence readings to establish the baseline
N_WINDOWS_TRIAL = 3    # real divergence readings per candidate trial


def run_until_n_windows(env, runner, policy, router, z_ref, friction,
                         n_windows, action_fn, reset_seed_base):
    """Step the real env (resetting through falls as needed, matching
    Robot.phase()'s real convention) until n_windows real GRU-window
    divergence readings have been produced. Returns (obs, divergences)."""
    divergences = []
    resets = 0
    steps = 0
    obs, _ = env.reset(seed=reset_seed_base)
    set_friction(env, friction)
    runner.reset()
    policy.reset()
    max_steps = 2000  # hard safety cap, not expected to bind
    while len(divergences) < n_windows and steps < max_steps:
        action = action_fn(obs)
        obs, _, term, trunc, _ = env.step(action)
        steps += 1
        z = runner.push(obs)
        if z is not None:
            d = router.route(z_ref, z.detach()).divergence
            divergences.append(d)
        if term or trunc:
            resets += 1
            obs, _ = env.reset(seed=reset_seed_base + resets)
            set_friction(env, friction)
            runner.reset()
            policy.reset()
    return obs, divergences, resets, steps


def main():
    print("=" * 78)
    print("CORRECTION SEARCH -- does a real action-delta reduce divergence on real ice?")
    print("=" * 78)

    encoder = load_base_encoder(DEFAULT_MODEL)
    print(f"\nReal encoder seq_len = {encoder.seq_len} (checked, not assumed -- "
          f"the class default is 10, this checkpoint is 30)")
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
    print(f"  z_ref ready (window filled {'yes' if z_ref is not None else 'NO -- '+str(WARMUP_STEPS)+' steps was not enough for seq_len='+str(encoder.seq_len)}).")
    assert z_ref is not None, "warmup too short for this encoder's seq_len"

    print(f"\nStep 2: real ice baseline -- a FRESH policy/RNG (not reused from "
          f"warmup), accumulating {N_WINDOWS_ENTRY} real windows across however "
          f"many resets it takes, same convention as Robot.phase().")
    ice_policy = CPGPolicy(np.random.default_rng(1042))
    obs, entry_divs, entry_resets, entry_steps = run_until_n_windows(
        env, runner, ice_policy, router, z_ref, FRICTION_ICE,
        N_WINDOWS_ENTRY, ice_policy, reset_seed_base=100,
    )
    print(f"  Real divergence readings on ice, no correction: "
          f"{[round(d, 3) for d in entry_divs]} ({entry_steps} steps, "
          f"{entry_resets} falls along the way)")
    entry_divergence = entry_divs[-1]
    print(f"  Entry divergence for the search: {entry_divergence:.4f} "
          f"(tau_low={router.tau_low}, tau_high={router.tau_high})")

    print(f"\nStep 3: generate {N_CANDIDATES} random candidate correction directions "
          f"(no hand-specified prior exists for Walker2d) and trial each for "
          f"{N_WINDOWS_TRIAL} real windows against real ice physics.")
    cand_rng = np.random.default_rng(7)
    candidates = []
    for i in range(N_CANDIDATES):
        scale = float(cand_rng.choice([0.1, 0.2, 0.3, 0.4]))
        direction = cand_rng.normal(0, 1, size=ACT_DIM)
        direction = direction / (np.linalg.norm(direction) + 1e-8) * scale
        candidates.append((direction.astype(np.float32), scale))

    trial_policy = CPGPolicy(np.random.default_rng(2042))
    rates = {}
    for i, (delta, scale) in enumerate(candidates):
        name = f"cand_{i}_scale{scale:.1f}"

        def action_fn(o, _delta=delta):
            return np.clip(trial_policy(o) + _delta, -1.0, 1.0).astype(np.float32)

        _, trial_divs, trial_resets, trial_steps = run_until_n_windows(
            env, runner, trial_policy, router, z_ref, FRICTION_ICE,
            N_WINDOWS_TRIAL, action_fn, reset_seed_base=200 + i * 10,
        )
        if not trial_divs:
            # Genuine finding, not an error: this delta destabilises the
            # walker badly enough that it falls before a single 30-step
            # window ever completes, across every reset attempt within the
            # step budget. That is itself informative -- worst possible
            # rate, excluded from winning, not silently dropped.
            print(f"  {name}: delta={delta.round(3).tolist()}  "
                  f"NEVER produced one real window in {trial_steps} steps "
                  f"({trial_resets} falls) -- excluded as destabilising.")
            rates[name] = float("-inf")
            continue
        rate = (entry_divergence - trial_divs[-1]) / max(trial_steps, 1)
        rates[name] = rate
        print(f"  {name}: delta={delta.round(3).tolist()}  "
              f"divs={[round(d,3) for d in trial_divs]}  "
              f"({trial_steps} steps, {trial_resets} falls)  rate={rate:+.5f}/step")

    print(f"\n{'=' * 78}")
    print("RESULT")
    print("=" * 78)
    best_name = max(rates, key=rates.__getitem__)
    best_rate = rates[best_name]
    for name, rate in sorted(rates.items(), key=lambda x: -x[1]):
        print(f"  {name}: rate={rate:+.5f}/step")

    if best_rate > 0:
        print(f"\nPASS -- a real divergence-reducing correction exists: "
              f"'{best_name}', rate={best_rate:+.5f}/step, measured against "
              f"real physics with a real, previously-silent bug fixed first.")
    else:
        print(f"\nHONEST NEGATIVE -- none of {N_CANDIDATES} random candidate "
              f"directions reduced real divergence on real ice. A random "
              f"constant action offset does not appear to be the right kind "
              f"of correction for this CPG gait's ice response.")

    env.close()


if __name__ == "__main__":
    main()
