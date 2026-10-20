"""
PERSIST Magnitude Search -- fixing the two real bugs in the earlier scale
sweep, and using PERSIST's actual increment mechanism instead of a blunt
1x/2x/4x/8x guess.
=============================================================================
extraction.py's search_and_verify() had two real problems, found by reading
its own output rather than trusting the printed "PASS": (1) a blunt power-
of-2 scale sweep instead of PERSIST's real increment search, and (2) a
pass/fail check that only verified statistical significance of a
difference, never that the difference was actually a REDUCTION -- a
confidently-measured result that made things slightly worse could have
been reported as a win.

This file fixes both. The direction comes from extract_correction()
(already verified correct after the abs-value fix). The magnitude comes
from PERSIST's real PersistenceLoop.run() -- imported directly, not
reimplemented -- seeded with that one real direction via a single-adaptor
library (so library.retrieve_top_k() trivially matches it and the real
increment loop runs: start small, grow the delta each attempt that doesn't
normalise divergence, stop on success, on escalation, or on hitting
delta_max).

One real adaptation PERSIST's own increment loop needed: its toy smoke
test calls step_fn() once per raw environment step and expects a fresh
divergence reading every time. FLEET's real divergence signal only
refreshes once per 30-step GRU window. Calling step_fn() on that cadence
here means each "attempt" the increment loop sees corresponds to one full
real window (30 real environment steps at a fixed delta), not one raw
step -- the real adaptation, not a shortcut.

Run:
    .venv/bin/python experiments/fleet/persist_magnitude_search.py
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
from experiments.fleet.extraction import extract_correction, _collect_divs
from experiments.persist.persistence_loop import Adaptor, AdaptorLibrary, PersistenceLoop

PPO_CHECKPOINT = "/Users/aadithya/Desktop/Lar_Main/lar_jepa/models/walker2d_ppo.zip"
ACT_DIM = 6
WINDOW = 30  # must match the real checkpoint's seq_len -- read, not assumed, below

NORMALISATION_THRESHOLD = 0.50  # a real, meaningfully-below-baseline(0.706) success bar
ESCALATION_THRESHOLD = 1.20     # well above baseline -- a correction making things this bad is not a win
FALL_DIVERGENCE_PENALTY = 1.50  # a fall must read as unambiguously bad, never neutral
PATIENCE = 10
DELTA_INIT = 1.0
DELTA_INCREMENT = 1.0
DELTA_MAX = 10.0
SETTLE_STEPS = 30  # matches extraction.py's _collect_divs settle convention


def run_one_increment_search(runner, router, z_ref, encoder_seq_len,
                              extracted_delta, seed):
    """One real PersistenceLoop.run() episode, seeded with the one real
    extracted direction via a single-adaptor library, on real ice physics."""
    library = AdaptorLibrary(similarity_threshold=0.0)  # only one adaptor ever stored -- always matches
    sig = np.ones(1)
    library.store(Adaptor(name="ice_response", delta=extracted_delta, divergence_signature=sig))

    loop = PersistenceLoop(
        library=library, action_dim=ACT_DIM,
        delta_init=DELTA_INIT, delta_increment=DELTA_INCREMENT, delta_max=DELTA_MAX,
        normalisation_threshold=NORMALISATION_THRESHOLD,
        escalation_threshold=ESCALATION_THRESHOLD,
        patience=PATIENCE,
        composition_residual_threshold=999.0,  # single-dimension problem here; disable composition
        tournament_k=1, tournament_band=0.0, tournament_trial_steps=1,
    )

    # Baseline reading (uncorrected) -- for the escalation pre-check and for
    # reporting only. Run on a SEPARATE, never-before-stepped env instance,
    # not the one the actual trial uses below. Found empirically, not
    # assumed: env.reset(seed=X) on this environment is NOT history-
    # independent -- the same seed produces a different trajectory if the
    # env object was previously stepped at all, even with a clean reset in
    # between. The same category of leak apply_wind's own comment already
    # warns about for xfrc_applied. A fresh env instance sidesteps it by
    # construction rather than hunting the exact MuJoCo field responsible.
    baseline_env = gymnasium.make("Walker2d-v5")
    baseline_runner = GRURunner(runner.encoder)
    baseline_policy = CPGPolicy(np.random.default_rng(seed))
    obs0, _ = baseline_env.reset(seed=seed)
    set_friction(baseline_env, FRICTION_ICE)
    baseline_policy.reset()
    z = None
    for _ in range(encoder_seq_len):
        action = baseline_policy(obs0)
        obs0, _, term, trunc, _ = baseline_env.step(action)
        if term or trunc:
            obs0, _ = baseline_env.reset(seed=seed + 1)
            set_friction(baseline_env, FRICTION_ICE)
            baseline_runner.reset()
            baseline_policy.reset()
        z = baseline_runner.push(obs0)
    entry_divergence = router.route(z_ref, z.detach()).divergence if z is not None else 0.706
    baseline_env.close()

    # The actual trial: its OWN fresh, never-before-stepped env instance --
    # the same history-independence issue the baseline fix above addresses
    # applies here too, across seeds, not just baseline-vs-trial within one
    # seed. Also a fresh reset so the correction is active from step 1 of
    # the gait rather than switching on abruptly mid-stride.
    trial_env = gymnasium.make("Walker2d-v5")
    policy = CPGPolicy(np.random.default_rng(seed))
    obs, _ = trial_env.reset(seed=seed)
    set_friction(trial_env, FRICTION_ICE)
    runner.reset()
    policy.reset()

    # PersistenceLoop calls base_action_fn(obs, delta) exactly once per
    # "attempt" and hands the single resulting action to step_fn. Here one
    # attempt must span a real 30-step window (FLEET's real divergence
    # cadence), not one raw step -- so step_fn, not PersistenceLoop, has to
    # recompute the action fresh every real step using the CURRENT obs and
    # the CPG's evolving gait phase. The delta itself doesn't change within
    # an attempt, only the policy's own output does -- captured via this
    # shared container since step_fn's signature only receives the single
    # precomputed action, not the delta on its own.
    state = {"obs": obs, "current_delta": None, "first_attempt": True}

    def base_action_fn(_obs, delta):
        state["current_delta"] = delta
        return np.clip(policy(_obs) + delta, -1.0, 1.0).astype(np.float32)

    def step_fn(action):
        """One real ATTEMPT = one real WINDOW: encoder_seq_len real steps,
        the CPG gait continuing to evolve step by step (not frozen), the
        same delta re-applied against each fresh obs. The FIRST attempt
        runs settle_steps extra real steps first, correction already
        active throughout -- matching _collect_divs' own settle convention
        -- since the window was just reset and needs to fill AND settle
        before a reading means anything; later attempts are already
        settled into the corrected gait, so they only need one fresh
        window's worth of steps."""
        delta = state["current_delta"]
        n_steps = (SETTLE_STEPS + encoder_seq_len) if state["first_attempt"] else encoder_seq_len
        state["first_attempt"] = False
        last_d = None
        for step_i in range(n_steps):
            a = action if step_i == 0 else np.clip(policy(state["obs"]) + delta, -1.0, 1.0).astype(np.float32)
            o, _, term, trunc, _ = trial_env.step(a)
            state["obs"] = o
            if term or trunc:
                o2, _ = trial_env.reset(seed=seed + 100)
                set_friction(trial_env, FRICTION_ICE)
                runner.reset()
                policy.reset()
                state["obs"] = o2
                return state["obs"], FALL_DIVERGENCE_PENALTY
            z = runner.push(o)
            if z is not None:
                last_d = router.route(z_ref, z.detach()).divergence
        return state["obs"], (last_d if last_d is not None else entry_divergence)

    result = loop.run(
        obs=obs, divergence_signature=sig, current_divergence=entry_divergence,
        base_action_fn=base_action_fn, step_fn=step_fn,
        zone="ice", phase_label=f"seed_{seed}",
    )
    trial_env.close()
    return result, entry_divergence


def main():
    print("=" * 78)
    print("PERSIST MAGNITUDE SEARCH -- real increment loop, real direction, fixed checks")
    print("=" * 78)

    env = gymnasium.make("Walker2d-v5")
    model = PPO.load(PPO_CHECKPOINT)

    def ppo_action(o):
        a, _ = model.predict(o, deterministic=True)
        return np.asarray(a, dtype=np.float32)

    def ppo_reset():
        pass

    print("\nStep 1: extract_correction() -- same verified function, PPO vs CPG, "
          "ice vs normal, 8 paired seeds.")
    extracted = extract_correction(
        ppo_action, ppo_reset, env,
        lambda e: set_friction(e, FRICTION_ICE),
        lambda e: set_friction(e, FRICTION_NORMAL),
        n_seeds=8, n_steps=200, seed_base=1,
    )
    print(extracted.summary())
    # zero out dimensions the extraction itself flagged as not trustworthy --
    # do not hand PersistenceLoop a direction component we already know is noise
    direction = np.where(extracted.trustworthy_dims(), extracted.delta, 0.0).astype(np.float32)
    print(f"Direction handed to PERSIST (untrustworthy dims zeroed): {direction.tolist()}")

    encoder = load_base_encoder(DEFAULT_MODEL)
    print(f"Real encoder seq_len = {encoder.seq_len} (checked, used as the real window length below)")
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

    print(f"\nStep 2: run PERSIST's real increment loop, 8 independent seeds, "
          f"normalisation_threshold={NORMALISATION_THRESHOLD}, "
          f"escalation_threshold={ESCALATION_THRESHOLD}, patience={PATIENCE}.")
    SEEDS = list(range(1, 9))
    final_deltas = []
    successes = 0
    for s in SEEDS:
        result, entry_d = run_one_increment_search(
            runner, router, z_ref, encoder.seq_len, direction, seed=1000 + s * 13
        )
        status = "SUCCESS" if result.success else ("ESCALATED" if result.escalated else "PATIENCE EXHAUSTED")
        print(f"  seed {s}: entry_D={entry_d:.4f}  {status}  "
              f"final_D={result.final_divergence:.4f}  steps={result.steps_to_resolution}  "
              f"curve={[round(d,3) for d in result.divergence_curve]}")
        if result.success:
            successes += 1
            # reconstruct the converged delta magnitude from steps taken
            n_increments = result.steps_to_resolution - 1
            converged_scale = DELTA_INIT + n_increments * DELTA_INCREMENT
            final_deltas.append(converged_scale)

    print(f"\n{'=' * 78}\nRESULT\n{'=' * 78}")
    print(f"PERSIST increment search succeeded on {successes}/{len(SEEDS)} seeds.")
    if final_deltas:
        print(f"Converged scales on successful seeds: {final_deltas} "
              f"(mean={np.mean(final_deltas):.2f})")

    print(f"\nStep 3: final honest verification -- the converged scale "
          f"({np.mean(final_deltas) if final_deltas else DELTA_INIT:.1f}x), "
          f"properly powered, checking the reduction is actually positive "
          f"this time, not just statistically different.")
    if final_deltas:
        final_scale = float(np.median(final_deltas))
        base_means, corr_means = [], []
        for i in range(8):
            seed = 5000 + i * 7
            # Same env-isolation fix as Step 2 (Bug 3): _collect_divs's two
            # calls per seed below both reset at the SAME seed, and the
            # second call would otherwise run on an env already stepped by
            # the first -- exactly the history-dependence that produced a
            # spurious fall in Step 2. A fresh env per call sidesteps it
            # here too, rather than trusting the shared `env` again.
            base_env = gymnasium.make("Walker2d-v5")
            base_policy = CPGPolicy(np.random.default_rng(seed))
            b_divs, b_resets, _ = _collect_divs(
                base_env, runner, router, z_ref, base_policy, base_policy,
                lambda e: set_friction(e, FRICTION_ICE), 20, seed, 30,
            )
            base_env.close()

            corr_env = gymnasium.make("Walker2d-v5")
            corr_policy = CPGPolicy(np.random.default_rng(seed))
            scaled = direction * final_scale
            def action_fn(o, _p=corr_policy, _d=scaled):
                return np.clip(_p(o) + _d, -1.0, 1.0).astype(np.float32)
            c_divs, c_resets, _ = _collect_divs(
                corr_env, runner, router, z_ref, corr_policy, action_fn,
                lambda e: set_friction(e, FRICTION_ICE), 20, seed, 30,
            )
            corr_env.close()

            base_means.append(np.mean(b_divs) if b_divs else np.nan)
            corr_means.append(np.mean(c_divs) if c_divs else np.nan)
            print(f"  seed {i}: baseline={base_means[-1]:.4f} (falls={b_resets})  "
                  f"corrected@{final_scale:.1f}x={corr_means[-1]:.4f} (falls={c_resets})")

        base_means, corr_means = np.array(base_means), np.array(corr_means)
        diffs = base_means - corr_means  # positive = genuine reduction
        se = np.nanstd(diffs, ddof=1) / np.sqrt(len(diffs))
        t = np.nanmean(diffs) / se if se > 1e-12 else 0.0
        same_sign = bool(np.all(diffs[~np.isnan(diffs)] > 0))
        is_real_reduction = bool(np.nanmean(diffs) > 0)  # the check that was missing before

        print(f"\nmean reduction={np.nanmean(diffs):+.4f}  t={t:+.2f}  "
              f"same_sign={same_sign}  ACTUALLY_A_REDUCTION={is_real_reduction}")
        if is_real_reduction and same_sign and abs(t) > 2.0:
            print(f"\nPASS -- PERSIST's real increment search found scale={final_scale:.1f}x, "
                  f"and it is verified as a genuine, statistically real REDUCTION "
                  f"(not just a significant difference), checked honestly this time.")
        else:
            print(f"\nNOT CONFIRMED as a clean win -- see numbers above.")
    else:
        print("No seed reached success -- nothing to verify.")

    env.close()


if __name__ == "__main__":
    main()
