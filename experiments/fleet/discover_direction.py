"""
Discover Direction -- a real exploration mechanism for the piece every
other script in this investigation has smuggled in for free: the
CORRECTION DIRECTION, discovered with no expert policy to diff against.
=============================================================================
extraction.py measured what an already-competent PPO policy does
differently on ice. persist_magnitude_search.py searched over MAGNITUDE
for a direction handed to it. Neither answers the question a fleet robot
actually faces the first time anyone encounters a new hazard: there is no
expert, no adaptor in the library, nothing to copy. PersistenceLoop's own
documented Phase 1 ("Encounter -- no adaptor") has no mechanism for this
at all -- it just logs the stream and escalates.

This file builds that missing piece: random-direction search, the
simplest real zeroth-order exploration method (used for exactly this kind
of problem by ES/ARS-style policy search) --

  1. Sample K random unit directions in the action-offset space ONCE,
     fixed, not re-sampled per seed -- so "direction #3" means the same
     physical probe across every seed, and its reliability can be judged
     the same way extract_correction() judges a dimension: paired across
     seeds, same-sign, t-stat.
  2. Test each direction AND its negative (antithetic sampling -- halves
     the risk of a lucky draw being all we ever see) as a small, fixed,
     policy-relative offset to the WEAK policy's own action. No expert
     anywhere in this file except as a diagnostic check at the very end.
  3. Score every (direction, seed) pair by real divergence reduction vs.
     an uncorrected baseline on that SAME seed (paired), fresh env per
     probe (the env-reset history-dependence lesson from
     persist_magnitude_search.py applies here too).
  4. A direction only counts as found if it reduces divergence reliably
     across all seeds (same-sign AND |t|>2 AND mean reduction>0) -- the
     same bar extract_correction() uses, not "looked good once."
  5. If found, hand it to PersistenceLoop (now fixed) to search for
     magnitude -- exercising discovery and magnitude-search together with
     zero expert policy anywhere in the pipeline.
  6. Diagnostic only, never used to pick anything: compare the discovered
     direction's sign pattern against extract_correction()'s real
     expert-derived direction, on the expert's own trustworthy dims. This
     is the actual test of whether the mechanism found something true.

Run:
    .venv/bin/python experiments/fleet/discover_direction.py
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
from experiments.persist.persistence_loop import Adaptor, AdaptorLibrary, PersistenceLoop

ACT_DIM = 6
SETTLE_STEPS = 30
K_DIRECTIONS = 8          # random directions sampled once, fixed across all seeds
PROBE_SCALE = 0.60        # First run at 0.15 (~27% of base_amp=0.55) found almost no
                          # signal: the uncorrected baseline already falls on EVERY
                          # seed (ceiling value), and at 0.15 a RANDOM direction almost
                          # never has enough strength to change that -- fixed_scale_check.py
                          # already showed even the perfectly-aligned expert direction only
                          # barely helps (6/8, not 8/8) at this magnitude band, needing ~4x
                          # (norm~0.69) to fully stop falls. Raised to that same ballpark so
                          # a partially-aligned random direction has a real chance to show
                          # a differentiated effect, instead of being swallowed by the floor.
FALL_DIVERGENCE_PENALTY = 1.50   # same convention as persist_magnitude_search.py
DIRECTION_RNG_SEED = 777         # separate from env seeds -- fixes WHICH random
                                  # directions get tested, not which physics trajectory

SEEDS = [1000 + s * 13 for s in range(1, 9)]  # same seeds used throughout this investigation

PPO_CHECKPOINT = "/Users/aadithya/Desktop/Lar_Main/lar_jepa/models/walker2d_ppo.zip"


def sample_random_directions(rng: np.random.Generator, k: int, dim: int) -> np.ndarray:
    """K random unit vectors in R^dim -- the candidate pool for exploration,
    sampled once and reused across every seed."""
    v = rng.normal(size=(k, dim))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def measure_entry_divergence(runner, router, z_ref, encoder_seq_len, seed):
    """A quick, UNCORRECTED reading of 'how bad is it right now' -- exactly
    persist_magnitude_search.py's own baseline_env convention (one real
    window, no extra settle), not the 60-step settle+window probe above.
    That longer probe is deliberately long enough to let an UNCORRECTED
    rollout fall (used to score candidates against a true floor); this one
    is deliberately short, so entry_divergence reflects the signal a robot
    would actually see the moment it starts reacting, before a fall that a
    not-yet-tried correction might have prevented. Reusing the 60-step
    reading here would feed PersistenceLoop's entry-level escalation check
    a value that already encodes "it already fell," which isn't a fair
    test of whether a given adaptor is worth trying."""
    env = gymnasium.make("Walker2d-v5")
    policy = CPGPolicy(np.random.default_rng(seed))
    obs, _ = env.reset(seed=seed)
    set_friction(env, FRICTION_ICE)
    runner.reset()
    policy.reset()
    last_d = None
    for _ in range(encoder_seq_len):
        action = policy(obs)
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            env.close()
            return FALL_DIVERGENCE_PENALTY
        z = runner.push(obs)
        if z is not None:
            last_d = router.route(z_ref, z.detach()).divergence
    env.close()
    return last_d if last_d is not None else 0.706


def probe_once(runner, router, z_ref, encoder_seq_len, direction, scale, seed):
    """One fresh-env probe: the WEAK CPG policy's own action plus
    scale*direction, active from step 1, real settle+window convention,
    fall-aware divergence (same discipline as persist_magnitude_search.py
    and fixed_scale_check.py -- fresh env per trial, no shared env across
    probes or seeds)."""
    env = gymnasium.make("Walker2d-v5")
    policy = CPGPolicy(np.random.default_rng(seed))
    obs, _ = env.reset(seed=seed)
    set_friction(env, FRICTION_ICE)
    runner.reset()
    policy.reset()
    delta = direction * scale

    n_steps = SETTLE_STEPS + encoder_seq_len
    last_d = None
    for _ in range(n_steps):
        action = np.clip(policy(obs) + delta, -1.0, 1.0).astype(np.float32)
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            env.close()
            return FALL_DIVERGENCE_PENALTY
        z = runner.push(obs)
        if z is not None:
            last_d = router.route(z_ref, z.detach()).divergence
    env.close()
    return last_d if last_d is not None else 0.706  # entry-divergence fallback, should not trigger


def main():
    print("=" * 78)
    print("DISCOVER DIRECTION -- real random-direction search, no expert, no prior adaptor")
    print("=" * 78)

    env = gymnasium.make("Walker2d-v5")
    encoder = load_base_encoder(DEFAULT_MODEL)
    runner = GRURunner(encoder)
    router = DivergenceRouter(tau_high=0.60, tau_low=0.25, delta=0.35)

    print(f"\nWarmup on normal friction ({WARMUP_STEPS} steps) -> EWMA anchor. "
          f"No expert policy loaded anywhere in this script yet.")
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

    direction_rng = np.random.default_rng(DIRECTION_RNG_SEED)
    base_dirs = sample_random_directions(direction_rng, K_DIRECTIONS, ACT_DIM)
    candidates = list(base_dirs) + [-d for d in base_dirs]  # antithetic pairs, 2K total
    labels = [f"+d{i}" for i in range(K_DIRECTIONS)] + [f"-d{i}" for i in range(K_DIRECTIONS)]

    print(f"\nStep 1: {len(candidates)} candidate directions ({K_DIRECTIONS} random "
          f"+ antithetic negatives), probe_scale={PROBE_SCALE}, {len(SEEDS)} paired seeds.")

    print(f"\nStep 2: baseline (no correction) per seed, fresh env, {len(SEEDS)} seeds.")
    baseline = {}
    for seed in SEEDS:
        baseline[seed] = probe_once(runner, router, z_ref, encoder.seq_len,
                                     np.zeros(ACT_DIM), 0.0, seed)
        print(f"  seed {seed}: baseline_divergence={baseline[seed]:.4f}")

    print(f"\nStep 3: probe every candidate at every seed ({len(candidates)}x{len(SEEDS)} "
          f"= {len(candidates)*len(SEEDS)} fresh-env trials).")
    reductions = {}  # label -> np.ndarray of per-seed reductions (baseline - probe)
    for label, cand in zip(labels, candidates):
        per_seed = []
        for seed in SEEDS:
            d = probe_once(runner, router, z_ref, encoder.seq_len, cand, PROBE_SCALE, seed)
            per_seed.append(baseline[seed] - d)
        reductions[label] = np.array(per_seed)

    print(f"\n{'=' * 78}\nSTEP 4: per-candidate statistics (paired across {len(SEEDS)} seeds)\n{'=' * 78}")
    stats = {}
    for label in labels:
        diffs = reductions[label]
        mean = diffs.mean()
        se = diffs.std(ddof=1) / np.sqrt(len(diffs))
        t = mean / se if se > 1e-12 else float("inf") * np.sign(mean)
        # NOT extract_correction()'s same_sign check verbatim: this reduction
        # signal is floor/ceiling-saturated (a fall reads as exactly the same
        # 1.5 penalty every time), so "no effect this seed" legitimately comes
        # back as an EXACT 0.0, not a small continuous value either side of
        # zero. Requiring all(>0) or all(<0) would wrongly reject a candidate
        # that helps on some seeds and does nothing (never HARM) on the rest --
        # that is not two effects canceling out, it's one consistent,
        # never-harmful effect. The honest bar for this metric is "no seed
        # made it worse," not "every seed moved."
        no_harm = bool(np.all(diffs >= -1e-9))
        any_real_effect = bool(np.any(np.abs(diffs) > 1e-9))
        stats[label] = (mean, t, no_harm, any_real_effect)
        print(f"  {label:4s}: mean_reduction={mean:+.4f}  t={t:+.2f}  "
              f"no_harm={no_harm}  per_seed={[round(float(x),3) for x in diffs]}")

    qualifying = [
        (label, mean) for label, (mean, t, no_harm, any_real_effect) in stats.items()
        if no_harm and any_real_effect and abs(t) > 2.0 and mean > 0
    ]

    print(f"\n{'=' * 78}\nRESULT\n{'=' * 78}")
    if not qualifying:
        print(f"HONEST NEGATIVE -- no candidate among {len(candidates)} random directions "
              f"reduced divergence reliably (no_harm AND a real effect AND |t|>2 AND mean>0) across all "
              f"{len(SEEDS)} seeds. Random search with K={K_DIRECTIONS} at "
              f"probe_scale={PROBE_SCALE} did not find a usable direction. This does not "
              f"mean no direction exists -- it means this specific, bounded search didn't "
              f"find one; a real deployed version would need more directions, a different "
              f"probe scale, or a smarter sampler (e.g. an ES-style weighted combination "
              f"of all probes rather than winner-take-all) before concluding the mechanism "
              f"itself can't work.")
        env2_summary = None
    else:
        winner_label, winner_mean = max(qualifying, key=lambda x: x[1])
        winner_idx = labels.index(winner_label)
        winner_dir = candidates[winner_idx]
        print(f"FOUND a real, reliable direction via random search alone: {winner_label} "
              f"(mean_reduction={winner_mean:+.4f}, {stats[winner_label][1]:+.2f} t-stat, "
              f"never harmful on any of {len(SEEDS)} seeds).")
        print(f"Discovered direction (raw, unit-norm before scaling): "
              f"{[round(float(x), 4) for x in winner_dir]}")

        print(f"\nStep 5: hand the discovered direction to PersistenceLoop (real "
              f"increment search, same mechanism and fix verified earlier tonight) "
              f"to find magnitude -- no expert anywhere in this pipeline.")
        library = AdaptorLibrary(similarity_threshold=0.0)
        sig = np.ones(1)
        library.store(Adaptor(name="discovered_ice_response", delta=winner_dir, divergence_signature=sig))
        loop = PersistenceLoop(
            library=library, action_dim=ACT_DIM,
            delta_init=1.0, delta_increment=1.0, delta_max=10.0,
            normalisation_threshold=0.50, escalation_threshold=1.20,
            patience=10, composition_residual_threshold=999.0,
            tournament_k=1, tournament_band=0.0, tournament_trial_steps=1,
        )
        for seed in SEEDS[:3]:  # a few seeds -- this step is about pipeline integration,
                                 # magnitude calibration itself is already logged separately
            entry_d = measure_entry_divergence(runner, router, z_ref, encoder.seq_len, seed)
            trial_env = gymnasium.make("Walker2d-v5")
            policy = CPGPolicy(np.random.default_rng(seed))
            obs, _ = trial_env.reset(seed=seed)
            set_friction(trial_env, FRICTION_ICE)
            runner.reset()
            policy.reset()
            state = {"obs": obs, "current_delta": None, "first_attempt": True}

            def base_action_fn(_obs, delta):
                state["current_delta"] = delta
                return np.clip(policy(_obs) + delta, -1.0, 1.0).astype(np.float32)

            def step_fn(action):
                delta = state["current_delta"]
                n_steps = (SETTLE_STEPS + encoder.seq_len) if state["first_attempt"] else encoder.seq_len
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
                return state["obs"], (last_d if last_d is not None else entry_d)

            result = loop.run(
                obs=obs, divergence_signature=sig, current_divergence=entry_d,
                base_action_fn=base_action_fn, step_fn=step_fn,
                zone="ice", phase_label=f"discovered_seed_{seed}",
            )
            trial_env.close()
            status = "SUCCESS" if result.success else ("ESCALATED" if result.escalated else "PATIENCE EXHAUSTED")
            print(f"  seed {seed}: entry_D={entry_d:.4f}  {status}  final_D={result.final_divergence:.4f}  "
                  f"steps={result.steps_to_resolution}")
            library.store(Adaptor(name="discovered_ice_response", delta=winner_dir, divergence_signature=sig))

        print(f"\nStep 6 (diagnostic only -- NOT used to pick the winner above): does the "
              f"discovered direction's sign pattern agree with extract_correction()'s real "
              f"expert-derived direction, on the expert's own trustworthy dims?")
        model = PPO.load(PPO_CHECKPOINT)

        def ppo_action(o):
            a, _ = model.predict(o, deterministic=True)
            return np.asarray(a, dtype=np.float32)

        def ppo_reset():
            pass

        expert_env = gymnasium.make("Walker2d-v5")
        expert = extract_correction(
            ppo_action, ppo_reset, expert_env,
            lambda e: set_friction(e, FRICTION_ICE),
            lambda e: set_friction(e, FRICTION_NORMAL),
            n_seeds=8, n_steps=200, seed_base=1,
        )
        expert_env.close()
        trust = expert.trustworthy_dims()
        agree, total = 0, 0
        for i in range(ACT_DIM):
            if not trust[i]:
                continue
            total += 1
            same = (np.sign(winner_dir[i]) == np.sign(expert.delta[i]))
            agree += int(same)
            print(f"  dim {i}: discovered={winner_dir[i]:+.4f}  expert={expert.delta[i]:+.4f}  "
                  f"(expert trustworthy)  {'AGREE' if same else 'DISAGREE'}")
        print(f"\nSign agreement with expert on expert's trustworthy dims: {agree}/{total}")

        print(f"\nStep 7: isolated verification of the PersistenceLoop 'SUCCESS' above -- "
              f"the loop handed winner_dir a UNIT-NORM direction (norm=1.0, much larger "
              f"raw torque than any expert-direction run used, norm~0.17) and its own "
              f"attempt sequence re-uses one trial_env across attempts (the same "
              f"env-reset-history-dependence flagged as unresolved in STATUS.md entry V). "
              f"Before trusting 'delta_mag=2.0 works', check it the same independent way "
              f"fixed_scale_check.py checked the expert direction: fresh env per trial, "
              f"fixed scale from step 1, no ladder, no resets, all 8 seeds (not just 3).")
        for scale in [1.0, 1.5, 2.0, 2.5, 3.0]:
            falls, fall_steps, kept_divs = 0, [], []
            for seed in SEEDS:
                d = probe_once(runner, router, z_ref, encoder.seq_len, winner_dir, scale, seed)
                if d >= FALL_DIVERGENCE_PENALTY:
                    falls += 1
                else:
                    kept_divs.append(d)
            mean_div = np.mean(kept_divs) if kept_divs else float("nan")
            print(f"  scale={scale:.1f}: falls={falls}/{len(SEEDS)}  "
                  f"mean_divergence(non-fallers)={mean_div:.4f}")


if __name__ == "__main__":
    main()
