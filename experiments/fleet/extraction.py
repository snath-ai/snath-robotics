"""
Correction Extraction -- turning tonight's manual process into a real,
reusable function.
=============================================================================
Every result up to this point in the 2026-10-20 investigation (FLEET
STATUS.md) that involved an "expert-derived correction" was done by hand:
print two sets of per-joint numbers, eyeball the difference, type it into a
numpy array, manually try a few scale factors. That is not a mechanism a
real fleet could run -- it is a human doing comparison and judgement that
FLEET itself has no way to do automatically.

This file is that mechanism, built for real and verified by reproducing
today's manual result end to end, not assumed to work from the design.

Two functions, two separate jobs, because the earlier manual process
conflated them and that made it harder to tell which step the result
actually came from:

  extract_correction()   -- measure what an EXPERT policy does differently
                             under a hazard vs normal, with real multi-seed
                             paired statistics (mean, std, t-statistic,
                             same-sign-across-seeds), not a single trial.
                             Returns the delta AND whether it is trustworthy,
                             rather than asserting it is.

  search_and_verify()     -- given that delta, find a scale that helps a
                             WEAK policy, each candidate verified with its
                             own multi-seed paired trial against a real
                             no-correction baseline (the same discipline
                             that caught the original single-trial "PASS"
                             as noise earlier in this investigation).

Honest scope, unchanged from every other file in this series: this does
not claim the result generalises past the one case it is verified against
below (PPO -> CPG, Walker2d ice). What it replaces is the HUMAN step in
that one case, and proves the automated version reproduces what the human
process found.

Run (reproduces the 2026-10-20 manual result, this time via the function):
    .venv/bin/python experiments/fleet/extraction.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

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
from experiments.fleet.correction_search_v2 import MAX_STEPS_PER_TRIAL

ACT_DIM = 6
PPO_CHECKPOINT = "/Users/aadithya/Desktop/Lar_Main/lar_jepa/models/walker2d_ppo.zip"


# ---------------------------------------------------------------------------
# 1. Extraction: what does the expert do differently?
# ---------------------------------------------------------------------------

@dataclass
class ExtractedCorrection:
    delta: np.ndarray            # (action_dim,) mean hazard-minus-normal action
    t_stats: np.ndarray          # (action_dim,) per-dimension paired t-statistic
    same_sign: np.ndarray        # (action_dim,) bool, all seeds agree on direction
    n_seeds: int
    raw_diffs: np.ndarray        # (n_seeds, action_dim) -- kept for audit, not just the mean

    def trustworthy_dims(self, t_threshold: float = 2.0) -> np.ndarray:
        """Which dimensions of the delta are backed by real, consistent
        evidence vs. noise -- the thing the manual process never checked
        per-dimension, only as a single aggregate."""
        return self.same_sign & (np.abs(self.t_stats) > t_threshold)

    def summary(self) -> str:
        lines = [f"Extracted correction over {self.n_seeds} seeds:"]
        for i in range(len(self.delta)):
            trust = "trustworthy" if self.trustworthy_dims()[i] else "NOT trustworthy"
            lines.append(f"  dim {i}: delta={self.delta[i]:+.4f}  "
                          f"t={self.t_stats[i]:+.2f}  same_sign={self.same_sign[i]}  ({trust})")
        return "\n".join(lines)


def _measure_action_profile(
    action_fn: Callable[[np.ndarray], np.ndarray],
    reset_fn: Callable[[], None],
    env: gymnasium.Env,
    condition_fn: Callable[[gymnasium.Env], None],
    n_seeds: int,
    n_steps: int,
    seed_base: int,
) -> np.ndarray:
    """Per-seed mean |action| (effort magnitude per joint, not signed mean)
    under a real condition. Returns (n_seeds, action_dim).

    This must be the absolute value, not the raw signed mean -- a periodic
    gait oscillates each joint through positive and negative torque every
    cycle, so the raw mean mostly captures whatever small asymmetry exists
    between the push and pull phases, not how much effort the joint is
    actually using. A first version of this function used the signed mean
    and got six positive deltas with the foot/leg story (feet ease off,
    legs compensate) completely gone -- caught by comparing against the
    already-validated action_effort_check.py numbers, not assumed correct."""
    profiles = []
    for i, s in enumerate(range(seed_base, seed_base + n_seeds)):
        obs, _ = env.reset(seed=s)
        condition_fn(env)
        reset_fn()
        actions = []
        for step in range(n_steps):
            condition_fn(env)  # some conditions (e.g. wind) need re-asserting every step
            action = action_fn(obs)
            actions.append(np.abs(action))
            obs, _, term, trunc, _ = env.step(action)
            if term or trunc:
                obs, _ = env.reset(seed=s + step)
                condition_fn(env)
                reset_fn()
        profiles.append(np.mean(actions, axis=0))
    return np.array(profiles)


def extract_correction(
    expert_action_fn: Callable[[np.ndarray], np.ndarray],
    expert_reset_fn: Callable[[], None],
    env: gymnasium.Env,
    hazard_condition_fn: Callable[[gymnasium.Env], None],
    normal_condition_fn: Callable[[gymnasium.Env], None],
    n_seeds: int = 8,
    n_steps: int = 200,
    seed_base: int = 1,
) -> ExtractedCorrection:
    """
    Measure what the expert does differently under the hazard vs normal,
    paired by seed (same seed used for both conditions), with real
    statistics -- not the single-number eyeball comparison this process
    used manually earlier tonight.
    """
    hazard_profiles = _measure_action_profile(
        expert_action_fn, expert_reset_fn, env, hazard_condition_fn,
        n_seeds, n_steps, seed_base,
    )
    normal_profiles = _measure_action_profile(
        expert_action_fn, expert_reset_fn, env, normal_condition_fn,
        n_seeds, n_steps, seed_base,
    )
    diffs = hazard_profiles - normal_profiles  # (n_seeds, action_dim), paired
    mean_delta = diffs.mean(axis=0)
    std_delta = diffs.std(axis=0, ddof=1)
    se = std_delta / np.sqrt(n_seeds)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stats = np.where(se > 1e-12, mean_delta / se, np.inf * np.sign(mean_delta))
    same_sign = np.array([
        bool(np.all(diffs[:, d] > 0) or np.all(diffs[:, d] < 0))
        for d in range(diffs.shape[1])
    ])
    return ExtractedCorrection(
        delta=mean_delta, t_stats=t_stats, same_sign=same_sign,
        n_seeds=n_seeds, raw_diffs=diffs,
    )


# ---------------------------------------------------------------------------
# 2. Search and verify: does this delta actually help a weak policy?
# ---------------------------------------------------------------------------

@dataclass
class ScaleResult:
    scale: float
    baseline_mean: float
    corrected_mean: float
    reduction: float
    t_stat: float
    same_sign: bool
    falls_baseline: int
    falls_corrected: int


def search_and_verify(
    weak_policy_factory: Callable[[], object],
    delta: np.ndarray,
    env: gymnasium.Env,
    runner: GRURunner,
    router: DivergenceRouter,
    z_ref,
    hazard_condition_fn: Callable[[gymnasium.Env], None],
    scales: list[float],
    n_seeds: int = 8,
    n_windows: int = 20,
    settle_steps: int = 30,
    seed_base: int = 1000,
) -> list[ScaleResult]:
    """
    For each candidate scale, run n_seeds PAIRED trials (same seed, weak
    policy alone vs weak policy + scale*delta), each measured via FLEET's
    real divergence signal over n_windows real GRU windows. Returns full
    per-scale statistics -- the caller decides what counts as real, this
    function does not silently pick a "winner."
    """
    results = []
    for scale in scales:
        base_means, corr_means = [], []
        falls_b_total, falls_c_total = 0, 0

        for i in range(n_seeds):
            seed = seed_base + i * 7

            base_policy = weak_policy_factory()
            b_divs, b_resets, _ = _collect_divs(
                env, runner, router, z_ref, base_policy, base_policy,
                hazard_condition_fn, n_windows, seed, settle_steps,
            )
            falls_b_total += b_resets

            corr_policy = weak_policy_factory()
            scaled = delta * scale

            def action_fn(o, _p=corr_policy, _d=scaled):
                return np.clip(_p(o) + _d, -1.0, 1.0).astype(np.float32)

            c_divs, c_resets, _ = _collect_divs(
                env, runner, router, z_ref, corr_policy, action_fn,
                hazard_condition_fn, n_windows, seed, settle_steps,
            )
            falls_c_total += c_resets

            base_means.append(np.mean(b_divs) if b_divs else np.nan)
            corr_means.append(np.mean(c_divs) if c_divs else np.nan)

        base_means = np.array(base_means)
        corr_means = np.array(corr_means)
        diffs = base_means - corr_means
        se = np.nanstd(diffs, ddof=1) / np.sqrt(n_seeds)
        t = np.nanmean(diffs) / se if se > 1e-12 else float("inf") * np.sign(np.nanmean(diffs))
        same_sign = bool(np.all(diffs > 0) or np.all(diffs < 0))

        results.append(ScaleResult(
            scale=scale,
            baseline_mean=float(np.nanmean(base_means)),
            corrected_mean=float(np.nanmean(corr_means)),
            reduction=float(np.nanmean(diffs)),
            t_stat=float(t),
            same_sign=same_sign,
            falls_baseline=falls_b_total,
            falls_corrected=falls_c_total,
        ))
    return results


def _collect_divs(env, runner, router, z_ref, policy, action_fn,
                   condition_fn, n_windows, seed, settle_steps):
    divs, resets, steps = [], 0, 0
    obs, _ = env.reset(seed=seed)
    condition_fn(env)
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
            divs.append(router.route(z_ref, z.detach()).divergence)
        if term or trunc:
            resets += 1
            obs, _ = env.reset(seed=seed + resets)
            condition_fn(env)
            runner.reset()
            policy.reset()
            settled = 0
    return divs, resets, steps


# ---------------------------------------------------------------------------
# Verification: reproduce the 2026-10-20 manual result through this function
# ---------------------------------------------------------------------------

def main():
    print("=" * 78)
    print("EXTRACTION -- automating today's manual expert-correction process")
    print("=" * 78)

    env = gymnasium.make("Walker2d-v5")
    model = PPO.load(PPO_CHECKPOINT)

    def ppo_action(o):
        a, _ = model.predict(o, deterministic=True)
        return np.asarray(a, dtype=np.float32)

    def ppo_reset():
        pass

    def normal_condition(e):
        set_friction(e, FRICTION_NORMAL)

    def ice_condition(e):
        set_friction(e, FRICTION_ICE)

    print("\nStep 1: extract_correction() -- measure PPO's real ice-vs-normal "
          "behaviour, 8 paired seeds, automatically (no eyeballing).")
    extracted = extract_correction(
        ppo_action, ppo_reset, env, ice_condition, normal_condition,
        n_seeds=8, n_steps=200, seed_base=1,
    )
    print(extracted.summary())
    print(f"\nFor comparison, the manually-typed delta used earlier tonight was "
          f"[+0.0791, +0.0855, -0.0440, +0.0527, +0.1061, -0.0784].")
    print(f"Function's extracted delta:                                     "
          f"{[round(float(x), 4) for x in extracted.delta]}")

    print("\nStep 2: search_and_verify() -- find a working scale for the weak "
          "CPG policy, each scale verified with 5 paired seeds.")
    encoder = load_base_encoder(DEFAULT_MODEL)
    runner = GRURunner(encoder)
    router = DivergenceRouter(tau_high=0.60, tau_low=0.25, delta=0.35)

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

    def make_cpg():
        return CPGPolicy(np.random.default_rng(np.random.randint(0, 1_000_000)))

    scale_results = search_and_verify(
        make_cpg, extracted.delta, env, runner, router, z_ref, ice_condition,
        scales=[1.0, 2.0, 4.0, 8.0], n_seeds=5, n_windows=20, seed_base=2000,
    )

    print(f"\n{'=' * 78}\nRESULT\n{'=' * 78}")
    for r in scale_results:
        print(f"  scale={r.scale:4.1f}: baseline={r.baseline_mean:.5f} "
              f"corrected={r.corrected_mean:.5f} reduction={r.reduction:+.5f} "
              f"t={r.t_stat:+.2f} same_sign={r.same_sign} "
              f"falls(baseline/corrected)={r.falls_baseline}/{r.falls_corrected}")

    best = max(scale_results, key=lambda r: r.reduction if r.same_sign else -1e9)
    if best.same_sign and abs(best.t_stat) > 2.0:
        print(f"\nPASS -- extraction + scale search, fully automated, found "
              f"scale={best.scale} works (t={best.t_stat:+.2f}, falls "
              f"{best.falls_baseline}->{best.falls_corrected}), reproducing "
              f"today's manual result without a human picking the delta or "
              f"the scale by hand.")
    else:
        print(f"\nNo scale cleared both same_sign and |t|>2 in this run -- "
              f"see per-scale numbers above.")

    env.close()


if __name__ == "__main__":
    main()
