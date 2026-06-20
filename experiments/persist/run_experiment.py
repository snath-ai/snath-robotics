"""
PERSIST: Full experimental validation.

Generates:
  - figures/divergence_curves.pdf  (6 curves, one per phase)
  - figures/zone_detection.pdf     (box plots: D per zone)
  - results.json                   (numbers for the paper)

Usage:
    poetry run python experiments/run_experiment.py
"""

from __future__ import annotations

import sys
import json
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from environments.ice_world import IceWorldEnv, ZONES
from experiments.persistence_loop import (
    Adaptor, AdaptorLibrary, AdaptorTournament, PersistenceLoop,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SEEDS                = [42, 7, 13, 99, 2026]
BASELINE_STEPS       = 300      # total obs to collect across episodes
STEPS_PER_ZONE       = 80       # steps per zone in detection experiment
FIGURES_DIR          = Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
PHASE1_N_CANDIDATES  = 40       # delta candidates evaluated in Phase 1 search

# A constant action that keeps Hopper briefly upright (from diagnostic)
STABLE_ACTION  = np.array([0.5, 0.5, 0.5], dtype=np.float32)

# Adaptor deltas: corrective action offsets per zone
ICE_DELTA      = np.array([ 0.25,  0.20,  0.15], dtype=np.float32)
SLOPE_DELTA    = np.array([-0.10,  0.35,  0.25], dtype=np.float32)
WIND_DELTA     = np.array([ 0.20, -0.15,  0.30], dtype=np.float32)
# Combined: handles ice friction + slope gravity simultaneously
COMBINED_DELTA = ICE_DELTA + SLOPE_DELTA


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

def stable_policy(obs: np.ndarray, delta: np.ndarray | None = None,
                  rng: np.random.Generator | None = None) -> np.ndarray:
    """Constant action + small noise + optional corrective delta."""
    action = STABLE_ACTION.copy()
    if rng is not None:
        action = action + 0.05 * rng.standard_normal(3).astype(np.float32)
    if delta is not None:
        action = action + delta
    return np.clip(action, -1.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Baseline builder — accumulates across episode resets
# ---------------------------------------------------------------------------

def build_baseline(env: IceWorldEnv, seed: int,
                   rng: np.random.Generator) -> IceWorldEnv:
    """
    Collect BASELINE_STEPS normal-zone observations across one or more
    episodes. Returns env with baseline_built=True.
    """
    obs, _ = env.reset(seed=seed, keep_baseline=False)
    collected = 0

    while collected < BASELINE_STEPS:
        action = stable_policy(obs, rng=rng)
        obs, _, terminated, truncated, info = env.step(action)
        if info["zone"] == "normal":
            collected += 1
        if terminated or truncated:
            obs, _ = env.reset(seed=seed, keep_baseline=True)

    return env


# ---------------------------------------------------------------------------
# Experiment 1: Zone Detection
# ---------------------------------------------------------------------------

def experiment_detection(seed: int) -> dict[str, list[float]]:
    """
    Build baseline on Normal zone, then force each zone and collect D(t).
    """
    rng = np.random.default_rng(seed)
    env = IceWorldEnv(baseline_steps=BASELINE_STEPS, divergence_window=20)
    build_baseline(env, seed, rng)

    saved_mean = env._baseline_mean.copy()
    saved_std  = env._baseline_std.copy()

    results: dict[str, list[float]] = {}
    for zone in ["normal", "ice", "ice_slope", "force", "novel"]:
        # Fresh episode, keep baseline, force zone from the start
        obs, _ = env.reset(seed=seed, keep_baseline=True)
        env._baseline_mean = saved_mean.copy()
        env._baseline_std  = saved_std.copy()
        env._baseline_built = True
        env._current_zone = zone

        divs = []
        for _ in range(STEPS_PER_ZONE):
            action = stable_policy(obs, rng=rng)
            obs, _, terminated, truncated, info = env.step(action)
            divs.append(info["divergence"])
            if terminated or truncated:
                obs, _ = env.reset(seed=seed, keep_baseline=True)
                env._baseline_mean = saved_mean.copy()
                env._baseline_std  = saved_std.copy()
                env._baseline_built = True
                env._current_zone = zone
        results[zone] = divs

    env.close()
    return results


# ---------------------------------------------------------------------------
# Adaptor signatures — constructed from obs statistics of each zone
# ---------------------------------------------------------------------------

def make_signatures() -> dict[str, np.ndarray]:
    """
    Build divergence signatures from a single seed to use across all phases.
    """
    rng = np.random.default_rng(42)
    env = IceWorldEnv(baseline_steps=BASELINE_STEPS, divergence_window=20)
    build_baseline(env, seed=42, rng=rng)

    saved_mean = env._baseline_mean.copy()
    saved_std  = env._baseline_std.copy()

    sigs = {}
    for zone in ["ice", "ice_slope", "force", "novel"]:
        obs, _ = env.reset(seed=42, keep_baseline=True)
        env._baseline_mean = saved_mean.copy()
        env._baseline_std  = saved_std.copy()
        env._baseline_built = True
        env._current_zone = zone

        zone_obs = []
        for _ in range(50):
            action = stable_policy(obs, rng=rng)
            obs, _, terminated, truncated, info = env.step(action)
            zone_obs.append(obs.copy())
            if terminated or truncated:
                obs, _ = env.reset(seed=42, keep_baseline=True)
                env._baseline_mean = saved_mean.copy()
                env._baseline_std  = saved_std.copy()
                env._baseline_built = True
                env._current_zone = zone

        # Signature = mean z-score of observations in this zone
        arr = np.array(zone_obs)
        sig = ((arr.mean(axis=0) - saved_mean) / saved_std)
        norm = np.linalg.norm(sig)
        sigs[zone] = sig / norm if norm > 1e-8 else sig

    env.close()
    return sigs


# ---------------------------------------------------------------------------
# Library builder
# ---------------------------------------------------------------------------

def _combined_sig(sigs: dict) -> np.ndarray:
    sig = 0.5 * sigs["ice"] + 0.5 * sigs["ice_slope"]
    sig /= np.linalg.norm(sig)
    return sig


def build_library(sigs: dict) -> AdaptorLibrary:
    """Full library: primitive adaptors + compound combined adaptor."""
    lib = AdaptorLibrary(similarity_threshold=0.5)
    lib.store(Adaptor("ice",      ICE_DELTA.copy(),      sigs["ice"],       success_count=2))
    lib.store(Adaptor("slope",    SLOPE_DELTA.copy(),    sigs["ice_slope"], success_count=1))
    lib.store(Adaptor("wind",     WIND_DELTA.copy(),     sigs["force"],     success_count=1))
    lib.store(Adaptor("combined", COMBINED_DELTA.copy(), _combined_sig(sigs), success_count=1))
    return lib


def build_single_library(sigs: dict) -> AdaptorLibrary:
    """Primitive adaptors only — no compound adaptor."""
    lib = AdaptorLibrary(similarity_threshold=0.5)
    lib.store(Adaptor("ice",   ICE_DELTA.copy(),   sigs["ice"],      success_count=2))
    lib.store(Adaptor("slope", SLOPE_DELTA.copy(), sigs["ice_slope"], success_count=1))
    lib.store(Adaptor("wind",  WIND_DELTA.copy(),  sigs["force"],    success_count=1))
    return lib


# ---------------------------------------------------------------------------
# Experiment 2: Six-Phase Persistence Protocol
# ---------------------------------------------------------------------------

def run_phase_experiment(seed: int, sigs: dict) -> dict:
    """Run all 6 phases against real IceWorldEnv."""
    rng = np.random.default_rng(seed)
    results = {}

    # Build one baseline for this seed, save stats
    env = IceWorldEnv(baseline_steps=BASELINE_STEPS, divergence_window=20,
                      escalation_threshold=3.0)
    build_baseline(env, seed, rng)
    saved_mean = env._baseline_mean.copy()
    saved_std  = env._baseline_std.copy()

    def restore(zone: str):
        """Reset episode and restore baseline with forced zone."""
        obs, _ = env.reset(seed=seed, keep_baseline=True)
        env._baseline_mean = saved_mean.copy()
        env._baseline_std  = saved_std.copy()
        env._baseline_built = True
        env._current_zone = zone
        return obs

    def make_step_fn(zone: str):
        """step_fn(action) -> (obs, divergence) for PersistenceLoop."""
        state = [None]   # mutable holder to avoid nonlocal issues
        def step_fn(action):
            env._current_zone = zone
            o, _, term, trunc, info = env.step(action)
            if term or trunc:
                state[0] = restore(zone)
                return state[0], info["divergence"]
            state[0] = o
            return o, info["divergence"]
        return step_fn

    def base_action_fn(obs, delta):
        return stable_policy(obs, delta=delta, rng=rng)

    def get_entry_divergence(zone: str, warm_steps: int = 15) -> tuple:
        """Take warm_steps in zone, return (obs, divergence_signature, current_div)."""
        obs = restore(zone)
        obs_stream = []
        div_last = 0.0
        for _ in range(warm_steps):
            action = stable_policy(obs, rng=rng)
            obs, _, term, trunc, info = env.step(action)
            obs_stream.append(obs.copy())
            div_last = info["divergence"]
            env._current_zone = zone
            if term or trunc:
                obs = restore(zone)
        # Signature: mean z-score of observed stream
        arr = np.array(obs_stream)
        sig = ((arr.mean(axis=0) - saved_mean) / saved_std)
        norm = np.linalg.norm(sig)
        sig = sig / norm if norm > 1e-8 else sig
        return obs, sig, div_last

    def train_adaptor_from_dhard(
        zone: str,
        dhard_sig: np.ndarray,
        n_candidates: int = 40,
        trial_steps: int = 8,
        search_range: float = 0.4,
        rng_offset: int = 1,
        name: str = "ice_learned",
    ) -> tuple:
        """
        Learn a corrective action delta from a D-hard escalation stream.

        After Phase 1 escalates (no adaptor found), sample random candidate
        deltas, evaluate each by running trial_steps from a fresh env entry,
        and return the highest-rate delta as a trained Adaptor.

        This closes the loop claimed in the paper: Phase 1 D-hard events
        trigger offline delta search; Phase 2 uses the result.

        IMPORTANT: uses rng=None (deterministic base policy) so that the
        shared episode rng is not consumed — Phase 2 onwards sees the same
        rng state as if training never happened.
        """
        search_rng = np.random.default_rng(seed + rng_offset)   # independent from episode rng
        best_delta = np.zeros(3, dtype=np.float32)
        best_rate = -np.inf

        print(f"\n  [D-hard Training '{name}'] zone='{zone}' | "
              f"{n_candidates} candidates × {trial_steps} steps | range=±{search_range}")

        for _ in range(n_candidates):
            candidate = search_rng.uniform(-search_range, search_range, size=3).astype(np.float32)

            # Fresh env entry — use deterministic policy (rng=None) so that
            # the shared `rng` state is not consumed during training
            obs_t = restore(zone)
            obs_stream_t = []
            div_t = 0.0
            for _ in range(10):                         # warm_steps=10
                a = stable_policy(obs_t, rng=None)      # no shared rng noise
                obs_t, _, term, trunc, info = env.step(a)
                obs_stream_t.append(obs_t.copy())
                div_t = info["divergence"]
                env._current_zone = zone
                if term or trunc:
                    obs_t = restore(zone)
            d_start = div_t

            sfn = make_step_fn(zone)
            obs_curr, d_curr = obs_t, d_start
            for _ in range(trial_steps):
                action = stable_policy(obs_curr, delta=candidate, rng=None)
                obs_curr, d_curr = sfn(action)

            rate = (d_start - d_curr) / trial_steps
            if rate > best_rate:
                best_rate = rate
                best_delta = candidate.copy()

        print(f"  [D-hard Training] Learned delta: {best_delta.round(3)} | "
              f"best_rate={best_rate:+.4f}/step")

        adaptor = Adaptor(
            name=name,
            delta=best_delta,
            divergence_signature=dhard_sig,
            success_count=0,
        )
        return adaptor, best_rate

    # ----------------------------------------------------------------
    # Phase 1: Encounter — ice zone, no adaptor → escalate, train
    # ----------------------------------------------------------------
    print("\n=== Phase 1: Encounter (Ice, no adaptor) ===")
    obs, enc_sig, current_div = get_entry_divergence("ice", warm_steps=20)

    enc_lib = AdaptorLibrary(similarity_threshold=0.5)
    step_fn1 = make_step_fn("ice")
    p1_loop = PersistenceLoop(library=enc_lib, action_dim=3,
                              normalisation_threshold=0.8,
                              escalation_threshold=3.0, patience=20)
    p1 = p1_loop.run(obs, enc_sig, current_div, base_action_fn, step_fn1,
                     zone="ice", phase_label="encounter")
    results["phase1_encounter"] = {
        "success": p1.success, "steps": p1.steps_to_resolution,
        "final_div": p1.final_divergence, "escalated": p1.escalated,
        "curve": p1.divergence_curve,
    }

    # D-hard delta search: 40 candidates, ±0.4 range, 8 trial steps per candidate.
    learned_adaptor, learned_rate = train_adaptor_from_dhard(
        "ice", enc_sig,
        n_candidates=PHASE1_N_CANDIDATES, search_range=0.40, trial_steps=8,
        rng_offset=1, name="ice_learned",
    )
    results["phase1_training"] = {
        "learned_delta": learned_adaptor.delta.tolist(),
        "learned_rate": float(learned_rate),
        "n_candidates": PHASE1_N_CANDIDATES,
    }

    # ----------------------------------------------------------------
    # Phase 2: First Try — ice zone, trained adaptor from Phase 1 D-hard stream
    # ----------------------------------------------------------------
    print("\n=== Phase 2: First Try (Ice, D-hard trained adaptor) ===")
    obs, _, current_div = get_entry_divergence("ice")

    lib2 = AdaptorLibrary(similarity_threshold=0.5)
    lib2.store(Adaptor(learned_adaptor.name, learned_adaptor.delta.copy(),
                       sigs["ice"], success_count=0))
    step_fn2 = make_step_fn("ice")
    P2_DELTA_INIT = 0.05
    P2_DELTA_INC  = 0.06
    p2_loop = PersistenceLoop(library=lib2, action_dim=3,
                              delta_init=P2_DELTA_INIT, delta_increment=P2_DELTA_INC,
                              delta_max=0.5, normalisation_threshold=0.8,
                              escalation_threshold=3.0, patience=40)
    p2 = p2_loop.run(obs, sigs["ice"], current_div, base_action_fn, step_fn2,
                     zone="ice", phase_label="first_try")
    results["phase2_first_try"] = {
        "success": p2.success, "steps": p2.steps_to_resolution,
        "final_div": p2.final_divergence, "escalated": p2.escalated,
        "curve": p2.divergence_curve,
    }

    # ----------------------------------------------------------------
    # Phase 3: Scope Boundary — ice+slope, single adaptors insufficient
    # ----------------------------------------------------------------
    print("\n=== Phase 3: Scope Boundary (Ice+Slope, primitives only) ===")
    obs, _, current_div = get_entry_divergence("ice_slope")

    # Single-adaptor library only — no compound adaptor
    lib3 = build_single_library(sigs)
    step_fn3 = make_step_fn("ice_slope")
    sig_blend = 0.6 * sigs["ice"] + 0.4 * sigs["ice_slope"]
    sig_blend /= np.linalg.norm(sig_blend)
    p3_loop = PersistenceLoop(library=lib3, action_dim=3,
                              delta_init=0.05, delta_increment=0.06,
                              delta_max=0.5, normalisation_threshold=0.8,
                              escalation_threshold=3.0, patience=40,
                              composition_residual_threshold=1.2)
    p3 = p3_loop.run(obs, sig_blend, current_div, base_action_fn, step_fn3,
                     zone="ice_slope", phase_label="composition")
    results["phase3_composition"] = {
        "success": p3.success, "steps": p3.steps_to_resolution,
        "final_div": p3.final_divergence, "escalated": p3.escalated,
        "curve": p3.divergence_curve,
    }

    # ----------------------------------------------------------------
    # Phase 4: Exhaustion — novel zone, all three combined, scope exceeded
    # ----------------------------------------------------------------
    print("\n=== Phase 4: Exhaustion (Novel) ===")
    obs, _, current_div = get_entry_divergence("novel")

    lib4 = build_library(sigs)
    step_fn4 = make_step_fn("novel")
    # Novel signature: far from all stored adaptors
    sig_novel = (sigs["novel"] if "novel" in sigs
                 else np.random.default_rng(0).standard_normal(11))
    sig_novel = sig_novel / np.linalg.norm(sig_novel)
    p4_loop = PersistenceLoop(library=lib4, action_dim=3,
                              normalisation_threshold=0.8,
                              escalation_threshold=3.0, patience=20)
    p4 = p4_loop.run(obs, sig_novel, current_div, base_action_fn, step_fn4,
                     zone="novel", phase_label="exhaustion")
    results["phase4_exhaustion"] = {
        "success": p4.success, "steps": p4.steps_to_resolution,
        "final_div": p4.final_divergence, "escalated": p4.escalated,
        "curve": p4.divergence_curve,
    }

    # ----------------------------------------------------------------
    # Phase 5-cold: Second encounter WITHOUT memory (baseline)
    # Enters ice zone fresh with an empty library — must re-escalate,
    # re-run the full delta search, then converge. Shows empirically
    # what Phase 5 would cost if memory did not exist.
    #
    # Uses an independent cold_rng for actions so that the shared
    # episode rng is not consumed — Phase 5-warm sees the same rng
    # state it would have seen without this cold run.
    # ----------------------------------------------------------------
    print("\n=== Phase 5-cold: Second Encounter (No Memory) ===")
    cold_rng = np.random.default_rng(seed + 100)   # independent, never shared
    def cold_action_fn(obs, delta):
        return stable_policy(obs, delta=delta, rng=cold_rng)

    obs, cold_sig, current_div = get_entry_divergence("ice")

    # Empty library — memory has been wiped
    cold_lib1 = AdaptorLibrary(similarity_threshold=0.5)
    step_fn5c1 = make_step_fn("ice")
    p5c_enc_loop = PersistenceLoop(library=cold_lib1, action_dim=3,
                                   normalisation_threshold=0.8,
                                   escalation_threshold=3.0, patience=20)
    p5c_enc = p5c_enc_loop.run(obs, cold_sig, current_div, cold_action_fn, step_fn5c1,
                                zone="ice", phase_label="cold_encounter")
    # (escalates — no adaptor, exactly like Phase 1)

    # Re-run delta search with a fresh independent rng (rng_offset=2)
    cold_adaptor, cold_rate = train_adaptor_from_dhard(
        "ice", cold_sig,
        n_candidates=PHASE1_N_CANDIDATES, search_range=0.40, trial_steps=8,
        rng_offset=2, name="ice_cold_learned",
    )

    # Converge with freshly-trained adaptor (no stored scale — cold delta_init)
    obs, _, current_div = get_entry_divergence("ice")
    cold_lib2 = AdaptorLibrary(similarity_threshold=0.5)
    cold_lib2.store(Adaptor(cold_adaptor.name, cold_adaptor.delta.copy(),
                            sigs["ice"], success_count=0))
    step_fn5c2 = make_step_fn("ice")
    p5c_conv_loop = PersistenceLoop(library=cold_lib2, action_dim=3,
                                    delta_init=P2_DELTA_INIT, delta_increment=P2_DELTA_INC,
                                    delta_max=0.5, normalisation_threshold=0.8,
                                    escalation_threshold=3.0, patience=40)
    p5c = p5c_conv_loop.run(obs, sigs["ice"], current_div, cold_action_fn, step_fn5c2,
                             zone="ice", phase_label="cold_converge")

    results["phase5_cold"] = {
        "success": p5c.success,
        "convergence_steps": p5c.steps_to_resolution,
        "search_candidates": PHASE1_N_CANDIDATES,
        "total_decisions": PHASE1_N_CANDIDATES + p5c.steps_to_resolution,
        "final_div": p5c.final_divergence,
        "escalated": p5c.escalated,
        "cold_rate": float(cold_rate),
    }

    # ----------------------------------------------------------------
    # Phase 5: Memory — ice again, same learned adaptor with stored successes
    # Memory effect: start at 60% of the convergence scale from Phase 2,
    # skipping the lower-delta attempts that are known to be insufficient.
    # ----------------------------------------------------------------
    print("\n=== Phase 5: Memory (Ice, second encounter) ===")
    obs, _, current_div = get_entry_divergence("ice")

    # Memory effect: second encounter retrieves the same adaptor, but with
    # success_count=5 (prior successes) and a higher starting delta derived from
    # the stored resolution step count — skipping the low-delta warm-up phase.
    p2_resolve_scale = (P2_DELTA_INIT + p2.steps_to_resolution * P2_DELTA_INC) if p2.success else P2_DELTA_INIT
    memory_delta_init = p2_resolve_scale * 0.60   # approach resolved scale from below

    lib5 = AdaptorLibrary(similarity_threshold=0.5)
    lib5.store(Adaptor(learned_adaptor.name, learned_adaptor.delta.copy(),
                       sigs["ice"], success_count=5))
    step_fn5 = make_step_fn("ice")
    p5_loop = PersistenceLoop(library=lib5, action_dim=3,
                              delta_init=memory_delta_init,
                              delta_increment=P2_DELTA_INC,
                              delta_max=0.5, normalisation_threshold=0.8,
                              escalation_threshold=3.0, patience=40)
    p5 = p5_loop.run(obs, sigs["ice"], current_div, base_action_fn, step_fn5,
                     zone="ice", phase_label="memory")
    results["phase5_memory"] = {
        "success": p5.success, "steps": p5.steps_to_resolution,
        "final_div": p5.final_divergence, "escalated": p5.escalated,
        "curve": p5.divergence_curve,
    }

    # ----------------------------------------------------------------
    # Phase 6: Tournament → Resolution
    # Step 1: tournament identifies compound winner from ambiguous candidates
    # Step 2: fresh env, persist with winner to close the loop
    # ----------------------------------------------------------------
    print("\n=== Phase 6: Tournament + Resolution (Ice+Slope) ===")

    sig_mid = 0.5 * sigs["ice"] + 0.5 * sigs["ice_slope"]
    sig_mid /= np.linalg.norm(sig_mid)

    # Step 1 — tournament with full library, ambiguous mid-signature
    obs, _, current_div = get_entry_divergence("ice_slope")
    lib6_full = build_library(sigs)
    step_fn6a = make_step_fn("ice_slope")
    candidates6 = lib6_full.retrieve_top_k(sig_mid, k=3, band=0.3)
    t6 = AdaptorTournament(trial_steps=5).run(
        candidates6, obs, current_div, base_action_fn, step_fn6a, delta_init=0.05
    )
    winner6 = t6.winner
    print(f"\n  [Phase 6] Tournament winner: '{winner6.name}' — persisting from fresh state...")

    # Step 2 — fresh env reset, persist with winner only (no state contamination)
    obs2, _, current_div2 = get_entry_divergence("ice_slope")
    lib6_winner = AdaptorLibrary(similarity_threshold=0.5)
    lib6_winner.store(winner6)
    step_fn6b = make_step_fn("ice_slope")
    p6_loop = PersistenceLoop(library=lib6_winner, action_dim=3,
                              delta_init=0.05, delta_increment=0.06,
                              delta_max=0.5, normalisation_threshold=0.8,
                              escalation_threshold=3.0, patience=40)
    p6 = p6_loop.run(obs2, sig_mid, current_div2, base_action_fn, step_fn6b,
                     zone="ice_slope", phase_label="tournament")

    # Combine tournament trial curves + persistence curve for the figure
    tour_curve: list[float] = []
    for curve in t6.divergence_curves.values():
        tour_curve.extend(curve[1:])

    results["phase6_tournament"] = {
        "success": p6.success, "steps": p6.steps_to_resolution,
        "final_div": p6.final_divergence, "escalated": p6.escalated,
        "curve": tour_curve + p6.divergence_curve,
        "tournament_winner": winner6.name,
        "tournament_rates": t6.rates,
    }

    env.close()
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

PHASE_COLORS = {
    "phase1_encounter":   "#e74c3c",
    "phase2_first_try":   "#e67e22",
    "phase3_composition": "#f1c40f",
    "phase4_exhaustion":  "#95a5a6",
    "phase5_memory":      "#2ecc71",
    "phase6_tournament":  "#3498db",
}

PHASE_LABELS = {
    "phase1_encounter":   "Phase 1 — Encounter",
    "phase2_first_try":   "Phase 2 — First try",
    "phase3_composition": "Phase 3 — Composition",
    "phase4_exhaustion":  "Phase 4 — Exhaustion",
    "phase5_memory":      "Phase 5 — Memory",
    "phase6_tournament":  "Phase 6 — Tournament",
}


def plot_divergence_curves(all_results: list[dict]):
    fig, ax = plt.subplots(figsize=(10, 5))

    for pk in PHASE_LABELS:
        curves = [r[pk]["curve"] for r in all_results if pk in r and r[pk]["curve"]]
        if not curves:
            continue
        max_len = max(len(c) for c in curves)
        padded = [c + [c[-1]] * (max_len - len(c)) for c in curves]
        arr = np.array(padded)
        mean = arr.mean(axis=0)
        std  = arr.std(axis=0)
        xs   = np.arange(len(mean))
        color = PHASE_COLORS[pk]
        ax.plot(xs, mean, color=color, label=PHASE_LABELS[pk], linewidth=1.8)
        ax.fill_between(xs, mean - std, mean + std, color=color, alpha=0.15)

    ax.axhline(0.8, color="black", linestyle="--", linewidth=1.0,
               label="Normalisation threshold (0.8)")
    ax.axhline(3.0, color="black", linestyle=":",  linewidth=1.0,
               label="Escalation threshold (3.0)")

    ax.set_xlabel("Steps", fontsize=12)
    ax.set_ylabel("Divergence D(t)", fontsize=12)
    ax.set_title("PERSIST: Divergence signal across six phases "
                 f"(n={len(all_results)} seeds, mean ± std)",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = FIGURES_DIR / "divergence_curves.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {out}")


def plot_zone_detection(all_detection: list[dict]):
    zones = ["normal", "ice", "ice_slope", "force", "novel"]
    data_per_zone = {z: [] for z in zones}
    for det in all_detection:
        for z in zones:
            data_per_zone[z].extend(det.get(z, []))

    fig, ax = plt.subplots(figsize=(8, 4))
    bp = ax.boxplot(
        [data_per_zone[z] for z in zones],
        tick_labels=[z.replace("_", "\n") for z in zones],
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
    )
    colors = ["#2ecc71", "#3498db", "#9b59b6", "#e67e22", "#e74c3c"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.axhline(0.8, color="gray", linestyle="--", linewidth=1.0,
               label="Normalisation threshold")
    ax.axhline(3.0, color="gray", linestyle=":",  linewidth=1.0,
               label="Escalation threshold")

    ax.set_ylabel("Divergence D(t)", fontsize=12)
    ax.set_title("Zone detection: Divergence per terrain type "
                 f"(n={len(all_detection)} seeds × {STEPS_PER_ZONE} steps/zone)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    out = FIGURES_DIR / "zone_detection.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {out}")


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def aggregate(all_results: list[dict]) -> dict:
    summary = {}
    for pk in PHASE_LABELS:
        rows = [r[pk] for r in all_results if pk in r]
        if not rows:
            continue
        steps = [r["steps"] for r in rows]
        fdivs = [r["final_div"] for r in rows]
        esc   = [r["escalated"] for r in rows]
        summary[pk] = {
            "success_rate":   sum(not e for e in esc) / len(esc),
            "mean_steps":     float(np.mean(steps)),
            "std_steps":      float(np.std(steps)),
            "mean_final_div": float(np.mean(fdivs)),
        }

    # Memory speedup: Phase 5-cold (measured) vs Phase 5-warm (measured)
    # cold = re-searched second encounter (N_c candidates + convergence steps)
    # warm = memory-aided second encounter (convergence steps only)
    cold_rows = [r["phase5_cold"] for r in all_results if "phase5_cold" in r]
    warm_rows = [r["phase5_memory"] for r in all_results if "phase5_memory" in r]
    if cold_rows and warm_rows:
        cold_totals = [r["total_decisions"] for r in cold_rows]
        warm_steps  = [r["steps"] for r in warm_rows]
        cold_mean = float(np.mean(cold_totals))
        cold_std  = float(np.std(cold_totals))
        warm_mean = float(np.mean(warm_steps))
        summary["memory_speedup"] = cold_mean / warm_mean
        summary["first_encounter_total"]     = cold_mean
        summary["first_encounter_total_std"] = cold_std
        summary["second_encounter_total"]    = warm_mean
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("PERSIST Experiment — IceWorld Validation")
    print("=" * 60)
    print(f"Seeds: {SEEDS}")
    print(f"Baseline steps: {BASELINE_STEPS}")
    print(f"Steps per zone (detection): {STEPS_PER_ZONE}")

    # Pre-compute signatures from seed 42
    print("\nBuilding divergence signatures from seed 42...")
    sigs = make_signatures()
    print("Signatures built for:", list(sigs.keys()))
    for k, v in sigs.items():
        print(f"  {k}: norm={np.linalg.norm(v):.3f}, top_3={v[:3].round(3)}")

    # --- Detection experiment ---
    print("\n[1/2] Zone detection experiment...")
    all_detection = []
    for seed in SEEDS:
        print(f"  Seed {seed}...", end=" ", flush=True)
        det = experiment_detection(seed)
        all_detection.append(det)
        means = {z: float(np.mean(v)) for z, v in det.items()}
        print(" | ".join(f"{z}={m:.2f}" for z, m in means.items()))

    # --- Phase experiment ---
    print("\n[2/2] Six-phase persistence experiment...")
    all_results = []
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        result = run_phase_experiment(seed, sigs)
        all_results.append(result)

    # --- Aggregate ---
    summary = aggregate(all_results)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for pk, label in PHASE_LABELS.items():
        if pk in summary:
            s = summary[pk]
            print(f"  {label:30s} | success={s['success_rate']:.0%} | "
                  f"steps={s['mean_steps']:.1f}±{s['std_steps']:.1f} | "
                  f"D_final={s['mean_final_div']:.3f}")

    if "memory_speedup" in summary:
        fe  = summary.get("first_encounter_total", 0)
        fes = summary.get("first_encounter_total_std", 0)
        se  = summary.get("second_encounter_total", 0)
        print(f"\n  Memory speedup  cold={fe:.1f}±{fes:.1f} decisions "
              f"→ warm={se:.1f} decisions: {summary['memory_speedup']:.2f}×")

    print("\n  Zone detection (mean D per zone):")
    for z in ["normal", "ice", "ice_slope", "force", "novel"]:
        all_divs = []
        for det in all_detection:
            all_divs.extend(det.get(z, []))
        if all_divs:
            print(f"    {z:12s}: mean={np.mean(all_divs):.3f}  "
                  f"std={np.std(all_divs):.3f}  "
                  f"max={np.max(all_divs):.3f}")

    print("\nGenerating figures...")
    plot_divergence_curves(all_results)
    plot_zone_detection(all_detection)

    out_json = Path(__file__).parent.parent / "results.json"
    with open(out_json, "w") as f:
        json.dump({"summary": summary, "seeds": SEEDS,
                   "detection": {
                       z: float(np.mean([d.get(z, [0]) for d in all_detection
                                         for _ in range(len(d.get(z, [])))]))
                       for z in ["normal", "ice", "ice_slope", "force", "novel"]
                   }}, f, indent=2)
    print(f"Results saved: {out_json}")
    print("\nDone.")
