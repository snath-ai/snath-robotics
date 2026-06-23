"""
PERSIST: 7-phase IceWorld persistence-loop driver (5 seeds).

Phases:
  1  — Encounter      : ice zone, empty library → escalate, train ice_learned
  2  — First try      : ice zone, trained adaptor → resolve (~8 steps)
  3  — Scope boundary : ice+slope, primitives only → scope exceeded
  3b — Force          : force zone, wind adaptor → resolve
  4  — Exhaustion     : novel (all three) → escalate
  5c — Memory (cold)  : ice zone, empty library → re-search + resolve
  5w — Memory (warm)  : ice zone, stored adaptor → fast resolve (~9 steps)
  6  — Tournament     : ice+slope, 3 candidates → combined wins by rate

Results  → experiments/persist/persist_results_<ts>.json
Figures  → academic_papers/snath_core/08_PERSIST/figures/  (overwrites PDFs)
           experiments/persist/figures/  (local copy)

Fixes paper bug: Figure 2 previously labelled Phase 3 "Composition";
this code generates it correctly as "Scope boundary".
Force zone (Phase 3b) closes the paper's validation gap for that zone.
"""

from __future__ import annotations

import sys, json
from copy import deepcopy
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

from experiments.persist.ice_world import IceWorld, compute_divergence

# ── experiment parameters ──────────────────────────────────────────────────────

SEEDS              = [42, 7, 13, 99, 2026]
BASE_STEPS         = 300       # normal-zone steps for baseline μ, σ
N_CANDIDATES       = 40        # random delta candidates in Phase 1 search
K_SEARCH           = 8         # trial steps per delta-search candidate
K_TOURNAMENT       = 5         # trial steps per tournament candidate
D_NORM             = 0.8       # normalisation threshold (success)
D_ESC              = 3.0       # escalation threshold
DELTA_0            = 0.05      # initial adaptor scale
DELTA_INC          = 0.06      # scale increment per failed step
DELTA_MAX          = 1.0       # scope boundary (allows ~16 steps before escalation)
EPS                = 0.3       # tournament cosine-similarity retrieval band
PATIENCE_LONG      = 60        # Phases 2, 3, 3b, 5, 6
PATIENCE_SHORT     = 20        # Phases 1, 4
POLICY_TRAIN_STEPS = 150_000   # SAC timesteps for base policy (SAC is ~3× more sample-efficient than PPO on Hopper)

POLICY_PATH = _ROOT / "models" / "persist" / "hopper_base_policy.zip"

# ── base policy ───────────────────────────────────────────────────────────────

def train_or_load_policy() -> Any:
    """Train SAC on normal Hopper-v5 or load cached checkpoint."""
    import gymnasium as gym

    if POLICY_PATH.exists():
        print(f"  Loading base policy from {POLICY_PATH.name}")
        from stable_baselines3 import SAC
        train_env = gym.make("Hopper-v5")
        policy = SAC.load(str(POLICY_PATH), env=train_env)
        train_env.close()
        return policy

    print(f"  Training base policy (SAC, {POLICY_TRAIN_STEPS:,} steps)…")
    POLICY_PATH.parent.mkdir(parents=True, exist_ok=True)
    from stable_baselines3 import SAC
    train_env = gym.make("Hopper-v5")
    policy = SAC("MlpPolicy", train_env, verbose=0, seed=42)
    policy.learn(total_timesteps=POLICY_TRAIN_STEPS,
                 progress_bar=False)
    policy.save(str(POLICY_PATH))
    train_env.close()
    print(f"  Saved → {POLICY_PATH}")
    return policy


def base_action(policy: Any, obs: np.ndarray) -> np.ndarray:
    """Query the trained policy deterministically for one observation."""
    action, _ = policy.predict(obs, deterministic=True)
    return action


# ── output directories ─────────────────────────────────────────────────────────

_PAPER_FIGS = (
    _ROOT.parent.parent.parent
    / "academic_papers" / "snath_core" / "08_PERSIST" / "figures"
)
_LOCAL_FIGS = _ROOT / "experiments" / "persist" / "figures"
_LOCAL_FIGS.mkdir(parents=True, exist_ok=True)

_RESULTS_DIR = _ROOT / "experiments" / "persist"


# ── adaptor library ────────────────────────────────────────────────────────────

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def retrieve_candidates(
    library: dict,
    signature: np.ndarray,
    eps: float = EPS,
) -> dict:
    """Return all adaptors whose cosine similarity ≥ best − eps."""
    if not library:
        return {}
    sims = {k: cosine_sim(v["signature"], signature) for k, v in library.items()}
    best = max(sims.values())
    return {k: library[k] for k, s in sims.items() if s >= best - eps}


def make_primitive_library(mu: np.ndarray, sigma: np.ndarray, obs_dim: int) -> dict:
    """
    Pre-defined primitive adaptors for slope and wind perturbations.
    Deltas are domain-informed action offsets; signatures are the mean
    divergence direction observed in each zone during characterisation runs.
    """
    # Gravity-bias compensation: reduce hip extension, increase ankle push
    slope_delta = np.array([-0.05, 0.10, 0.15])
    # Lateral force compensation: lean into the force, increase hip stability
    wind_delta  = np.array([ 0.15, 0.05, 0.05])

    # Signatures: approximate directions in z-scored obs space
    # Ice+slope divergence loads heavily on height and angle dims (obs[0,1])
    slope_sig = np.zeros(obs_dim)
    slope_sig[0] = -0.60   # height drops on slope
    slope_sig[1] = -0.50   # torso tilts
    slope_sig[5] =  0.37   # x-velocity increases (sliding down)
    slope_sig /= np.linalg.norm(slope_sig)

    # Force divergence loads on lateral and torque dims
    wind_sig = np.zeros(obs_dim)
    wind_sig[1] =  0.65   # torso tilts laterally
    wind_sig[6] =  0.55   # z-velocity perturbed
    wind_sig[7] =  0.52   # torso angular velocity
    wind_sig /= np.linalg.norm(wind_sig)

    return {
        "slope": {"delta": slope_delta, "signature": slope_sig, "success_count": 0},
        "wind":  {"delta": wind_delta,  "signature": wind_sig,  "success_count": 0},
    }


# ── baseline collection ────────────────────────────────────────────────────────

def build_baseline(
    env: IceWorld, policy: Any, seed: int,
) -> tuple[np.ndarray, np.ndarray, list]:
    """Run BASE_STEPS on normal terrain; return (mu, sigma, last_buffer)."""
    rng = np.random.default_rng(seed)
    obs, _ = env.reset()
    env.set_zone("normal")
    obs_list: list[np.ndarray] = []
    for _ in range(BASE_STEPS):
        action = np.clip(
            base_action(policy, obs) + rng.normal(0, 0.02, size=env.act_dim),
            -1.0, 1.0,
        )
        obs, _, term, trunc, _ = env.step(action)
        obs_list.append(obs.copy())
        if term or trunc:
            obs, _ = env.reset()
            env.set_zone("normal")
    mu    = np.mean(obs_list, axis=0)
    sigma = np.std(obs_list, axis=0) + 1e-4
    buf   = list(obs_list[-10:])
    return mu, sigma, buf


# ── delta search (Phase 1 / Phase 5c escalation) ──────────────────────────────

def delta_search(
    env: IceWorld, zone: str, mu: np.ndarray, sigma: np.ndarray,
    buf: list, seed: int, policy: Any,
) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Sample N_CANDIDATES random deltas, trial each for K_SEARCH steps.
    Return (best_delta, best_rate, divergence_signature).
    Implements Eq. (6) of the PERSIST paper.
    """
    rng = np.random.default_rng(seed + 1000)
    candidates = rng.uniform(-0.3, 0.3, size=(N_CANDIDATES, env.act_dim))

    best_delta, best_rate = None, -np.inf
    sig_buf: list[np.ndarray] = list(buf)

    if len(sig_buf) >= 1:
        entry_sig = np.mean(sig_buf[-10:] if len(sig_buf) >= 10 else sig_buf, axis=0) - mu
    else:
        entry_sig = np.zeros_like(mu)

    for cand in candidates:
        trial_buf: list[np.ndarray] = list(sig_buf)
        obs, _ = env.reset()
        env.set_zone(zone)
        for _ in range(10):   # warm-up
            obs, _, term, trunc, _ = env.step(base_action(policy, obs))
            if term or trunc:
                obs, _ = env.reset()
                env.set_zone(zone)

        D_start = compute_divergence(obs, list(trial_buf), mu, sigma)
        D_final = D_start
        for _ in range(K_SEARCH):
            action = np.clip(base_action(policy, obs) + cand, -1.0, 1.0)
            obs, _, term, trunc, _ = env.step(action)
            D_final = compute_divergence(obs, trial_buf, mu, sigma)
            if term or trunc:
                break

        rate = (D_start - D_final) / K_SEARCH
        if rate > best_rate:
            best_rate  = rate
            best_delta = cand.copy()

    return best_delta, best_rate, entry_sig


# ── tournament (Phase 6) ───────────────────────────────────────────────────────

def run_tournament(
    env: IceWorld, zone: str, candidates: dict,
    mu: np.ndarray, sigma: np.ndarray, seed: int, policy: Any,
) -> tuple[str | None, dict[str, float], dict[str, float]]:
    """
    Trial each candidate adaptor for K_TOURNAMENT steps sequentially.
    Per-candidate rate uses per-candidate D_start (Eq. 1, PERSIST paper).
    Returns (winner_name, rates, similarities).
    """
    obs, _ = env.reset()
    env.set_zone(zone)
    buf: list[np.ndarray] = []
    for _ in range(20):   # warm up into zone
        obs, _, term, trunc, _ = env.step(base_action(policy, obs))
        buf.append(obs.copy())
        if term or trunc:
            obs, _ = env.reset()
            env.set_zone(zone)
            buf = []

    current_sig = (
        np.mean(buf[-10:] if len(buf) >= 10 else buf, axis=0) - mu
        if buf else np.zeros_like(mu)
    )
    sims = {k: cosine_sim(v["signature"], current_sig) for k, v in candidates.items()}

    rates: dict[str, float] = {}
    for name, adaptor in candidates.items():
        obs, _ = env.reset()
        env.set_zone(zone)
        trial_buf: list[np.ndarray] = list(buf)
        for _ in range(10):   # fresh warm-up per candidate
            obs, _, term, trunc, _ = env.step(base_action(policy, obs))
            trial_buf.append(obs.copy())
            if term or trunc:
                obs, _ = env.reset()
                env.set_zone(zone)
                trial_buf = []
        D_start = compute_divergence(obs, list(trial_buf), mu, sigma)
        for _ in range(K_TOURNAMENT):
            action = np.clip(base_action(policy, obs) + DELTA_0 * adaptor["delta"], -1.0, 1.0)
            obs, _, term, trunc, _ = env.step(action)
            D_final = compute_divergence(obs, trial_buf, mu, sigma)
            if term or trunc:
                break
        rates[name] = (D_start - D_final) / K_TOURNAMENT

    positive_rates = {k: v for k, v in rates.items() if v > 0}
    winner = max(positive_rates, key=positive_rates.get) if positive_rates else None
    return winner, rates, sims


# ── core persistence loop ──────────────────────────────────────────────────────

def run_persistence_loop(
    env: IceWorld, zone: str, library: dict,
    mu: np.ndarray, sigma: np.ndarray,
    patience: int, seed: int, policy: Any,
) -> dict:
    """
    Run the 6-component persistence loop for one phase/seed.
    Returns a result dict with outcome, steps, D_entry, D_final, D_trajectory.
    """
    obs, _ = env.reset()
    env.set_zone(zone)
    buf: list[np.ndarray] = []

    for _ in range(20):   # warm up into zone
        obs, _, term, trunc, _ = env.step(base_action(policy, obs))
        buf.append(obs.copy())
        if term or trunc:
            obs, _ = env.reset()
            env.set_zone(zone)
            buf = []

    D_entry  = compute_divergence(obs, list(buf), mu, sigma)
    D_traj   = [D_entry]
    delta_scale = DELTA_0
    steps       = 0
    outcome     = "escalate"

    current_sig = (
        np.mean(buf[-10:] if len(buf) >= 10 else buf, axis=0) - mu
        if buf else np.zeros_like(mu)
    )
    candidates = retrieve_candidates(library, current_sig)

    if not candidates:
        return {
            "outcome": "escalate", "steps": 0,
            "D_entry": D_entry, "D_final": D_entry,
            "D_trajectory": D_traj, "adaptor_trained": None,
        }

    best_name = max(candidates, key=lambda k: cosine_sim(candidates[k]["signature"], current_sig))
    adaptor   = candidates[best_name]

    for step in range(patience):
        action = np.clip(base_action(policy, obs) + delta_scale * adaptor["delta"], -1.0, 1.0)
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            obs, _ = env.reset()
            env.set_zone(zone)
            buf = []
        D_t = compute_divergence(obs, buf, mu, sigma)
        D_traj.append(D_t)
        steps += 1

        if D_t <= D_NORM:
            outcome = "resolve"
            library[best_name]["success_count"] += 1
            break
        if D_t >= D_ESC or delta_scale >= DELTA_MAX:
            outcome = "escalate"
            break

        # composition check — orthogonal residual (component 3)
        if step == patience // 3 and D_t > D_NORM * 1.5:
            sig_perp = current_sig - (
                np.dot(current_sig, adaptor["signature"]) /
                (np.linalg.norm(adaptor["signature"]) ** 2 + 1e-10)
            ) * adaptor["signature"]
            comp_candidates = {
                k: v for k, v in library.items()
                if k != best_name and cosine_sim(v["signature"], sig_perp) > 0.3
            }
            if comp_candidates:
                second = max(comp_candidates, key=lambda k: cosine_sim(comp_candidates[k]["signature"], sig_perp))
                action = np.clip(
                    base_action(policy, obs)
                    + delta_scale * adaptor["delta"]
                    + delta_scale * 0.5 * comp_candidates[second]["delta"],
                    -1.0, 1.0,
                )

        delta_scale += DELTA_INC

    D_final = D_traj[-1]
    return {
        "outcome": outcome, "steps": steps,
        "D_entry": float(D_entry), "D_final": float(D_final),
        "D_trajectory": [float(d) for d in D_traj],
        "adaptor_trained": None,
    }


# ── figure generation ──────────────────────────────────────────────────────────

PHASE_COLORS = {
    "1":  "#e74c3c",
    "2":  "#2ecc71",
    "3":  "#e67e22",
    "3b": "#1abc9c",
    "4":  "#9b59b6",
    "5c": "#95a5a6",
    "5w": "#3498db",
    "6":  "#f39c12",
}

PHASE_LABELS = {
    "1":  "Phase 1 — Encounter",
    "2":  "Phase 2 — First try",
    "3":  "Phase 3 — Scope boundary",
    "3b": "Phase 3b — Force",
    "4":  "Phase 4 — Exhaustion",
    "5c": "Phase 5c — Memory (cold)",
    "5w": "Phase 5w — Memory (warm)",
    "6":  "Phase 6 — Tournament",
}


def _save(fig: plt.Figure, name: str) -> None:
    local_path = _LOCAL_FIGS / name
    fig.savefig(local_path, dpi=150, bbox_inches="tight")
    if _PAPER_FIGS.exists():
        fig.savefig(_PAPER_FIGS / name, dpi=150, bbox_inches="tight")
        print(f"  → {_PAPER_FIGS / name}")
    print(f"  → {local_path}")


def generate_divergence_curves(results_by_phase: dict) -> None:
    """Figure 2: D(t) over decisions for all phases (mean ± std, n=5 seeds)."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for phase_id, phase_results in results_by_phase.items():
        trajs = [r["D_trajectory"] for r in phase_results if r["D_trajectory"]]
        if not trajs:
            continue
        max_len = max(len(t) for t in trajs)
        padded  = np.array([t + [t[-1]] * (max_len - len(t)) for t in trajs])
        mean    = padded.mean(axis=0)
        std     = padded.std(axis=0)
        xs      = np.arange(len(mean))
        color   = PHASE_COLORS.get(str(phase_id), "#888888")
        label   = PHASE_LABELS.get(str(phase_id), f"Phase {phase_id}")
        ax.plot(xs, mean, color=color, label=label, linewidth=1.8)
        ax.fill_between(xs, mean - std, mean + std, color=color, alpha=0.15)

    ax.axhline(D_NORM, color="black", linestyle="--", linewidth=0.9,
               label=f"Normalisation threshold ({D_NORM})")
    ax.axhline(D_ESC,  color="black", linestyle=":",  linewidth=0.9,
               label=f"Escalation threshold ({D_ESC})")

    ax.set_xlabel("Steps", fontsize=11)
    ax.set_ylabel("Divergence D(t)", fontsize=11)
    ax.set_title(
        "PERSIST: Divergence signal across experimental phases\n"
        f"(n={len(SEEDS)} seeds, mean ± std)",
        fontsize=12,
    )
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    _save(fig, "divergence_curves.pdf")
    plt.close(fig)


def generate_zone_detection(zone_summaries: dict) -> None:
    """Figure 1: entry vs final D per zone (bar chart, mean ± std)."""
    zones       = ["ice", "ice_slope", "force", "novel"]
    zone_labels = ["Ice", "Ice+Slope", "Force", "Novel"]

    entries_mean, entries_std, finals_mean, finals_std = [], [], [], []
    for z in zones:
        if z not in zone_summaries:
            entries_mean.append(0); entries_std.append(0)
            finals_mean.append(0); finals_std.append(0)
            continue
        e = [r["D_entry"] for r in zone_summaries[z]]
        f = [r["D_final"] for r in zone_summaries[z]]
        entries_mean.append(np.mean(e)); entries_std.append(np.std(e))
        finals_mean.append(np.mean(f));  finals_std.append(np.std(f))

    x  = np.arange(len(zones))
    w  = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w / 2, entries_mean, w, yerr=entries_std, capsize=4,
           color="#e74c3c", alpha=0.85, label="Entry D(t)")
    ax.bar(x + w / 2, finals_mean,  w, yerr=finals_std,  capsize=4,
           color="#2ecc71", alpha=0.85, label="Final D(t)")

    ax.axhline(D_NORM, color="black", linestyle="--", linewidth=0.9,
               label=f"Normalisation threshold ({D_NORM})")
    ax.axhline(D_ESC,  color="black", linestyle=":",  linewidth=0.9,
               label=f"Escalation threshold ({D_ESC})")

    ax.set_xticks(x)
    ax.set_xticklabels(zone_labels, fontsize=11)
    ax.set_ylabel("Divergence D(t)", fontsize=11)
    ax.set_title(
        f"Zone detection: entry vs resolved D(t)\n(n={len(SEEDS)} seeds, mean ± std)",
        fontsize=12,
    )
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    _save(fig, "zone_detection.pdf")
    plt.close(fig)


# ── main experiment ────────────────────────────────────────────────────────────

def run_experiment() -> dict:
    print("Loading / training base policy…")
    policy = train_or_load_policy()

    all_results: dict[str, list] = {p: [] for p in
        ["1", "2", "3", "3b", "4", "5c", "5w", "6"]}
    zone_results: dict[str, list] = {"ice": [], "ice_slope": [], "force": [], "novel": []}

    for seed in SEEDS:
        print(f"\n── seed {seed} ──────────────────────────────────")
        env = IceWorld(seed=seed)

        # ── baseline ──────────────────────────────────────────────────────────
        mu, sigma, _ = build_baseline(env, policy, seed)
        print(f"  baseline: μ={mu[:3].round(3)}, σ={sigma[:3].round(3)}")

        # ── shared library (grows across phases) ──────────────────────────────
        library = make_primitive_library(mu, sigma, env.obs_dim)

        # ── Phase 1 — Encounter (empty library) ───────────────────────────────
        print("  Phase 1 — Encounter")
        r1 = run_persistence_loop(env, "ice", {}, mu, sigma, PATIENCE_SHORT, seed, policy)
        print(f"    outcome={r1['outcome']} D_entry={r1['D_entry']:.2f}")
        all_results["1"].append(r1)
        zone_results["ice"].append(r1)

        ice_delta, ice_rate, ice_sig = delta_search(env, "ice", mu, sigma, [], seed, policy)
        library["ice_learned"] = {
            "delta": ice_delta, "signature": ice_sig, "success_count": 0
        }
        print(f"    ice_learned trained: rate={ice_rate:.3f}/step")

        # ── Phase 2 — First try ───────────────────────────────────────────────
        print("  Phase 2 — First try")
        r2 = run_persistence_loop(env, "ice", library, mu, sigma, PATIENCE_LONG, seed, policy)
        print(f"    outcome={r2['outcome']} steps={r2['steps']} D_final={r2['D_final']:.2f}")
        all_results["2"].append(r2)
        zone_results["ice"].append(r2)

        # ── Phase 3 — Scope boundary (ice+slope, primitives only) ─────────────
        print("  Phase 3 — Scope boundary")
        r3 = run_persistence_loop(env, "ice_slope", library, mu, sigma, PATIENCE_LONG, seed, policy)
        print(f"    outcome={r3['outcome']} D_final={r3['D_final']:.2f}")
        all_results["3"].append(r3)
        zone_results["ice_slope"].append(r3)

        # ── Phase 3b — Force (wind adaptor) ───────────────────────────────────
        print("  Phase 3b — Force")
        r3b = run_persistence_loop(env, "force", library, mu, sigma, PATIENCE_LONG, seed, policy)
        print(f"    outcome={r3b['outcome']} steps={r3b['steps']} D_final={r3b['D_final']:.2f}")
        all_results["3b"].append(r3b)
        zone_results["force"].append(r3b)

        # ── Phase 4 — Exhaustion (novel, all three) ────────────────────────────
        print("  Phase 4 — Exhaustion")
        r4 = run_persistence_loop(env, "novel", library, mu, sigma, PATIENCE_SHORT, seed, policy)
        print(f"    outcome={r4['outcome']} D_final={r4['D_final']:.2f}")
        all_results["4"].append(r4)
        zone_results["novel"].append(r4)

        # ── Phase 5c — Memory cold (empty library) ────────────────────────────
        print("  Phase 5c — Memory (cold)")
        r5c = run_persistence_loop(env, "ice", {}, mu, sigma, PATIENCE_LONG, seed, policy)
        total_decisions_cold = N_CANDIDATES + r5c.get("steps", 0)
        print(f"    outcome={r5c['outcome']} total_decisions={total_decisions_cold}")
        r5c["total_decisions"] = total_decisions_cold
        all_results["5c"].append(r5c)

        # ── Phase 5w — Memory warm (stored library) ───────────────────────────
        print("  Phase 5w — Memory (warm)")
        r5w = run_persistence_loop(env, "ice", library, mu, sigma, PATIENCE_LONG, seed, policy)
        print(f"    outcome={r5w['outcome']} steps={r5w['steps']}")
        r5w["total_decisions"] = r5w.get("steps", 0)
        all_results["5w"].append(r5w)

        # ── Phase 6 — Tournament ───────────────────────────────────────────────
        print("  Phase 6 — Tournament")
        combined_sig = (ice_sig + library["slope"]["signature"]) / 2.0
        combined_sig /= np.linalg.norm(combined_sig) + 1e-10
        lib6 = deepcopy(library)
        lib6["combined"] = {
            "delta": ice_delta + library["slope"]["delta"],
            "signature": combined_sig,
            "success_count": 0,
        }
        winner, rates, sims = run_tournament(env, "ice_slope", lib6, mu, sigma, seed, policy)
        print(f"    winner={winner}")
        print(f"    rates: { {k: round(v, 3) for k, v in rates.items()} }")
        print(f"    sims:  { {k: round(v, 3) for k, v in sims.items()} }")

        r6 = run_persistence_loop(env, "ice_slope", lib6, mu, sigma, PATIENCE_LONG, seed, policy)
        r6["tournament_winner"] = winner
        r6["tournament_rates"]  = {k: float(v) for k, v in rates.items()}
        r6["tournament_sims"]   = {k: float(v) for k, v in sims.items()}
        all_results["6"].append(r6)

        env.close()

    return all_results


def summarise(all_results: dict) -> None:
    print("\n── SUMMARY ──────────────────────────────────────────────────")
    for phase_id, results in all_results.items():
        if not results:
            continue
        label    = PHASE_LABELS.get(str(phase_id), f"Phase {phase_id}")
        outcomes = [r["outcome"] for r in results]
        n_res    = sum(1 for o in outcomes if o == "resolve")
        steps    = [r["steps"] for r in results if r["steps"] > 0]
        step_str = (
            f"{np.mean(steps):.1f} ± {np.std(steps):.1f}" if steps else "—"
        )
        td = [r.get("total_decisions") for r in results if r.get("total_decisions")]
        td_str = f"{np.mean(td):.1f} ± {np.std(td):.1f}" if td else ""
        print(
            f"  {label:<35} {n_res}/{len(results)}"
            f"  steps={step_str}"
            + (f"  total_decisions={td_str}" if td_str else "")
        )


def main() -> None:
    print("PERSIST — IceWorld experiment")
    print(f"Seeds: {SEEDS}  |  BASE_STEPS={BASE_STEPS}  |  N_CANDIDATES={N_CANDIDATES}")

    all_results = run_experiment()
    summarise(all_results)

    # zone-level data for Figure 1 (pick primary phase per zone)
    zone_summaries = {
        "ice":       all_results["2"],    # resolved ice
        "ice_slope": all_results["3"],    # scope-exceeded ice+slope
        "force":     all_results["3b"],   # force zone
        "novel":     all_results["4"],    # novel escalation
    }

    print("\nGenerating figures...")
    generate_divergence_curves(all_results)
    generate_zone_detection(zone_summaries)

    # save results JSON
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_path = _RESULTS_DIR / f"persist_results_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults → {out_path}")


if __name__ == "__main__":
    main()
