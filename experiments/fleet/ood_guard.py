"""
OOD Guard Experiment — closing the confidence trap (DAS §8.2).
===============================================================
The fleet N=2 run (fleet_n2_transfer.py) validated recognition transfer
but failed its novelty control: lateral wind perturbing raw dynamics
2.2-2.4x MORE than ice reads as encoder D ~= 0 — the two-class concept
encoder projects unseen physics onto the 'normal' logit at conf ~0.999,
and the router COMMITs. DAS §8.2 anticipated exactly this and called for
a pre-routing OOD safety layer.

Three guard designs were tried before this one, each rejected with
evidence (see git history and the pilot diagnostics):
  1. Warmup-baseline z-scored window stats: normal-phase episode drift
     (6.13) indistinguishable from ice (5.69).
  2. kNN novelty over z-scaled window mean+std with held-out-calibrated
     threshold: a fresh normal episode drifts MORE across window stats
     (mean |z| 0.56) than wind (0.39) or ice (0.33) shifts them. No
     static function of marginal window statistics separates moderate
     unseen physics from episode-to-episode variability in this regime.
  3. One-step |residual| of a ridge dynamics model f(obs,act)->delta_obs:
     condition-independent (normal 3.28 / ice 3.29 / wind10 3.22) — the
     linear model is equally wrong about contact everywhere, and per-step
     force bias (~0.07 sigma/step at 10 N on a ~24 kg walker) drowns in
     contact noise.

This design keeps the dynamics model but scores its DRIFT: the signed
per-dim rolling mean of residuals, z-scored against held-out fresh-normal
drift statistics, max over dims. Persistent-force physics biases the
residual in a consistent direction (drift grows ~L; zero-mean contact
noise grows ~sqrt(L)). A new reset seed does not change the physics, so
the model stays valid on any normal episode. This makes PAV Definition 1
operational — "the policy's implicit physics model is no longer valid" —
and it is once more the divergence principle: stream A = predicted next
state, stream B = observed next state.

Decision logic (contradiction of two views):
  NOVEL alarm = guard says dynamics-OOD while the concept view matches
  the 'normal' anchor. Known non-normal classes stay recognised: on ice
  both views agree (guard OOD + concept matches 'ice_learned').

The seed-42 pilot showed detection is GRADED in perturbation magnitude,
so the experiment measures the sensitivity curve (wind 10/20/30/40 N)
rather than a single binary level. The original pre-registered criteria
are kept and reported honestly:
  G1 novelty detected : wind-10N novel-alarm rate >= 80% each seed
                        (expected FAIL from the pilot — 10 N sits in the
                        blind band of BOTH views; reported as such)
  G2 no false novelty : normal novel-alarm rate <= 5% each seed
  G3 recognition kept : ice novel-alarm <= 5% AND ice recognised >= 75%

Run:
    .venv/bin/python experiments/fleet/ood_guard.py
"""
from __future__ import annotations

import sys, json, argparse, shutil
from pathlib import Path
from datetime import datetime
from collections import deque
from statistics import mean, stdev

import numpy as np
import torch

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

from dhard import DHardQueue
from experiments.fleet.fleet_n2_transfer import (
    Robot, load_base_encoder, consolidate, adapted_anchor,
    set_friction, apply_wind,
    FRICTION_NORMAL, FRICTION_ICE, DEFAULT_MODEL, RUNS_DIR,
)

WIND_LEVELS = [10.0, 20.0, 30.0, 40.0]   # sensitivity sweep; 10N is the
                                          # original pre-registered novel level
GUARD_TRAIN_STEPS = 3000   # normal-terrain transitions for the dynamics fit
GUARD_STAT_STEPS  = 1200   # held-out: drift-vector mean/std estimation
GUARD_THR_STEPS   = 1200   # held-out: threshold quantile estimation
GUARD_RIDGE_LAM   = 1e-3
GUARD_RESID_WIN   = 20     # rolling SIGNED-residual (drift) window
GUARD_QUANTILE    = 0.99


class GuardedRobot(Robot):
    """Robot + one-step dynamics-residual OOD guard."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._resid_buf: deque = deque(maxlen=GUARD_RESID_WIN)
        self._last_transition = None
        self._guard_ready = False

    # override: record (obs, act, obs') transitions; invalidate across resets
    def _step(self, obs, friction):
        apply_wind(self.env, self._wind)
        act = self.policy(obs)
        self._last_transition = None
        nxt, _, term, trunc, _ = self.env.step(act)
        if term or trunc:
            nxt, _ = self.env.reset()
            set_friction(self.env, friction)
            self.runner.reset()
            self.policy.reset()
            self._resid_buf.clear()          # residuals invalid across a reset
        else:
            self._last_transition = (obs.copy(), np.asarray(act).copy(),
                                     nxt.copy())
            if self._guard_ready:
                self._resid_buf.append(self._step_residual(*self._last_transition))
        return nxt

    # ── dynamics model ────────────────────────────────────────────────────
    def _collect_transitions(self, n_steps: int):
        obs = self._obs
        X, Y = [], []
        for _ in range(n_steps):
            self.runner.push(obs)
            obs = self._step(obs, FRICTION_NORMAL)
            if self._last_transition is not None:
                o, a, nxt = self._last_transition
                X.append(np.concatenate([o, a]))
                Y.append(nxt - o)
        self._obs = obs
        return np.stack(X), np.stack(Y)

    def fit_guard(self):
        """
        Ridge one-step model on normal terrain. The score is the DRIFT of
        signed residuals — |rolling mean| per dim, z-scored, max over dims:
        persistent-force physics biases residuals in a consistent direction
        (drift grows ~L while zero-mean contact noise grows ~sqrt(L)).
        Drift statistics and the threshold both come from HELD-OUT fresh
        normal running, collected exactly as at runtime (same rolling
        buffer, cleared at every reset).
        """
        X, Y = self._collect_transitions(GUARD_TRAIN_STEPS)
        self._x_mu, self._x_sd = X.mean(0), X.std(0) + 1e-8
        Xz = (X - self._x_mu) / self._x_sd
        A = Xz.T @ Xz + GUARD_RIDGE_LAM * np.eye(Xz.shape[1])
        self._W = np.linalg.solve(A, Xz.T @ Y)          # (23, 17)

        # held-out pass 1: drift-vector mean/std per dim
        self._guard_ready = True
        self._resid_buf.clear()
        drifts = self._collect_drift_vectors(GUARD_STAT_STEPS)
        self._d_mu = drifts.mean(0)
        self._d_sd = drifts.std(0) + 1e-9

        # held-out pass 2: threshold = q-quantile of runtime-form scores
        scores = [self._drift_score(d) for d in
                  self._collect_drift_vectors(GUARD_THR_STEPS)]
        self._g_thresh = float(np.quantile(scores, GUARD_QUANTILE))
        self._resid_buf.clear()

    def _collect_drift_vectors(self, n_steps: int) -> np.ndarray:
        """Run on normal terrain; emit rolling drift vectors as at runtime."""
        obs = self._obs
        out = []
        for _ in range(n_steps):
            self.runner.push(obs)
            obs = self._step(obs, FRICTION_NORMAL)
            if len(self._resid_buf) == GUARD_RESID_WIN:
                out.append(np.mean(self._resid_buf, axis=0))
        self._obs = obs
        return np.stack(out)

    def _step_residual(self, o, a, nxt) -> np.ndarray:
        """SIGNED per-dim residual of the one-step dynamics model."""
        x = (np.concatenate([o, a]) - self._x_mu) / self._x_sd
        return (nxt - o) - x @ self._W

    def _drift_score(self, drift: np.ndarray) -> float:
        return float((np.abs(drift - self._d_mu) / self._d_sd).max())

    def guard_score(self) -> float:
        if len(self._resid_buf) < GUARD_RESID_WIN:
            return 0.0                        # not enough evidence yet
        return self._drift_score(np.mean(self._resid_buf, axis=0))

    # ── guarded routing ───────────────────────────────────────────────────
    def guarded_phase(self, friction: float, n_routed: int,
                      reset_seed: int | None = None,
                      wind: float = 0.0,
                      max_iters: int = 800) -> dict:
        """Concept routing + guard; NOVEL = guard OOD while concept says normal."""
        self._wind = wind
        obs = self._obs
        if reset_seed is not None:
            obs, _ = self.env.reset(seed=reset_seed)
            set_friction(self.env, friction)
            self.runner.reset()
            self.policy.reset()
            self._resid_buf.clear()
            for _ in range(self.encoder.seq_len - 1):
                self.runner.push(obs)
                obs = self._step(obs, friction)

        records, it = [], 0
        while len(records) < n_routed and it < max_iters:
            it += 1
            z_p = self.runner.push(obs)
            if z_p is None:
                obs = self._step(obs, friction)
                continue
            anchor_name, r = self.marouter.route(self.anchors, z_p)
            g = self.guard_score()
            guard_ood = g > self._g_thresh
            concept_normal = (r.decision.value == "COMMIT_TRAJECTORY"
                              and anchor_name == "normal")
            novel = guard_ood and concept_normal
            if novel:
                outcome = "ALARM_NOVEL"
            elif r.decision.value == "COMMIT_TRAJECTORY":
                outcome = "COMMIT_normal" if anchor_name == "normal" \
                    else "RECOGNISED_known"
            else:
                outcome = "ALARM_concept"
            records.append({"iter": it, "D": r.divergence, "anchor": anchor_name,
                            "decision": r.decision.value, "guard": g,
                            "guard_ood": guard_ood, "outcome": outcome})
            obs = self._step(obs, friction)
        self._obs = obs
        self._wind = 0.0

        n = max(len(records), 1)
        frac = lambda o: sum(r["outcome"] == o for r in records) / n
        return {
            "mean_D": sum(r["D"] for r in records) / n,
            "mean_guard": sum(r["guard"] for r in records) / n,
            "guard_ood_rate": sum(r["guard_ood"] for r in records) / n,
            "novel_alarm": frac("ALARM_NOVEL"),
            "recognised": frac("RECOGNISED_known"),
            "commit_normal": frac("COMMIT_normal"),
            "concept_alarm": frac("ALARM_concept"),
            "n_routed": len(records),
            "records": records,
        }


def run_seed(base_encoder, seed: int, n_routed: int) -> dict:
    run_dir = RUNS_DIR / f"ood_seed_{seed}"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    out = {"seed": seed}

    # Robot A: real ice -> events -> shipped (identical to the fleet run)
    A = Robot("A", base_encoder, seed)
    A.warmup()
    queue_a = DHardQueue(str(run_dir / "robot_a_dhard.jsonl"))
    a_ice = A.phase(FRICTION_ICE, n_routed, reset_seed=seed + 1,
                    queue=queue_a, scenario=f"ood_a_ice_s{seed}")
    A.close()
    out["events_shipped"] = a_ice["n_events"]

    received = run_dir / "robot_b_received_dhard.jsonl"
    shutil.copy(queue_a.path, received)
    adapter_b = consolidate(received, run_dir / "adapters_b", seed)
    if adapter_b is None:
        out["adapter_transferred"] = False
        return out
    out["adapter_transferred"] = True

    # Robot B with guard: anchors {normal, ice_learned from A's events}
    B = GuardedRobot("B-guarded", base_encoder, seed + 1000)
    B.warmup()
    B.fit_guard()
    B.anchors.append(("ice_learned", adapted_anchor(B.anchors[0][1], adapter_b)))
    out["guard_threshold"] = B._g_thresh
    # every eval phase resets, so all conditions face the same regime
    out["normal"] = B.guarded_phase(FRICTION_NORMAL, n_routed,
                                    reset_seed=seed + 1004)
    out["ice"] = B.guarded_phase(FRICTION_ICE, n_routed,
                                 reset_seed=seed + 1001)
    for i, w in enumerate(WIND_LEVELS):
        out[f"wind{int(w)}"] = B.guarded_phase(
            FRICTION_NORMAL, n_routed, reset_seed=seed + 1010 + i, wind=w)
    B.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seeds", type=str, default="42,7,13,99,2026")
    ap.add_argument("--model", type=str, default=str(DEFAULT_MODEL))
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    print("\n" + "═" * 74)
    print("  Snath Robotics — OOD GUARD (dynamics-residual, two-view contradiction)")
    print("  novel = dynamics model invalid while concept view says normal")
    print("═" * 74)

    base = load_base_encoder(Path(args.model))
    results = []
    for seed in seeds:
        print(f"\n  seed {seed} …", flush=True)
        r = run_seed(base, seed, args.steps)
        results.append(r)
        if not r.get("adapter_transferred"):
            print("    ✗ no adapter transferred"); continue
        phases = ["normal", "ice"] + [f"wind{int(w)}" for w in WIND_LEVELS]
        for ph in phases:
            p = r[ph]
            print(f"    {ph:<7}: novel={p['novel_alarm']:.0%} "
                  f"recognised={p['recognised']:.0%} "
                  f"commit={p['commit_normal']:.0%} "
                  f"concept-alarm={p['concept_alarm']:.0%} │ "
                  f"guard={p['mean_guard']:.2f} (thr={r['guard_threshold']:.2f}) "
                  f"encD={p['mean_D']:.3f}")

    ok = [r for r in results if r.get("adapter_transferred")]
    print("\n" + "═" * 74)
    print(f"  AGGREGATE over {len(ok)}/{len(seeds)} seeds")
    print("═" * 74)

    def agg(fn, label, spec=".1f"):
        vals = [fn(r) for r in ok]
        sd = stdev(vals) if len(vals) > 1 else 0.0
        print(f"  {label:<44} {mean(vals):{spec}} ± {sd:{spec}}")
        return vals

    nrm = agg(lambda r: r["normal"]["novel_alarm"] * 100, "normal novel-alarm (%)")
    icn = agg(lambda r: r["ice"]["novel_alarm"] * 100, "ice novel-alarm (%)")
    icr = agg(lambda r: r["ice"]["recognised"] * 100, "ice recognised (%)")
    agg(lambda r: r["guard_threshold"], "self-calibrated threshold", ".2f")

    print("\n  Sensitivity curve (novel-alarm % / guard-OOD % / encoder D):")
    curve = {}
    for w in WIND_LEVELS:
        k = f"wind{int(w)}"
        na = [r[k]["novel_alarm"] * 100 for r in ok]
        go = [r[k]["guard_ood_rate"] * 100 for r in ok]
        ed = [r[k]["mean_D"] for r in ok]
        curve[k] = {"novel_alarm": mean(na), "guard_ood": mean(go),
                    "enc_D": mean(ed)}
        print(f"    {k:<7}: novel={mean(na):5.1f}±{stdev(na) if len(na)>1 else 0:.1f}  "
              f"guard-OOD={mean(go):5.1f}  encD={mean(ed):.3f}")

    w10 = [r["wind10"]["novel_alarm"] * 100 for r in ok]
    g1 = all(v >= 80 for v in w10)
    g2 = all(v <= 5 for v in nrm)
    g3 = all(v <= 5 for v in icn) and all(v >= 75 for v in icr)
    print(f"\n  G1  novelty at original 10N level (novel ≥80%)  : "
          f"{'PASS ✓' if g1 else 'FAIL ✗'}")
    print(f"  G2  no false novelty (normal novel ≤5%)         : "
          f"{'PASS ✓' if g2 else 'FAIL ✗'}")
    print(f"  G3  recognition kept (ice novel ≤5%, recog ok)  : "
          f"{'PASS ✓' if g3 else 'FAIL ✗'}")
    print("═" * 74 + "\n")

    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    out_path = _ROOT / "experiments" / "fleet" / f"ood_guard_results_{stamp}.json"
    with open(out_path, "w") as f:
        json.dump({
            "protocol": {
                "wind_levels_N": WIND_LEVELS,
                "guard": {"type": "one-step dynamics DRIFT residual (ridge)",
                          "score": "max-dim z of signed rolling residual mean",
                          "train_steps": GUARD_TRAIN_STEPS,
                          "stat_steps": GUARD_STAT_STEPS,
                          "thr_steps": GUARD_THR_STEPS,
                          "ridge_lambda": GUARD_RIDGE_LAM,
                          "resid_window": GUARD_RESID_WIN,
                          "quantile": GUARD_QUANTILE,
                          "threshold": "held-out fresh-normal quantile"},
                "routed_windows_per_phase": args.steps, "seeds": seeds,
                "encoder": Path(args.model).name,
                "criteria": {
                    "G1": "wind10 novel-alarm >= 80% each seed (original level)",
                    "G2": "normal novel-alarm <= 5% each seed",
                    "G3": "ice novel-alarm <= 5% and recognised >= 75% each seed"},
            },
            "results": results,
            "sensitivity_curve": curve,
            "pass": {"G1": g1, "G2": g2, "G3": g3},
        }, f, indent=2)
    print(f"  Results → {out_path.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
