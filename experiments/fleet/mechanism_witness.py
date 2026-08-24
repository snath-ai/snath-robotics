"""
Mechanism-Shape Witness — a transferred causal fingerprint, verified not obeyed.
====================================================================================
Motivating idea (session discussion, 2026-08-24): a child who touches a hot
stove learns from direct experience. A child who is TOLD "this steel pot is
hot" learns the same lesson from a transferred, reasoned description --
without ever touching it -- because "hot" carries a causal claim, not just
a label. The claim isn't obeyed blindly; the child still generalises and
still gets burned exactly once if the claim is wrong. That is the shape of
a genuinely independent witness: information is transferred, but trusted
only insofar as it is corroborated by the receiver's own senses.

FLEET's existing D-hard event ships a LABEL (a 19-number concept vector).
It carries no causal content, which is exactly why wind and ice can collide
in concept space: both disrupt the gait similarly even though their
PHYSICAL MECHANISM is different (friction loss vs. an external lateral
force). This script tests whether the dynamics-residual guard's DRIFT
VECTOR SHAPE -- not just its scalar magnitude, which is all the existing
guard uses -- carries that mechanism distinction, and whether it survives
being transferred and checked by an independent robot.

Protocol:
  1. Robot A gets a guard too (previously only Robot B had one). While A
     logs its 30 ice D-hard events, it also records its own z-scored drift
     VECTOR at each event. The mean, unit-normalised vector across A's real
     ice events is the shipped "mechanism fingerprint" -- a genuinely new
     artifact riding alongside the 19-number event log, not a replacement
     for it. (This changes the "570 numbers, no raw data" framing if it
     becomes a paper result -- flagged here, not hidden.)
  2. Robot B calibrates a corroboration threshold from ITS OWN real ice
     phase: the cosine similarity between B's own live drift vector and
     A's shipped fingerprint, on windows where B genuinely experiences ice.
     The 5th percentile of that held-out, same-condition distribution is
     the threshold -- self-calibrated, same convention as the guard's own
     OOD threshold.
  3. On the wind-30N test, for every window that SURVIVES the existing
     guard-veto class-capture fix (i.e. would already be RECOGNISED_known
     under ood_guard.py's logic), additionally check: does B's own live
     drift vector's SHAPE actually corroborate A's ice fingerprint (cosine
     >= threshold), or does it look like something else? A contradiction
     downgrades the window to ALARM_NOVEL even though concept view AND the
     scalar guard both let it through. B never trusts A's fingerprint by
     itself -- it is only ever checked against B's own independently
     computed evidence.

This is a real, uncertain hypothesis, not a guaranteed result: if wind's
drift SHAPE turns out just as entangled with ice's as their concept
embeddings are, this witness will not help, and that is reported exactly
as plainly as every other finding in this series.

Run:
    .venv/bin/python experiments/fleet/mechanism_witness.py
"""
from __future__ import annotations

import sys, json, argparse, shutil
from pathlib import Path
from datetime import datetime
from statistics import mean, stdev

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dhard import DHardQueue, RoboticsDHardEvent
from experiments.fleet.fleet_n2_transfer import (
    load_base_encoder, consolidate, adapted_anchor,
    set_friction, FRICTION_NORMAL, FRICTION_ICE, DEFAULT_MODEL, RUNS_DIR, DETECT,
)
from experiments.fleet.ood_guard import GuardedRobot, WIND_LEVELS, GUARD_RESID_WIN

CALIBRATION_QUANTILE = 0.05   # held-out, same-condition (B's own real ice) low quantile
CALIB_EPISODES = 3            # independent held-out ice episodes pooled for calibration
CALIB_K = 2.0                 # threshold = mean(sims) - CALIB_K * std(sims)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class FingerprintRobot(GuardedRobot):
    """GuardedRobot + access to the full drift VECTOR (not just its scalar
    score) and a fingerprint-collecting variant of the ice-logging phase."""

    def guard_drift_vector(self):
        if len(self._resid_buf) < GUARD_RESID_WIN:
            return None
        drift = np.mean(self._resid_buf, axis=0)
        return (drift - self._d_mu) / self._d_sd

    def phase_with_fingerprint(self, friction: float, n_routed: int,
                               reset_seed: int | None = None,
                               queue: DHardQueue | None = None,
                               scenario: str = "", max_iters: int = 800) -> dict:
        """Identical event-logging semantics to Robot.phase(), plus
        collection of the z-scored drift vector on every D-hard window."""
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

        records, n_events, it = [], 0, 0
        drift_samples = []
        while len(records) < n_routed and it < max_iters:
            it += 1
            z_p = self.runner.push(obs)
            if z_p is None:
                obs = self._step(obs, friction)
                continue
            anchor_name, r = self.marouter.route(self.anchors, z_p)
            records.append({"decision": r.decision.value})
            if queue is not None and r.decision in DETECT:
                dv = self.guard_drift_vector()
                if dv is not None:
                    drift_samples.append(dv)
                queue.push(RoboticsDHardEvent(
                    z_vision=self.anchors[0][1].tolist(),
                    z_proprio=z_p.tolist(),
                    divergence=r.divergence,
                    decision=r.decision.value,
                    failure_class="environmental_transient",
                    scenario_id=scenario,
                    winner="proprio",
                ))
                n_events += 1
            obs = self._step(obs, friction)
        self._obs = obs

        fingerprint = None
        if drift_samples:
            fp = np.mean(drift_samples, axis=0)
            norm = np.linalg.norm(fp)
            fingerprint = (fp / norm).tolist() if norm > 1e-9 else fp.tolist()

        return {"n_routed": len(records), "n_events": n_events,
                "fingerprint": fingerprint, "fingerprint_n_samples": len(drift_samples)}

    def witnessed_phase(self, friction: float, n_routed: int,
                        fingerprint: np.ndarray, threshold: float,
                        reset_seed: int | None = None, wind: float = 0.0,
                        max_iters: int = 800, collect_calibration: bool = False) -> dict:
        """Concept routing + guard-veto (unchanged, = ood_guard.py's fix)
        + mechanism-shape corroboration layered on top. If
        collect_calibration, also returns the raw cosine similarities on
        genuine-anchor-matched windows for threshold calibration."""
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
        calib_sims = []
        while len(records) < n_routed and it < max_iters:
            it += 1
            z_p = self.runner.push(obs)
            if z_p is None:
                obs = self._step(obs, friction)
                continue
            anchor_name, r = self.marouter.route(self.anchors, z_p)
            g = self.guard_score()
            guard_ood = g > self._g_thresh
            concept_commit = (r.decision.value == "COMMIT_TRAJECTORY")
            novel = guard_ood and concept_commit          # existing guard-veto fix

            dv = self.guard_drift_vector()
            sim = _cos(dv, fingerprint) if (dv is not None) else None

            if novel:
                outcome = "ALARM_NOVEL"
            elif concept_commit and anchor_name == "normal":
                outcome = "COMMIT_normal"
            elif concept_commit and anchor_name != "normal":
                if collect_calibration and sim is not None:
                    calib_sims.append(sim)
                if sim is not None and sim < threshold:
                    outcome = "ALARM_NOVEL"                # witness contradicts
                else:
                    outcome = "RECOGNISED_known"
            else:
                outcome = "ALARM_concept"
            records.append({"outcome": outcome, "sim": sim})
            obs = self._step(obs, friction)
        self._obs = obs
        self._wind = 0.0

        n = max(len(records), 1)
        frac = lambda o: sum(r["outcome"] == o for r in records) / n
        return {
            "novel_alarm": frac("ALARM_NOVEL"),
            "recognised": frac("RECOGNISED_known"),
            "commit_normal": frac("COMMIT_normal"),
            "concept_alarm": frac("ALARM_concept"),
            "n_routed": len(records),
            "calib_sims": calib_sims,
        }


def run_seed(base_encoder, seed: int, n_routed: int) -> dict:
    run_dir = RUNS_DIR / f"witness_seed_{seed}"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    out = {"seed": seed}

    # Robot A: now a guarded robot too, ships a mechanism fingerprint
    # alongside the 19-number events (real, new artifact -- see docstring).
    A = FingerprintRobot("A", base_encoder, seed)
    A.warmup()
    A.fit_guard()
    queue_a = DHardQueue(str(run_dir / "robot_a_dhard.jsonl"))
    a_ice = A.phase_with_fingerprint(FRICTION_ICE, n_routed, reset_seed=seed + 1,
                                     queue=queue_a, scenario=f"witness_a_ice_s{seed}")
    A.close()
    out["events_shipped"] = a_ice["n_events"]
    out["fingerprint_n_samples"] = a_ice["fingerprint_n_samples"]
    if a_ice["fingerprint"] is None:
        out["adapter_transferred"] = False
        return out
    fingerprint = np.array(a_ice["fingerprint"])

    received = run_dir / "robot_b_received_dhard.jsonl"
    shutil.copy(queue_a.path, received)
    adapter_b = consolidate(received, run_dir / "adapters_b", seed)
    if adapter_b is None:
        out["adapter_transferred"] = False
        return out
    out["adapter_transferred"] = True

    B = FingerprintRobot("B-witness", base_encoder, seed + 1000)
    B.warmup()
    B.fit_guard()
    B.anchors.append(("ice_learned", adapted_anchor(B.anchors[0][1], adapter_b)))

    # Calibration: B's OWN real ice, pooled across CALIB_EPISODES independent
    # held-out episodes (not the single 30-window budget of one phase — a
    # 5th-percentile order statistic on ~30 samples proved too noisy at
    # scale, see STATUS.md). Threshold is parametric (mean - k*std), more
    # stable at small-to-moderate sample sizes than an empirical low quantile.
    sims = []
    for ep in range(CALIB_EPISODES):
        calib = B.witnessed_phase(FRICTION_ICE, n_routed, fingerprint, threshold=-1.0,
                                  reset_seed=seed + 2001 + ep, collect_calibration=True)
        sims.extend(calib["calib_sims"])
    if len(sims) >= 2:
        threshold = float(np.mean(sims) - CALIB_K * np.std(sims, ddof=1))
    elif len(sims) == 1:
        threshold = sims[0] - 0.5
    else:
        threshold = -1.0
    threshold = max(-1.0, min(1.0, threshold))
    out["calibration_threshold"] = threshold
    out["calibration_n_sims"] = len(sims)

    # Re-run ice cleanly at the calibrated threshold (fresh reset — the
    # calibration pass above already consumed this phase's randomness).
    out["ice"] = B.witnessed_phase(FRICTION_ICE, n_routed, fingerprint, threshold,
                                   reset_seed=seed + 1002)
    out["normal"] = B.witnessed_phase(FRICTION_NORMAL, n_routed, fingerprint, threshold,
                                      reset_seed=seed + 1004)
    for i, w in enumerate(WIND_LEVELS):
        out[f"wind{int(w)}"] = B.witnessed_phase(
            FRICTION_NORMAL, n_routed, fingerprint, threshold,
            reset_seed=seed + 1010 + i, wind=w)
    B.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seeds", type=str, default="42,7,13,99,2026")
    ap.add_argument("--model", type=str, default=str(DEFAULT_MODEL))
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    print("\n" + "=" * 74)
    print("  Snath Robotics — MECHANISM-SHAPE WITNESS (transferred fingerprint,")
    print("  verified not obeyed, layered on the guard-veto fix)")
    print("=" * 74)

    base = load_base_encoder(Path(args.model))
    results = []
    for seed in seeds:
        print(f"\n  seed {seed} ...", flush=True)
        r = run_seed(base, seed, args.steps)
        results.append(r)
        if not r.get("adapter_transferred"):
            print("    x no adapter transferred"); continue
        print(f"    calibration: threshold={r['calibration_threshold']:.3f} "
              f"from {r['calibration_n_sims']} held-out real-ice sims")
        phases = ["normal", "ice"] + [f"wind{int(w)}" for w in WIND_LEVELS]
        for ph in phases:
            p = r[ph]
            print(f"    {ph:<7}: novel={p['novel_alarm']:.0%} "
                  f"recognised={p['recognised']:.0%} "
                  f"commit={p['commit_normal']:.0%} "
                  f"concept-alarm={p['concept_alarm']:.0%}")

    ok = [r for r in results if r.get("adapter_transferred")]
    print("\n" + "=" * 74)
    print(f"  AGGREGATE over {len(ok)}/{len(seeds)} seeds")
    print("=" * 74)

    def agg(fn, label, spec=".1f"):
        vals = [fn(r) for r in ok]
        sd = stdev(vals) if len(vals) > 1 else 0.0
        print(f"  {label:<44} {mean(vals):{spec}} +/- {sd:{spec}}")
        return vals

    agg(lambda r: r["normal"]["novel_alarm"] * 100, "normal novel-alarm (%)")
    agg(lambda r: r["ice"]["novel_alarm"] * 100, "ice novel-alarm (%)")
    agg(lambda r: r["ice"]["recognised"] * 100, "ice recognised (%)")
    agg(lambda r: r["wind30"]["recognised"] * 100, "wind30 recognised-as-ice (%)")
    agg(lambda r: r["calibration_threshold"], "calibration threshold", ".3f")

    print("=" * 74 + "\n")

    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    suffix = f"_{args.tag}" if args.tag else ""
    out_path = Path(__file__).parent / f"mechanism_witness_results_{stamp}{suffix}.json"
    with open(out_path, "w") as f:
        json.dump({
            "protocol": {"calibration_quantile": CALIBRATION_QUANTILE,
                         "routed_windows_per_phase": args.steps, "seeds": seeds,
                         "encoder": Path(args.model).name,
                         "note": "mechanism fingerprint = mean unit-normalised "
                                 "z-scored drift vector across A's real ice "
                                 "D-hard events; corroboration threshold = "
                                 "5th pct of cosine-sim on B's own held-out "
                                 "real ice phase; layered on the guard-veto fix"},
            "results": results,
        }, f, indent=2)
    print(f"  Results -> {out_path.name}")


if __name__ == "__main__":
    main()
