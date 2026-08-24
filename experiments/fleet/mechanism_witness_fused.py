"""
Fused Mechanism Witness — persistence stabilises the shape estimate, not a
second hard gate.
============================================================================
Session discussion, 2026-08-25: the persistence gate (verified_recognition.py)
and the mechanism-shape witness (mechanism_witness.py) were each tested
separately. Naively stacking them as two independent hard vetoes would
compound their false-negative costs -- a seed that already loses some real
ice recognition to persistence's K-consecutive-window requirement, and
separately loses some to the shape check, loses MORE under both together.

The fused version instead uses persistence to STABILISE the shape
measurement rather than gate it a second time: on consecutive windows
where the same anchor keeps matching, the drift vectors are averaged
(rolling, capped at FUSE_WINDOW) before the cosine-similarity check against
A's shipped fingerprint, rather than checking a single noisy window's shape
every time. Recognition can still happen on the very first matched window
(using whatever's accumulated so far -- one sample, same as before); it
just gets less noisy, not more delayed, as a streak continues. This is a
real, different design, not a guaranteed improvement -- if the noise in a
single drift-vector sample wasn't actually the dominant source of the
mechanism-witness's residual failure tail, this will not help, and that
will be reported exactly as plainly as everything else tonight.

Run:
    .venv/bin/python experiments/fleet/mechanism_witness_fused.py
"""
from __future__ import annotations

import sys, json, argparse, shutil
from pathlib import Path
from datetime import datetime
from collections import deque
from statistics import mean, stdev

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dhard import DHardQueue
from experiments.fleet.fleet_n2_transfer import (
    load_base_encoder, consolidate, adapted_anchor,
    set_friction, FRICTION_NORMAL, FRICTION_ICE, DEFAULT_MODEL, RUNS_DIR,
)
from experiments.fleet.ood_guard import WIND_LEVELS
from experiments.fleet.mechanism_witness import FingerprintRobot, _cos

CALIB_EPISODES = 3
CALIB_K = 2.0
FUSE_WINDOW = 5   # cap on how many consecutive matched-window drift
                  # vectors get averaged into the shape estimate


class FusedWitnessRobot(FingerprintRobot):

    def witnessed_phase_fused(self, friction: float, n_routed: int,
                              fingerprint: np.ndarray, threshold: float,
                              reset_seed: int | None = None, wind: float = 0.0,
                              max_iters: int = 800,
                              collect_calibration: bool = False) -> dict:
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
        streak_anchor = None
        streak_buf: deque = deque(maxlen=FUSE_WINDOW)
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
            novel = guard_ood and concept_commit            # unchanged guard-veto fix

            dv = self.guard_drift_vector()
            matched_now = concept_commit and anchor_name != "normal"
            if matched_now and dv is not None:
                if anchor_name == streak_anchor:
                    streak_buf.append(dv)
                else:
                    streak_anchor = anchor_name
                    streak_buf = deque([dv], maxlen=FUSE_WINDOW)
                avg_dv = np.mean(streak_buf, axis=0)
                sim = _cos(avg_dv, fingerprint)
            else:
                streak_anchor, streak_buf = None, deque(maxlen=FUSE_WINDOW)
                sim = None

            if novel:
                outcome = "ALARM_NOVEL"
            elif concept_commit and anchor_name == "normal":
                outcome = "COMMIT_normal"
            elif matched_now:
                if collect_calibration and sim is not None:
                    calib_sims.append(sim)
                if sim is not None and sim < threshold:
                    outcome = "ALARM_NOVEL"
                else:
                    outcome = "RECOGNISED_known"
            else:
                outcome = "ALARM_concept"
            records.append({"outcome": outcome, "sim": sim,
                            "streak_len": len(streak_buf)})
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
    run_dir = RUNS_DIR / f"fusedwitness_seed_{seed}"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    out = {"seed": seed}

    A = FingerprintRobot("A", base_encoder, seed)
    A.warmup()
    A.fit_guard()
    queue_a = DHardQueue(str(run_dir / "robot_a_dhard.jsonl"))
    a_ice = A.phase_with_fingerprint(FRICTION_ICE, n_routed, reset_seed=seed + 1,
                                     queue=queue_a, scenario=f"fusedwitness_a_ice_s{seed}")
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

    B = FusedWitnessRobot("B-fused-witness", base_encoder, seed + 1000)
    B.warmup()
    B.fit_guard()
    B.anchors.append(("ice_learned", adapted_anchor(B.anchors[0][1], adapter_b)))

    sims = []
    for ep in range(CALIB_EPISODES):
        calib = B.witnessed_phase_fused(FRICTION_ICE, n_routed, fingerprint, threshold=-1.0,
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

    out["ice"] = B.witnessed_phase_fused(FRICTION_ICE, n_routed, fingerprint, threshold,
                                         reset_seed=seed + 1002)
    out["normal"] = B.witnessed_phase_fused(FRICTION_NORMAL, n_routed, fingerprint, threshold,
                                            reset_seed=seed + 1004)
    for i, w in enumerate(WIND_LEVELS):
        out[f"wind{int(w)}"] = B.witnessed_phase_fused(
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
    print(f"  Snath Robotics — FUSED MECHANISM WITNESS (persistence stabilises")
    print(f"  the shape estimate, window={FUSE_WINDOW}, not a second hard gate)")
    print("=" * 74)

    base = load_base_encoder(Path(args.model))
    results = []
    for seed in seeds:
        r = run_seed(base, seed, args.steps)
        results.append(r)

    ok = [r for r in results if r.get("adapter_transferred")]
    print(f"\n  {len(ok)}/{len(seeds)} seeds transferred")

    def agg(fn, label, spec=".1f"):
        vals = [fn(r) for r in ok]
        sd = stdev(vals) if len(vals) > 1 else 0.0
        print(f"  {label:<44} {mean(vals):{spec}} +/- {sd:{spec}}")

    agg(lambda r: r["normal"]["novel_alarm"] * 100, "normal novel-alarm (%)")
    agg(lambda r: r["ice"]["novel_alarm"] * 100, "ice novel-alarm (%)")
    agg(lambda r: r["ice"]["recognised"] * 100, "ice recognised (%)")
    agg(lambda r: r["wind30"]["recognised"] * 100, "wind30 recognised-as-ice (%)")

    g3 = sum(1 for r in ok if r["ice"]["novel_alarm"]*100<=5 and r["ice"]["recognised"]*100>=75)
    g2 = sum(1 for r in ok if r["normal"]["novel_alarm"]*100<=5)
    print(f"  G3 pass: {g3}/{len(ok)} ({100*g3/len(ok):.1f}%)")
    print(f"  G2 pass: {g2}/{len(ok)} ({100*g2/len(ok):.1f}%)")
    print("=" * 74 + "\n")

    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    suffix = f"_{args.tag}" if args.tag else ""
    out_path = Path(__file__).parent / f"mechanism_witness_fused_results_{stamp}{suffix}.json"
    with open(out_path, "w") as f:
        json.dump({"protocol": {"fuse_window": FUSE_WINDOW,
                                "calibration_episodes": CALIB_EPISODES,
                                "calibration_k": CALIB_K,
                                "routed_windows_per_phase": args.steps, "seeds": seeds},
                   "results": results}, f, indent=2)
    print(f"  Results -> {out_path.name}")


if __name__ == "__main__":
    main()
