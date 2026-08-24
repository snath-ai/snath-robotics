"""
Verification-Gated Recognition — persistence gate on top of the OOD guard.
============================================================================
FLEET's Finding "Class capture" names the remedy: PERSIST's detect->verify
loop, which trials a candidate adaptor and demands divergence actually fall
before trusting a match, rather than accepting a single low-divergence
window at face value.

That mechanism does not port literally to this codebase: PERSIST trials an
adaptor by APPLYING it and measuring whether live divergence falls over the
trial. The fleet/OOD-guard anchors here are passive comparison vectors —
"ice_learned" is used only to compute a divergence score against the live
embedding, never applied to correct the robot's actual behaviour. There is
no physical action to trial the effect of, so a literal port has nothing to
attach to.

This script implements the honest, implementable analogue instead: a class
match is not trusted from a single routed window. It must PERSIST — the
SAME anchor must be the minimum-divergence match with D < tau_low across
VERIFY_K consecutive routed windows — before being finalised as
RECOGNISED_known. An unverified non-normal match is conservatively treated
as ALARM_NOVEL (not trusted) rather than silently granted.

This is a real, falsifiable hypothesis, not a guaranteed fix: class capture
was found to be near-instantaneous (encoder confidence ~0.999 on the very
window it occurs), so if the wind-onto-ice mapping is a SUSTAINED, systemic
confusion rather than a transient one-window coincidence, persistence will
not catch it, and that failure mode is reported exactly as plainly as every
other result in this series if it occurs.

Layered on top of (not instead of) the existing class-capture guard-veto
fix (ood_guard.py's GuardedRobot) — both mechanisms are active.

Run:
    .venv/bin/python experiments/fleet/verified_recognition.py
"""
from __future__ import annotations

import sys, json, argparse, shutil
from pathlib import Path
from datetime import datetime
from statistics import mean, stdev

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dhard import DHardQueue
from experiments.fleet.fleet_n2_transfer import (
    load_base_encoder, consolidate, adapted_anchor,
    set_friction, FRICTION_NORMAL, FRICTION_ICE, DEFAULT_MODEL, RUNS_DIR,
)
from experiments.fleet.ood_guard import GuardedRobot, WIND_LEVELS, Robot

VERIFY_K = 3   # consecutive routed windows a match must hold across


class VerifiedGuardedRobot(GuardedRobot):
    """GuardedRobot + persistence-gated recognition."""

    def guarded_phase(self, friction: float, n_routed: int,
                      reset_seed: int | None = None,
                      wind: float = 0.0,
                      max_iters: int = 800) -> dict:
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

        tau_low = self.marouter.router.tau_low
        streak_anchor, streak_len = None, 0
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
            concept_commit = (r.decision.value == "COMMIT_TRAJECTORY")

            # persistence streak: same anchor, COMMIT-eligible divergence,
            # across consecutive routed windows
            matched = concept_commit and r.divergence < tau_low
            if matched and anchor_name == streak_anchor:
                streak_len += 1
            elif matched:
                streak_anchor, streak_len = anchor_name, 1
            else:
                streak_anchor, streak_len = None, 0

            novel = guard_ood and concept_commit   # unchanged guard-veto fix
            verified = matched and streak_len >= VERIFY_K

            if novel:
                outcome = "ALARM_NOVEL"
            elif concept_commit and anchor_name == "normal":
                outcome = "COMMIT_normal"
            elif concept_commit and anchor_name != "normal" and verified:
                outcome = "RECOGNISED_known"
            elif concept_commit and anchor_name != "normal":
                outcome = "ALARM_NOVEL"          # unverified match: not trusted yet
            else:
                outcome = "ALARM_concept"
            records.append({"iter": it, "D": r.divergence, "anchor": anchor_name,
                            "decision": r.decision.value, "guard": g,
                            "guard_ood": guard_ood, "streak": streak_len,
                            "outcome": outcome})
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
    run_dir = RUNS_DIR / f"verified_seed_{seed}"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    out = {"seed": seed}

    A = Robot("A", base_encoder, seed)
    A.warmup()
    queue_a = DHardQueue(str(run_dir / "robot_a_dhard.jsonl"))
    a_ice = A.phase(FRICTION_ICE, n_routed, reset_seed=seed + 1,
                    queue=queue_a, scenario=f"verified_a_ice_s{seed}")
    A.close()
    out["events_shipped"] = a_ice["n_events"]

    received = run_dir / "robot_b_received_dhard.jsonl"
    shutil.copy(queue_a.path, received)
    adapter_b = consolidate(received, run_dir / "adapters_b", seed)
    if adapter_b is None:
        out["adapter_transferred"] = False
        return out
    out["adapter_transferred"] = True

    B = VerifiedGuardedRobot("B-verified", base_encoder, seed + 1000)
    B.warmup()
    B.fit_guard()
    B.anchors.append(("ice_learned", adapted_anchor(B.anchors[0][1], adapter_b)))
    out["guard_threshold"] = B._g_thresh
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

    print("\n" + "=" * 74)
    print(f"  Snath Robotics — VERIFIED RECOGNITION (persistence gate, K={VERIFY_K})")
    print("  recognised = same anchor matched (D<tau_low) K consecutive windows")
    print("=" * 74)

    base = load_base_encoder(Path(args.model))
    results = []
    for seed in seeds:
        print(f"\n  seed {seed} ...", flush=True)
        r = run_seed(base, seed, args.steps)
        results.append(r)
        if not r.get("adapter_transferred"):
            print("    x no adapter transferred"); continue
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

    print("\n  Wind sensitivity (recognised-as-known % / novel-alarm %):")
    curve = {}
    for w in WIND_LEVELS:
        k = f"wind{int(w)}"
        rec = [r[k]["recognised"] * 100 for r in ok]
        na = [r[k]["novel_alarm"] * 100 for r in ok]
        curve[k] = {"recognised": mean(rec), "novel_alarm": mean(na)}
        print(f"    {k:<7}: recognised={mean(rec):5.1f}"
              f"+/-{stdev(rec) if len(rec)>1 else 0:.1f}  "
              f"novel={mean(na):5.1f}+/-{stdev(na) if len(na)>1 else 0:.1f}")
    print("=" * 74 + "\n")

    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    out_path = Path(__file__).parent / f"verified_recognition_results_{stamp}.json"
    with open(out_path, "w") as f:
        json.dump({
            "protocol": {"verify_k": VERIFY_K,
                         "routed_windows_per_phase": args.steps, "seeds": seeds,
                         "encoder": Path(args.model).name,
                         "note": "persistence gate layered on top of the "
                                 "existing guard-veto class-capture fix"},
            "results": results,
            "wind_sensitivity_curve": curve,
        }, f, indent=2)
    print(f"  Results -> {out_path.name}")


if __name__ == "__main__":
    main()
