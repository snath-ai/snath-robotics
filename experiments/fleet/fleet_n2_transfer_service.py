"""
Fleet Transfer Experiment (N=2) — SERVICE-ROUTED variant.
===========================================================
Same protocol, same physics, same encoder, same router, same pre-registered
P1/P2/P3 criteria as fleet_n2_transfer.py. The ONLY thing that changes is
the A -> B transfer mechanism:

  ORIGINAL (fleet_n2_transfer.py):
    Robot A writes its HMAC-signed D-hard event JSONL to a local file.
    Robot B copies that file locally and runs its OWN in-process
    RoboticsDMN.consolidate() (dmn/robotics_dmn.py) on it — which trains a
    rank-1 LoRA correction (100 epochs AdamW, L1 loss) mapping the class's
    normal anchor onto (approximately) the mean of the shipped z_proprio
    vectors, applied as z' = z + z@A@B ("ice_learned" anchor).

  HERE:
    Robot A POSTs its HMAC-verified D-hard events to a real, running,
    separately-implemented DMN HTTP service (dmn_service/app.py, started as
    a genuine uvicorn subprocess) via POST /events. Robot B does a real
    GET /library/{class_name} against that service and uses the returned
    centroid — computed by dmn_service/consolidation.py's consolidate_class,
    a PLAIN ARITHMETIC MEAN of the submitted z_proprio vectors (the LTL
    paper's Delta_bar_c definition) — directly as its "ice_learned" anchor.
    No LoRA training happens anywhere in this path.

    concept_anchor = z_proprio (event.z_proprio), matching the same
    convention dmn_service/real_data.py already uses for these exact
    ood_guard-produced D-hard events ("proprioception is the diagnostic
    stream for this failure class").

    class_name is seed-scoped (`environmental_transient_s{seed}`) so that
    5 independent Monte-Carlo repeats of the SAME protocol, run against ONE
    persistent service instance in one process lifetime, do not contaminate
    each other's centroids — exactly the isolation separate real-world
    incidents would have. `failure_class` stays the unscoped
    "environmental_transient" so the service's decay-lambda table lookup
    (LTL paper's Table, robotics instantiation) still resolves correctly.

    Robot A's own LOCAL self-adaptation (a_ice -> a_ice_adapted, its own
    RoboticsDMN LoRA on its own anchor) is UNCHANGED — it is not part of
    the A -> B transfer and the task only asks to swap the transfer leg.

Everything else (Robot physics stepping, GRURunner, GRUProprioEncoder,
MultiAnchorRouter/DivergenceRouter, EWMA anchor, fixed routed-window counts,
P1/P2/P3 definitions) is imported unmodified from fleet_n2_transfer.py.

Run:
    .venv/bin/python experiments/fleet/fleet_n2_transfer_service.py
"""
from __future__ import annotations

import sys, os, json, shutil, subprocess, time, argparse
from pathlib import Path
from datetime import datetime
from statistics import mean, stdev

import torch
import requests

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import fleet_n2_transfer as base
from fleet_n2_transfer import (
    Robot, load_base_encoder, consolidate, adapted_anchor,
    FRICTION_NORMAL, FRICTION_ICE, WIND_NOVEL, DEFAULT_MODEL, RUNS_DIR, fmt,
)

# ─────────────────────────────────────────────────────────────────────────
# Real DMN HTTP service — genuine uvicorn subprocess, not an in-process call
# ─────────────────────────────────────────────────────────────────────────
DMN_SERVICE_DIR = Path("/Users/aadithya/Desktop/Private/personal/dmn_service")
# The service needs fastapi/uvicorn/httpx/pydantic + Lar 2.3.0's checkpoint.py.
# The robot experiment's own .venv has torch/gymnasium/mujoco but NOT those
# HTTP-service deps; the system pyenv python has the service's deps but not
# torch/gymnasium. So the service is launched as a SEPARATE subprocess under
# its own interpreter — real process boundary, real HTTP between them.
DMN_PYTHON = os.environ.get("DMN_PYTHON", "/Users/aadithya/.pyenv/shims/python3")
DMN_PORT = int(os.environ.get("DMN_PORT", "8899"))
DMN_BASE_URL = f"http://127.0.0.1:{DMN_PORT}"
DMN_HMAC_SECRET = "fleet-n2-service-transfer-2026"
DMN_STORAGE_DIR = DMN_SERVICE_DIR / "_fleet_n2_run" / "storage"
DMN_MIN_EVENTS = "3"


def start_dmn_service() -> subprocess.Popen:
    if DMN_STORAGE_DIR.exists():
        shutil.rmtree(DMN_STORAGE_DIR)
    DMN_STORAGE_DIR.mkdir(parents=True)
    env = dict(os.environ)
    env["DMN_STORAGE_DIR"] = str(DMN_STORAGE_DIR)
    env["DMN_HMAC_SECRET"] = DMN_HMAC_SECRET
    env["DMN_MIN_EVENTS_FOR_CONSOLIDATION"] = DMN_MIN_EVENTS
    proc = subprocess.Popen(
        [DMN_PYTHON, "-m", "uvicorn", "app:app", "--port", str(DMN_PORT), "--host", "127.0.0.1"],
        cwd=str(DMN_SERVICE_DIR), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    deadline = time.time() + 20.0
    while time.time() < deadline:
        if proc.poll() is not None:
            out = proc.stdout.read() if proc.stdout else ""
            raise RuntimeError(f"DMN service process exited early (code {proc.returncode}):\n{out}")
        try:
            r = requests.get(f"{DMN_BASE_URL}/health", timeout=1)
            if r.status_code == 200:
                print(f"  DMN service up: {r.json()}", flush=True)
                return proc
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(0.3)
    proc.kill()
    raise RuntimeError("DMN service never became healthy within 20s")


def stop_dmn_service(proc: subprocess.Popen) -> str:
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    out, _ = proc.communicate()
    return out or ""


def ship_events_to_service(events, unit_id: str, class_name: str, failure_class: str) -> dict:
    """POST every HMAC-verified D-hard event, then force a final consolidate
    so the served library reflects every event just shipped."""
    last_ack = None
    for e in events:
        payload = {
            "unit_id": unit_id,
            "class_name": class_name,
            "concept_anchor": e.z_proprio,
            "n_source_events": 1,
            "failure_class": failure_class,
        }
        r = requests.post(f"{DMN_BASE_URL}/events", json=payload, timeout=5)
        r.raise_for_status()
        last_ack = r.json()
    r = requests.post(f"{DMN_BASE_URL}/consolidate/{class_name}",
                       params={"failure_class": failure_class}, timeout=5)
    r.raise_for_status()
    return {"ack": last_ack, "consolidate": r.json()}


def fetch_library_anchor(class_name: str):
    r = requests.get(f"{DMN_BASE_URL}/library/{class_name}", timeout=5)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()


# ─────────────────────────────────────────────────────────────────────────
# Per-seed protocol — A side identical to base; B side service-routed
# ─────────────────────────────────────────────────────────────────────────

def run_seed_service(base_encoder, seed: int, n_routed: int) -> dict:
    run_dir = RUNS_DIR / f"seed_{seed}_service"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    out = {"seed": seed}

    # ── Robot A: identical physics/encoder/router; still writes a local
    #    HMAC-signed queue (unchanged event *generation* -- only *shipping*
    #    to B changes below) ───────────────────────────────────────────────
    A = Robot("A", base_encoder, seed)
    A.warmup()
    out["a_normal"] = A.phase(FRICTION_NORMAL, n_routed)
    queue_a = base.DHardQueue(str(run_dir / "robot_a_dhard.jsonl"))
    out["a_ice"] = A.phase(FRICTION_ICE, n_routed, reset_seed=seed + 1,
                           queue=queue_a, scenario=f"fleet_a_ice_s{seed}")

    n_ev = out["a_ice"]["n_events"]
    out["events_shipped"] = n_ev
    out["numbers_shipped"] = n_ev * 19

    # A's own local self-adaptation is UNCHANGED (not part of the A->B leg)
    adapter_a = consolidate(Path(queue_a.path), run_dir / "adapters_a", seed)
    out["a_ice_adapted"] = None
    out["a_self_drop_pct"] = None
    if adapter_a is not None:
        A.anchors.append(("ice_learned", adapted_anchor(A.anchors[0][1], adapter_a)))
        out["a_ice_adapted"] = A.phase(FRICTION_ICE, n_routed, reset_seed=seed + 1)
        if out["a_ice"]["mean_D"] > 0:
            out["a_self_drop_pct"] = 100 * (out["a_ice"]["mean_D"]
                                            - out["a_ice_adapted"]["mean_D"]) \
                                     / out["a_ice"]["mean_D"]
    A.close()

    # ── Fleet transfer: SERVICE-ROUTED (real HTTP, real HMAC-signed store,
    #    real independently-implemented consolidation math) ───────────────
    verified_events = queue_a.verified()   # same integrity filter the local
    # flow applies via DHardQueue.resolved() -> verified(); only HMAC-valid
    # events are ever shipped, service- or file-routed.
    class_name = f"environmental_transient_s{seed}"
    unit_id = f"robot_a_seed{seed}"
    ship_info = ship_events_to_service(verified_events, unit_id, class_name,
                                       "environmental_transient")
    entry = fetch_library_anchor(class_name)
    out["service_consolidated"] = entry is not None
    out["service_entry"] = entry
    out["service_ship_info"] = ship_info
    adapter_b_available = entry is not None

    # ── Robot B: paired cold vs fleet on identical trajectories ───────────
    seed_b = seed + 1000

    B_cold = Robot("B-cold", base_encoder, seed_b)
    B_cold.warmup()
    out["b_cold_normal"] = B_cold.phase(FRICTION_NORMAL, n_routed)
    out["b_cold_ice"] = B_cold.phase(FRICTION_ICE, n_routed, reset_seed=seed_b + 1)
    out["b_cold_wind"] = B_cold.phase(FRICTION_NORMAL, n_routed,
                                      reset_seed=seed_b + 2, wind=WIND_NOVEL)
    B_cold.close()

    for k in ("b_fleet_normal", "b_fleet_ice", "b_fleet_wind"):
        out[k] = None
    out["transfer_drop_pct"] = None
    out["ice_recognised"] = None
    if adapter_b_available:
        B_fleet = Robot("B-fleet", base_encoder, seed_b)  # same seed: identical trajectories
        B_fleet.warmup()
        centroid = torch.tensor(entry["centroid"], dtype=torch.float32)
        B_fleet.anchors.append(("ice_learned", centroid))
        out["b_fleet_normal"] = B_fleet.phase(FRICTION_NORMAL, n_routed)
        out["b_fleet_ice"] = B_fleet.phase(FRICTION_ICE, n_routed,
                                           reset_seed=seed_b + 1)
        out["b_fleet_wind"] = B_fleet.phase(FRICTION_NORMAL, n_routed,
                                            reset_seed=seed_b + 2, wind=WIND_NOVEL)
        B_fleet.close()
        if out["b_cold_ice"]["mean_D"] > 0:
            out["transfer_drop_pct"] = 100 * (out["b_cold_ice"]["mean_D"]
                                              - out["b_fleet_ice"]["mean_D"]) \
                                       / out["b_cold_ice"]["mean_D"]
        out["ice_recognised"] = out["b_fleet_ice"]["commit_via"].get("ice_learned", 0.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30,
                    help="routed windows per phase (fixed effective n)")
    ap.add_argument("--seeds", type=str, default="42,7,13,99,2026")
    ap.add_argument("--model", type=str, default=str(DEFAULT_MODEL))
    ap.add_argument("--policy", choices=["random", "cpg"], default="cpg")
    args = ap.parse_args()
    base.POLICY = args.policy
    seeds = [int(s) for s in args.seeds.split(",")]

    print("\n" + "═" * 78)
    print("  Snath Robotics — FLEET TRANSFER (N=2), SERVICE-ROUTED")
    print("  Robot A -> real DMN HTTP service -> Robot B (real HMAC-signed centroid)")
    print(f"  encoder={Path(args.model).name} · policy={base.POLICY} · real friction")
    print("═" * 78)

    print("\n  starting real DMN service (uvicorn subprocess) …")
    proc = start_dmn_service()

    try:
        base_encoder = load_base_encoder(Path(args.model))
        all_results = []
        for seed in seeds:
            print(f"\n  seed {seed} …", flush=True)
            r = run_seed_service(base_encoder, seed, args.steps)
            all_results.append(r)
            print(f"    A      : normal COMMIT={r['a_normal']['commit']:.0%} │ "
                  f"ice D={r['a_ice']['mean_D']:.4f} detect={r['a_ice']['detect']:.0%} │ "
                  f"events={r['events_shipped']} ({r['numbers_shipped']} numbers) │ "
                  f"self-drop={fmt(r['a_self_drop_pct'], '.1f')}%")
            print(f"    service: consolidated={r['service_consolidated']} │ "
                  f"n_events={r['service_entry']['n_events'] if r['service_entry'] else '—'} │ "
                  f"tau_sim={r['service_entry']['tau_sim'] if r['service_entry'] else '—'}")
            bc_i, bf_i = r["b_cold_ice"], r["b_fleet_ice"]
            bc_w, bf_w = r["b_cold_wind"], r["b_fleet_wind"]
            print(f"    B-cold : normal COMMIT={r['b_cold_normal']['commit']:.0%} │ "
                  f"ice D={bc_i['mean_D']:.4f} detect={bc_i['detect']:.0%} │ "
                  f"wind detect={bc_w['detect']:.0%}")
            if bf_i is not None:
                print(f"    B-fleet: normal COMMIT={r['b_fleet_normal']['commit']:.0%} │ "
                      f"ice D={bf_i['mean_D']:.4f} recognised={r['ice_recognised']:.0%} "
                      f"drop={fmt(r['transfer_drop_pct'], '.1f')}% │ "
                      f"wind detect={bf_w['detect']:.0%}")

        ok = [r for r in all_results if r["transfer_drop_pct"] is not None]
        print("\n" + "═" * 78)
        print(f"  AGGREGATE over {len(ok)}/{len(seeds)} seeds with a service-consolidated anchor")
        print("═" * 78)

        def agg(key_fn, label, spec=".4f"):
            vals = [key_fn(r) for r in ok]
            sd = stdev(vals) if len(vals) > 1 else 0.0
            print(f"  {label:<40} {mean(vals):{spec}} ± {sd:{spec}}")
            return vals

        agg(lambda r: r["b_cold_ice"]["mean_D"],   "B-cold  ice mean D")
        agg(lambda r: r["b_fleet_ice"]["mean_D"],  "B-fleet ice mean D (min over anchors)")
        drops = agg(lambda r: r["transfer_drop_pct"], "Fleet transfer D drop (%)", ".1f")
        recog = agg(lambda r: r["ice_recognised"] * 100, "Ice recognised via ice anchor (%)", ".1f")
        agg(lambda r: r["a_self_drop_pct"], "A self-adaptation D drop (%)", ".1f")
        commits = agg(lambda r: r["b_fleet_normal"]["commit"] * 100,
                      "B-fleet normal COMMIT (%)", ".1f")
        winds = agg(lambda r: r["b_fleet_wind"]["detect"] * 100,
                    "B-fleet novel-wind detect (%)", ".1f")
        agg(lambda r: float(r["numbers_shipped"]), "Numbers shipped A->B", ".0f")

        p1 = all(d >= 50 for d in drops) and all(x >= 80 for x in recog)
        p2 = all(c >= 95 for c in commits)
        p3 = all(w >= 80 for w in winds)
        print(f"\n  P1  recognition transfer (D drop ≥50% ∧ recognised ≥80%): "
              f"{'PASS ✓' if p1 else 'FAIL ✗'}")
        print(f"  P2  no corruption (normal COMMIT ≥95%)                  : "
              f"{'PASS ✓' if p2 else 'FAIL ✗'}")
        print(f"  P3  novelty preserved (wind detect ≥80%)                : "
              f"{'PASS ✓' if p3 else 'FAIL ✗'}")
        print("═" * 78 + "\n")

        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        out_path = _ROOT / "experiments" / "fleet" / f"fleet_n2_service_results_{stamp}.json"
        with open(out_path, "w") as f:
            json.dump({
                "protocol": {
                    "n_robots": 2, "routed_windows_per_phase": args.steps,
                    "seeds": seeds, "encoder": Path(args.model).name,
                    "policy": base.POLICY, "wind_novel_N": WIND_NOVEL,
                    "transfer_mechanism": "real HTTP to dmn_service (app.py), "
                                          "POST /events + GET /library/{class}, "
                                          "consolidation.consolidate_class "
                                          "(arithmetic-mean centroid, LTL paper)",
                    "criteria": {
                        "P1": "B-fleet ice D drop >= 50% and recognised >= 80%, each seed",
                        "P2": "B-fleet normal COMMIT >= 95%, each seed",
                        "P3": "B-fleet novel-wind detect >= 80%, each seed"},
                },
                "results": all_results,
                "pass": {"P1": p1, "P2": p2, "P3": p3},
            }, f, indent=2, default=str)
        print(f"  Results → {out_path.relative_to(_ROOT)}")
    finally:
        log = stop_dmn_service(proc)
        log_path = DMN_SERVICE_DIR / "_fleet_n2_run" / "server.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(log)
        print(f"  DMN service stopped. Full server log → {log_path}")


if __name__ == "__main__":
    main()
