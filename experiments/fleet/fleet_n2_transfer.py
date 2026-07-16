"""
Fleet Transfer Experiment (N=2) — other-fleet learning via D-hard events.
==========================================================================
First empirical test of the fleet claim (PAV §4.3 / future direction 6,
PERSIST future direction 4): a robot improves from ANOTHER robot's
D-hard events without ever experiencing the violation itself.

Corrections relative to the published PAV proof (2026-07-16 audit):
  * REAL ice — sliding friction set on ALL geoms (PERSIST mechanism).
    The published set_friction touched only the floor geom, a dynamical
    no-op on Walker2d-v5 (feet carry friction 1.9; contact friction
    resolves to the pair max).
  * Multi-seed encoder (20 seeds/class, CPG gait) with held-out-seed
    verification: live acc 1.00/0.995, live D gap 0.70.
  * EWMA reference anchor (alpha=0.90) as the PAV paper specifies.
  * Fixed number of ROUTED windows per phase (effective n does not float
    with termination luck).
  * Adapter-library (multi-anchor) routing. The published Phase-5
    mechanism — rank-1 LoRA injected into the encoder at gamma=0.1 —
    cannot close a real divergence: the corrected concept space is
    near-one-hot per class (conf ~0.999), so closing D requires moving
    the reference to the new class, not nudging encoder weights by 10%.
    Here the DMN's System-2 LoRA is applied to the robot's OWN normal
    anchor in the space it was trained in (z' = z + zAB, gamma=1),
    yielding a named ice anchor; the router takes the min divergence
    over anchors. Content blindness (V4) is preserved: the routing
    decision is still a pure function of per-anchor scalars.

Protocol (per seed):
  Robot A (unit seed s):
    warmup normal -> anchor; normal phase (COMMIT sanity);
    ice phase (accumulate D-hard events, 19 numbers each, HMAC-signed);
    DMN consolidation -> adapter; self-adaptation check (multi-anchor).
  Transfer: A's event JSONL is the ONLY artifact shipped. B runs its own
    DMN on the received events.
  Robot B (unit seed s+1000, never experiences ice before the test):
    B-cold : anchors={normal}. normal / ice / novel(wind) phases.
    B-fleet: anchors={normal, ice_est=own anchor + received adapter}.
             normal (false-alarm check) / ice (recognition check) /
             novel wind (specificity check — must STILL alarm).
    Paired conditions run identical trajectories (same env seed + same
    action RNG stream).

Pre-registered success criteria (each seed):
  P1 recognition transfer: B-fleet ice min-D reduced >= 50% vs B-cold
     AND >= 80% of B-fleet ice windows route COMMIT via the ice anchor.
  P2 no corruption: B-fleet normal COMMIT >= 95%.
  P3 novelty preserved: B-fleet wind detect (REPLAN+IMPASSE) >= 80%.

Run:
    .venv/bin/python experiments/fleet/fleet_n2_transfer.py
"""
from __future__ import annotations

import sys, json, copy, math, argparse, shutil
from pathlib import Path
from datetime import datetime
from collections import deque
from statistics import mean, stdev

import numpy as np
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

import gymnasium
from encoders.robotics.gru_proprio_encoder import GRUProprioEncoder
from divergence_router import DivergenceRouter
from dhard import DHardQueue, RoboticsDHardEvent
from dmn.robotics_dmn import RoboticsDMN
from core.types import RouteDecision
from experiments.fleet.policies import make_policy

FRICTION_NORMAL = 0.80
FRICTION_ICE    = 0.05
WIND_NOVEL      = 4.0     # lateral torso force (N) — physics unseen by either class
DEFAULT_MODEL   = _ROOT / "models" / "pav" / "gru_cls_multiseed_cpg.pt"
RUNS_DIR        = _ROOT / "experiments" / "fleet" / "runs"
EWMA_ALPHA      = 0.90
WARMUP_STEPS    = 120
POLICY          = "cpg"

DETECT = (RouteDecision.TRIGGER_REPLAN, RouteDecision.STRUCTURAL_IMPASSE)


def set_friction(env, f):
    """Sliding friction on ALL geoms (real ice; PERSIST ice_world.py mechanism)."""
    env.unwrapped.model.geom_friction[:, 0] = f


def apply_wind(env, fx: float):
    """Constant lateral force on the torso (PERSIST 'force' zone analogue)."""
    mid = env.unwrapped.model.body("torso").id
    env.unwrapped.data.xfrc_applied[mid, 0] = fx


class GRURunner:
    """Rolling obs buffer -> concept vector."""
    def __init__(self, encoder: GRUProprioEncoder):
        self.encoder = encoder
        self.buf = deque(maxlen=encoder.seq_len)

    def push(self, obs: np.ndarray) -> torch.Tensor | None:
        self.buf.append(obs.copy())
        if len(self.buf) < self.encoder.seq_len:
            return None
        win = torch.from_numpy(np.array(self.buf)).float().unsqueeze(0)
        with torch.no_grad():
            return self.encoder(win).squeeze(0)

    def reset(self):
        self.buf.clear()


def load_base_encoder(model_path: Path) -> GRUProprioEncoder:
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    enc = GRUProprioEncoder(
        payload["obs_dim"], payload["hidden_dim"],
        payload["embed_dim"], payload["seq_len"],
    )
    enc.load_state_dict(payload["encoder_state"])
    enc.eval()
    return enc


def adapted_anchor(z_ref: torch.Tensor, adapter_path: Path) -> torch.Tensor:
    """
    Named-class anchor from a consolidated adapter, applied in the space
    the DMN trained it in (post-LayerNorm concept space, gamma = 1):
        z_class = z_ref + z_ref @ A @ B
    """
    payload = torch.load(str(adapter_path), map_location="cpu", weights_only=False)
    A, B = payload["A"], payload["B"]
    with torch.no_grad():
        return (z_ref + z_ref @ A @ B).detach()


class MultiAnchorRouter:
    """
    Adapter-library routing: evaluate the V1-V6 router against every
    anchor in the library and act on the minimum-divergence anchor.
    V4 preserved — the decision is a pure function of per-anchor scalars
    (D, conf_a, conf_b); anchor content is never inspected here.
    """
    def __init__(self, **router_kwargs):
        self.router = DivergenceRouter(**router_kwargs)

    def route(self, anchors: list[tuple[str, torch.Tensor]], z_live: torch.Tensor):
        best = None
        for name, z_a in anchors:
            r = self.router.route(z_a, z_live)
            if best is None or r.divergence < best[1].divergence:
                best = (name, r)
        return best  # (anchor_name, RoutingResult)


class Robot:
    """One fleet unit: own env, own action RNG, own anchor library."""

    def __init__(self, robot_id: str, encoder: GRUProprioEncoder, seed: int):
        self.id = robot_id
        self.encoder = encoder
        self.runner = GRURunner(encoder)
        self.env = gymnasium.make("Walker2d-v5")
        self.rng = np.random.default_rng(seed)
        self.policy = make_policy(POLICY, self.rng)
        self.seed = seed
        self.marouter = MultiAnchorRouter(tau_high=0.60, tau_low=0.25,
                                          delta=0.35, dhard=None)
        self.anchors: list[tuple[str, torch.Tensor]] = []
        self._obs = None
        self._wind = 0.0

    def _step(self, obs, friction):
        apply_wind(self.env, self._wind)
        nxt, _, term, trunc, _ = self.env.step(self.policy(obs))
        if term or trunc:
            nxt, _ = self.env.reset()
            set_friction(self.env, friction)
            self.runner.reset()
            self.policy.reset()
        return nxt

    def warmup(self):
        """EWMA normal-terrain anchor (PAV paper, alpha=0.90) + obs-stat baseline."""
        obs, _ = self.env.reset(seed=self.seed)
        set_friction(self.env, FRICTION_NORMAL)
        self.runner.reset()
        self.policy.reset()
        self._wind = 0.0
        z_ref = None
        raw_means = []
        for _ in range(WARMUP_STEPS):
            z = self.runner.push(obs)
            if z is not None:
                z = z.detach()
                z_ref = z if z_ref is None \
                    else EWMA_ALPHA * z_ref + (1 - EWMA_ALPHA) * z
                raw_means.append(np.array(self.runner.buf).mean(0))
            obs = self._step(obs, FRICTION_NORMAL)
        self.anchors = [("normal", z_ref)]
        # PERSIST-Eq.6-style baseline: encoder-free OOD guard diagnostic
        raw = np.stack(raw_means)
        self._raw_mu = raw.mean(0)
        self._raw_sd = raw.std(0) + 1e-8
        self._obs = obs

    def obs_stat_div(self) -> float:
        """z-scored rolling-mean L2 vs normal baseline (PERSIST Eq. 6 analogue)."""
        raw = np.array(self.runner.buf).mean(0)
        return float(np.linalg.norm((raw - self._raw_mu) / self._raw_sd))

    def phase(self, friction: float, n_routed: int,
              reset_seed: int | None = None,
              wind: float = 0.0,
              queue: DHardQueue | None = None,
              scenario: str = "",
              max_iters: int = 800) -> dict:
        """Route until n_routed windows have been routed (fixed effective n)."""
        self._wind = wind
        obs = self._obs
        if reset_seed is not None:
            obs, _ = self.env.reset(seed=reset_seed)
            set_friction(self.env, friction)
            self.runner.reset()
            self.policy.reset()
            for _ in range(self.encoder.seq_len - 1):
                self.runner.push(obs)
                obs = self._step(obs, friction)

        records, n_events, it = [], 0, 0
        while len(records) < n_routed and it < max_iters:
            it += 1
            z_p = self.runner.push(obs)
            if z_p is None:
                obs = self._step(obs, friction)
                continue
            anchor_name, r = self.marouter.route(self.anchors, z_p)
            records.append({"iter": it, "D": r.divergence,
                            "decision": r.decision.value,
                            "anchor": anchor_name,
                            "obs_stat": self.obs_stat_div()})
            if queue is not None and r.decision in DETECT:
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
        self._wind = 0.0

        n = max(len(records), 1)
        return {
            "mean_D": sum(r["D"] for r in records) / n,
            "mean_obs_stat": sum(r["obs_stat"] for r in records) / n,
            "commit": sum(r["decision"] == "COMMIT_TRAJECTORY" for r in records) / n,
            "detect": sum(r["decision"] in ("TRIGGER_REPLAN", "STRUCTURAL_IMPASSE")
                          for r in records) / n,
            "commit_via": {a: sum(1 for r in records
                                  if r["decision"] == "COMMIT_TRAJECTORY"
                                  and r["anchor"] == a) / n
                           for a in {r["anchor"] for r in records}},
            "n_routed": len(records),
            "iters": it,
            "n_events": n_events,
            "records": records,
        }

    def close(self):
        self.env.close()


def consolidate(queue_path: Path, adapter_dir: Path, torch_seed: int) -> Path | None:
    """Local DMN cycle over a (possibly received) event log."""
    torch.manual_seed(torch_seed)
    dmn = RoboticsDMN(queue_path=str(queue_path), adapter_dir=str(adapter_dir))
    built = dmn.consolidate(min_events=4, verbose=False)
    if not built:
        return None
    p = adapter_dir / "environmental_transient.pt"
    return p if p.exists() else None


def run_seed(base_encoder: GRUProprioEncoder, seed: int, n_routed: int) -> dict:
    run_dir = RUNS_DIR / f"seed_{seed}"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    out = {"seed": seed}

    # ── Robot A: experiences ice, ships its event log ─────────────────────
    A = Robot("A", base_encoder, seed)
    A.warmup()
    out["a_normal"] = A.phase(FRICTION_NORMAL, n_routed)
    queue_a = DHardQueue(str(run_dir / "robot_a_dhard.jsonl"))
    out["a_ice"] = A.phase(FRICTION_ICE, n_routed, reset_seed=seed + 1,
                           queue=queue_a, scenario=f"fleet_a_ice_s{seed}")

    n_ev = out["a_ice"]["n_events"]
    out["events_shipped"] = n_ev
    out["numbers_shipped"] = n_ev * 19   # 2G + 3 per event (PAV Eq. 4)

    # A's own consolidation + self-adaptation (corrected PAV Phase 5)
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

    # ── Fleet transfer: B consolidates A's events locally ─────────────────
    received = run_dir / "robot_b_received_dhard.jsonl"
    shutil.copy(queue_a.path, received)              # the only thing B receives
    adapter_b = consolidate(received, run_dir / "adapters_b", seed)
    out["adapter_transferred"] = adapter_b is not None

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
    if adapter_b is not None:
        B_fleet = Robot("B-fleet", base_encoder, seed_b)  # same seed: identical trajectories
        B_fleet.warmup()
        B_fleet.anchors.append(
            ("ice_learned", adapted_anchor(B_fleet.anchors[0][1], adapter_b)))
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


def fmt(x, spec=".4f"):
    return format(x, spec) if x is not None else "  —  "


def main():
    global POLICY
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30,
                    help="routed windows per phase (fixed effective n)")
    ap.add_argument("--seeds", type=str, default="42,7,13,99,2026")
    ap.add_argument("--model", type=str, default=str(DEFAULT_MODEL))
    ap.add_argument("--policy", choices=["random", "cpg"], default="cpg")
    args = ap.parse_args()
    POLICY = args.policy
    seeds = [int(s) for s in args.seeds.split(",")]

    print("\n" + "═" * 74)
    print("  Snath Robotics — FLEET TRANSFER (N=2), adapter-library routing")
    print("  Robot B learns ice from Robot A's D-hard events alone")
    print(f"  encoder={Path(args.model).name} · policy={POLICY} · real friction")
    print("═" * 74)

    base = load_base_encoder(Path(args.model))
    all_results = []
    for seed in seeds:
        print(f"\n  seed {seed} …", flush=True)
        r = run_seed(base, seed, args.steps)
        all_results.append(r)
        print(f"    A      : normal COMMIT={r['a_normal']['commit']:.0%} │ "
              f"ice D={r['a_ice']['mean_D']:.4f} detect={r['a_ice']['detect']:.0%} │ "
              f"events={r['events_shipped']} ({r['numbers_shipped']} numbers) │ "
              f"self-drop={fmt(r['a_self_drop_pct'], '.1f')}%")
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

    # ── Aggregate ──────────────────────────────────────────────────────────
    ok = [r for r in all_results if r["transfer_drop_pct"] is not None]
    print("\n" + "═" * 74)
    print(f"  AGGREGATE over {len(ok)}/{len(seeds)} seeds with a transferred adapter")
    print("═" * 74)

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
    agg(lambda r: r["b_fleet_wind"]["mean_obs_stat"],
        "B-fleet wind obs-stat div (guard diag)", ".2f")
    agg(lambda r: r["b_fleet_ice"]["mean_obs_stat"],
        "B-fleet ice obs-stat div (guard diag)", ".2f")
    agg(lambda r: r["b_fleet_normal"]["mean_obs_stat"],
        "B-fleet normal obs-stat div (guard diag)", ".2f")
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
    print("═" * 74 + "\n")

    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    out_path = _ROOT / "experiments" / "fleet" / f"fleet_n2_results_{stamp}.json"
    with open(out_path, "w") as f:
        json.dump({
            "protocol": {
                "n_robots": 2, "routed_windows_per_phase": args.steps,
                "seeds": seeds, "encoder": Path(args.model).name,
                "policy": POLICY, "wind_novel_N": WIND_NOVEL,
                "friction_mechanism": "all-geom sliding friction (real ice)",
                "anchor": f"EWMA alpha={EWMA_ALPHA}, {WARMUP_STEPS} warmup steps",
                "routing": "adapter-library multi-anchor, min divergence",
                "router": {"tau_high": 0.60, "tau_low": 0.25, "delta": 0.35},
                "transfer_artifact": "HMAC-signed D-hard JSONL (19 numbers/event)",
                "criteria": {
                    "P1": "B-fleet ice D drop >= 50% and recognised >= 80%, each seed",
                    "P2": "B-fleet normal COMMIT >= 95%, each seed",
                    "P3": "B-fleet novel-wind detect >= 80%, each seed"},
            },
            "results": all_results,
            "pass": {"P1": p1, "P2": p2, "P3": p3},
        }, f, indent=2)
    print(f"  Results → {out_path.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
