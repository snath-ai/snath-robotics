# Fleet Transfer Experiment (N=2) — 2026-07-16

First empirical test of the Snath fleet claim (PAV §4.3 / future direction 6,
PERSIST future direction 4): **Robot B improves from Robot A's D-hard events
without ever experiencing the violation itself.**

Hardware: consumer MacBook, CPU only. Runtime: ~8 min encoder training +
~3 min for the full 5-seed experiment.

## Files

| File | Role |
|---|---|
| `policies.py` | `random` (PAV-paper policy) and `cpg` (scripted periodic gait) |
| `train_cls_multiseed.py` | Multi-seed CLS-GRU training with held-out-seed live verification |
| `fleet_n2_transfer.py` | The N=2 experiment (adapter-library multi-anchor routing) |
| `runs/seed_*/` | Per-seed artifacts: A's event log, the shipped copy, both DMN outputs |
| `fleet_n2_results_*.json` | Full per-window records + aggregate |

## Corrections to the published PAV substrate (all found in the 2026-07-16 audit)

1. **Real ice.** The published `set_friction` modified only the floor geom.
   MuJoCo resolves contact friction as the elementwise max of the geom pair and
   Walker2d-v5 feet carry friction 1.9, so floor-only values of 0.80 and 0.05
   produce **bit-identical trajectories** — the published PAV experiment never
   changed the physics. Fixed here by setting `geom_friction[:, 0]` on all
   geoms (the mechanism PERSIST's `ice_world.py` already uses).
2. **Multi-seed encoder with held-out verification.** The published encoder was
   trained on one seed per class and learned trajectory fingerprints (held-out
   live accuracy 44–50%). Retrained on 20 seeds/class: held-out live accuracy
   1.00 / 0.995, live D gap 0.70.
3. **Structured CPG gait** (PAV future direction 2). Under uniform random
   actions the friction concept is not reliably decodable from held-out windows
   even with real ice and multi-seed data.
4. **EWMA anchor (α = 0.90)** as the PAV paper specifies (the proof code kept
   only the last embedding).
5. **Fixed routed-window counts per phase** (the published run's effective
   sample sizes floated with termination luck: n = 6/20/21).
6. **Adapter-library (multi-anchor) routing.** The published Phase-5 mechanism
   (rank-1 LoRA injected into encoder weights at γ = 0.1) produces a 0.0%
   divergence drop under real physics — the corrected concept space is
   near-one-hot per class, and closing D requires a new reference anchor, not
   a 10% weight nudge. Here the DMN's System-2 LoRA is applied to the robot's
   OWN normal anchor in the space it was trained in (`z' = z + zAB`, γ = 1),
   yielding a named `ice_learned` anchor; the router takes the minimum
   divergence over anchors. V4 content-blindness is preserved (decision is a
   pure function of per-anchor scalars).

## Results (5 seeds: 42, 7, 13, 99, 2026)

| Metric | Value |
|---|---|
| A ice detection (real ice) | 100% every seed |
| Events shipped A→B per seed | 30 (= **570 numbers**; no raw trajectories) |
| B-cold ice mean D | 0.6764 ± 0.0675 |
| B-fleet ice mean D (min over anchors) | **0.0035 ± 0.0069** |
| Fleet transfer D drop | **99.4% ± 1.2** (≥ 97.2% every seed) |
| Ice recognised via transferred anchor | 95.3% ± 10.4 (headline); **conditional recognition 100%** |
| B-fleet normal COMMIT (false-alarm check) | **100% every seed** |
| A self-adaptation drop (corrected Phase 5) | 99.4% ± 1.2 |
| B-fleet novel-wind detection | **0% (FAIL — see finding 3)** |

Pre-registered outcomes: **P1 FAIL by the letter** (seed 13 recognition 77% <
80%) — but the paired window-by-window comparison shows fleet recognition
matches cold-alarm behaviour 30/30 windows: every window that would have
alarmed cold was recognised as ice (23/23), and windows that never looked like
ice were correctly routed COMMIT via the *normal* anchor. The 77% is the
fraction of windows that were ice-like at all in that trajectory.
**P2 PASS** (100% every seed). **P3 FAIL** — a genuine architecture-level
finding, below.

## Three findings

1. **The published PAV experiment never created ice** (floor-only friction is
   a MuJoCo no-op on Walker2d-v5). Its detection numbers measured trajectory
   novelty after `env.reset(seed+1)`, and its encoder's 84.4% "terrain"
   accuracy was seed-fingerprint separation. PERSIST is unaffected
   (`ice_world.py` sets all geoms).
2. **The published Phase-5 adaptation mechanism cannot close a real
   divergence** (0.0% drop). Adapter-library multi-anchor routing — which is
   what the papers' own "adapter library / named physics class" language
   describes — closes it (99.4%).
3. **Confidence trap under OOD physics, quantified** (DAS §8.2's open remark,
   confirmed in the physical domain): lateral wind at 10–20 N perturbs raw
   observation statistics 2.2–2.4× MORE than ice does (obs-stat 5.96/5.33 vs
   ice 2.46, same-trajectory baseline), yet the 2-class CE encoder's
   divergence stays ≈ 0 (0.08 / 0.0005) — the router COMMITs on physics it has
   never seen. Novel-physics detection requires a pre-routing OOD guard; an
   encoder-free obs-statistics signal (PERSIST Eq. 6 style) sees wind clearly
   in controlled comparison, but our naive warmup-baseline implementation of
   it does not survive episode-to-episode drift (normal-phase obs-stat 6.13 ≈
   ice 5.69), so the guard needs PERSIST's within-trajectory baseline
   protocol. Left as the follow-up experiment.

## The fleet claim, as validated

With a concept space that actually separates physics classes, other-fleet
learning reduces to shipping HMAC-signed 19-number events: B consolidates A's
events with its own local DMN, gains a named anchor for a physics class it
has never experienced, recognises it on first contact (100% conditional),
keeps zero false alarms on normal terrain — at a communication cost of 570
numbers instead of raw trajectories. The hard part is not the transfer; it is
building the concept space (finding 1) and knowing when you are outside it
(finding 3).

## Reproduce

```bash
.venv/bin/python experiments/fleet/train_cls_multiseed.py --policy cpg
.venv/bin/python experiments/fleet/fleet_n2_transfer.py
```
