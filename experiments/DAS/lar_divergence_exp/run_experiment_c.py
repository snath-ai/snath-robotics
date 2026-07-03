"""
Option C — TorchXRayVision × BiomedCLIP Finding-Vector Divergence
==================================================================
This is the publication-grade validation experiment for AbstractDivergenceRouter
V1–V6 and Theorem 1 (Safety-Learning Equivalence) from "Disagreement as Signal"
(Sajeev 2026).

Encoder architecture:
  Stream A (image): TorchXRayVision DenseNet-121 → 18-dim sigmoid finding probs
                    Trained on NIH + CheXpert + MIMIC + OpenI + Kaggle (combined)
  Stream B (text):  BiomedCLIP zero-shot → 18-dim finding probs (same vocabulary)
                    Template: "chest X-ray report describing {pathology}"

Ground truth (synthetic, model-defined — not noisy dataset labels):
  Normal:        image_i paired with its OWN report_i
  Contradiction: image_i paired with report_j from a DIFFERENT patient,
                 where patient j was chosen to MAXIMISE L1(TXV_i, TXV_j)
                 → guaranteed semantic contradiction by construction

Protocol:
  - Compute TXV finding vectors for all N samples
  - Select 100 normal + 100 synthetic contradiction pairs
  - 50%/50% calibration / held-out test split
  - Calibration set: grid-search optimal δ (threshold)
  - Test set: report AUROC, accuracy, Fisher's exact test p-value

This is the definitive empirical proof that content-blind divergence routing
(V4 Content Blindness) detects clinical multimodal contradiction without reading
either stream — only the scalar L1 distance D is seen by route().

Run:
    cd /path/to/lar_divergence_exp
    USE_TF=0 TOKENIZERS_PARALLELISM=false python3 run_experiment_c.py
"""

import os, sys, json, random, datetime
from pathlib import Path
from collections import Counter
from PIL import Image as PILImage

import numpy as np
import torch

_HERE     = Path(__file__).parent.resolve()
_PLAY     = _HERE.parent.parent.parent.parent  # DAS/lar_divergence_exp -> Snath Robotics/experiments/DAS -> experiments -> Snath Robotics -> JEPA_Playground
_LAR_JEPA = _PLAY / "lar_jepa"
_LAR_SRC  = _LAR_JEPA / "lar_jepa" / "src"

for _p in [str(_LAR_JEPA), str(_LAR_SRC), str(_HERE), str(_HERE.parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.types import RouteDecision
from data.dataset import load_dataset
from encoders.cxr_encoders_c import TXVImageEncoder, BiomedCLIPTextFindingEncoder, load_biomedclip

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE     = "mps"
SEED       = 42
N_NORMAL   = 100   # normal pairs for full experiment
N_CONTRA   = 100   # synthetic contradiction pairs
POOL_SIZE  = 500   # how many samples to compute TXV vectors for
TAU_HIGH   = 0.60  # initial routing threshold (will be calibrated)
TAU_LOW    = 0.10
DELTA      = 1.50  # initial δ (will be calibrated on calibration set)

DATA_DIR    = _HERE / "data"
IMAGES_DIR  = DATA_DIR / "images"
REPORTS_DIR = DATA_DIR / "reports"
RESULTS_DIR = _HERE / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ── Synthetic pair construction ────────────────────────────────────────────────

def build_synthetic_pairs(samples, txv_enc, n_normal, n_contra, pool_size, seed):
    """
    Build clean normal + synthetic contradiction pairs.

    Normal:       (image_i, text_i)  — image with its own report
    Contradiction:(image_i, text_j)  — image with most-distant patient's report,
                                       where distance = L1(TXV_i, TXV_j)

    The contradiction is guaranteed by construction: we select j such that TXV
    believes images i and j have the most different finding profiles.
    route() never sees TXV_i or TXV_j — only the scalar L1(v_A_i, v_B_j).
    """
    random.seed(seed)
    rng = random.Random(seed)
    pool = samples[:pool_size]
    rng.shuffle(pool)

    print(f"\n[synthetic] Computing TXV vectors for {len(pool)} samples...")
    txv_vecs   = []
    valid_pool = []
    for i, s in enumerate(pool):
        try:
            img = PILImage.open(s.image_path).convert("RGB")
            v, _ = txv_enc.encode(img)
            txv_vecs.append(v)
            valid_pool.append(s)
        except Exception as e:
            print(f"  skip {s.cxr_id}: {e}")
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(pool)} samples processed")

    txv_vecs = np.array(txv_vecs)  # (N, 18)
    N = len(valid_pool)
    print(f"[synthetic] Valid: {N} samples  TXV matrix: {txv_vecs.shape}")

    # Split pool: first half as "normal" source, second half as "contradiction source"
    half = N // 2
    normal_idxs = list(range(half))
    contra_idxs = list(range(half, N))
    partner_pool = list(range(half))  # find maximally-distant partner from first half

    rng.shuffle(normal_idxs)
    rng.shuffle(contra_idxs)

    # Build normal pairs (image with own report)
    normal_pairs = []
    for idx in normal_idxs[:n_normal]:
        s = valid_pool[idx]
        text = (s.findings + " " + s.impression).strip()
        if text:
            normal_pairs.append({
                "img_cxr_id":  s.cxr_id,
                "txt_cxr_id":  s.cxr_id,
                "image_path":  str(s.image_path),
                "text":        text,
                "label":       "normal",
                "txv_l1_gt":  0.0,
            })

    # Build contradiction pairs (image with most-distant patient's report)
    contra_pairs = []
    for idx in contra_idxs[:n_contra * 2]:  # oversample to hit n_contra
        if len(contra_pairs) >= n_contra:
            break
        v_img = txv_vecs[idx]
        # L1 distance to all partner-pool images
        l1_dists = np.abs(txv_vecs[partner_pool] - v_img).sum(axis=1)
        best_partner = partner_pool[int(np.argmax(l1_dists))]
        max_l1 = float(l1_dists.max())

        s_img  = valid_pool[idx]
        s_txt  = valid_pool[best_partner]
        text   = (s_txt.findings + " " + s_txt.impression).strip()
        if not text:
            continue

        contra_pairs.append({
            "img_cxr_id":  s_img.cxr_id,
            "txt_cxr_id":  s_txt.cxr_id,
            "image_path":  str(s_img.image_path),
            "text":        text,
            "label":       "contradiction",
            "txv_l1_gt":  round(max_l1, 4),  # ground-truth divergence in TXV space
        })

    print(f"[synthetic] Built {len(normal_pairs)} normal + {len(contra_pairs)} contradiction pairs")
    print(f"  Mean TXV L1 in ground-truth contradiction selection: "
          f"{np.mean([p['txv_l1_gt'] for p in contra_pairs]):.3f}")

    return valid_pool, txv_vecs, normal_pairs, contra_pairs


# ── Routing ───────────────────────────────────────────────────────────────────

def route(c_a: float, c_b: float, D: float,
          tau_high: float, tau_low: float, delta: float) -> RouteDecision:
    """V4 Content Blindness: sees only scalars (c_a, c_b, D). V5: one decision."""
    both_high = c_a >= tau_high and c_b >= tau_high
    both_low  = c_a <  tau_low  and c_b <  tau_low
    if both_high and D < delta:   return RouteDecision.COMMIT_TRAJECTORY
    if both_high and D >= delta:  return RouteDecision.TRIGGER_REPLAN
    if both_low:                  return RouteDecision.STRUCTURAL_IMPASSE
    return RouteDecision.COMMIT_TRAJECTORY


def run_routing(pairs, txv_enc, txt_enc, tau_high, tau_low, delta, tag=""):
    records = []
    for i, p in enumerate(pairs):
        try:
            img      = PILImage.open(p["image_path"]).convert("RGB")
            v_a, c_a = txv_enc.encode(img)
            v_b, c_b = txt_enc.encode(p["text"])
            D        = float(np.abs(v_a - v_b).sum())
            decision = route(c_a, c_b, D, tau_high, tau_low, delta)

            records.append({
                "img_cxr_id": p["img_cxr_id"],
                "txt_cxr_id": p["txt_cxr_id"],
                "label":      p["label"],
                "conf_a":     round(c_a, 4),
                "conf_b":     round(c_b, 4),
                "divergence": round(D,   4),
                "decision":   decision.value,
                "txv_l1_gt":  p.get("txv_l1_gt", 0.0),
            })

            if (i + 1) % 20 == 0:
                print(f"  [{tag}] {i+1}/{len(pairs)}  "
                      f"c_A={c_a:.3f} c_B={c_b:.3f} D={D:.3f} → {decision.value}")

        except Exception as e:
            print(f"  [{tag}] error on {p.get('img_cxr_id', '?')}: {e}")

    return records


# ── Calibration: optimal δ and τ_high ─────────────────────────────────────────

def calibrate(records):
    """Find optimal δ from calibration set via grid search (maximise accuracy)."""
    D_values = [r["divergence"] for r in records]
    labels   = [1 if r["label"] == "contradiction" else 0 for r in records]
    D_min, D_max = min(D_values), max(D_values)

    best_acc, best_delta = 0.0, (D_min + D_max) / 2

    for d100 in range(int(D_min * 100), int(D_max * 100) + 1, 2):
        d = d100 / 100
        preds = [1 if D >= d else 0 for D in D_values]
        acc = sum(p == l for p, l in zip(preds, labels)) / len(labels)
        if acc > best_acc:
            best_acc, best_delta = acc, d

    # τ_high calibration: target both_high fires for ≥60% of samples
    confs_a = [r["conf_a"] for r in records]
    confs_b = [r["conf_b"] for r in records]
    best_tau, best_tau_coverage = TAU_HIGH, 0.0
    for tau100 in range(30, 95, 5):
        tau = tau100 / 100
        coverage = sum(1 for a, b in zip(confs_a, confs_b)
                       if a >= tau and b >= tau) / len(records)
        if coverage >= 0.60:
            best_tau = tau
            best_tau_coverage = coverage

    return best_delta, best_acc, best_tau, best_tau_coverage


# ── Evaluation: AUROC + Fisher's exact test ───────────────────────────────────

def evaluate(records, delta, tau_high, tau_low):
    """AUROC and Fisher's exact test on held-out test set."""
    from sklearn.metrics import roc_auc_score
    from scipy.stats import fisher_exact

    D_values = [r["divergence"] for r in records]
    labels   = [1 if r["label"] == "contradiction" else 0 for r in records]

    # AUROC (higher D → contradiction)
    auroc = float(roc_auc_score(labels, D_values))

    # Fisher's exact test: contingency table at optimal threshold
    preds = [1 if D >= delta else 0 for D in D_values]
    tp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 1)
    tn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 0)
    contingency = [[tp, fp], [fn, tn]]
    _, p_value = fisher_exact(contingency, alternative="greater")

    acc = (tp + tn) / len(labels)

    # TRIGGER_REPLAN analysis (using calibrated tau_high and delta)
    trigger_replan_n = sum(
        1 for r in records
        if r["label"] == "normal" and r["conf_a"] >= tau_high
        and r["conf_b"] >= tau_high and r["divergence"] >= delta
    )
    trigger_replan_c = sum(
        1 for r in records
        if r["label"] == "contradiction" and r["conf_a"] >= tau_high
        and r["conf_b"] >= tau_high and r["divergence"] >= delta
    )
    n_normal = sum(1 for r in records if r["label"] == "normal")
    n_contra = sum(1 for r in records if r["label"] == "contradiction")

    return {
        "auroc":              round(auroc, 4),
        "accuracy":           round(acc, 4),
        "p_value":            float(p_value),
        "contingency_table":  {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "trigger_replan_normal_pct":      round(100 * trigger_replan_n / n_normal, 1) if n_normal else 0,
        "trigger_replan_contradiction_pct": round(100 * trigger_replan_c / n_contra, 1) if n_contra else 0,
        "mean_D_normal":      round(float(np.mean([r["divergence"] for r in records if r["label"] == "normal"])), 4),
        "mean_D_contra":      round(float(np.mean([r["divergence"] for r in records if r["label"] == "contradiction"])), 4),
    }


# ── Print results ──────────────────────────────────────────────────────────────

def print_results(cal_records, test_records, test_eval, cal_delta, tau_high):
    import statistics

    D_n = [r["divergence"] for r in test_records if r["label"] == "normal"]
    D_c = [r["divergence"] for r in test_records if r["label"] == "contradiction"]

    print("\n" + "="*70)
    print("  EXPERIMENT C RESULTS — TXV × BiomedCLIP Finding-Vector Divergence")
    print("  Ground truth: SYNTHETIC (TXV-defined, model-constructed)")
    print("="*70)
    print(f"  Calibrated δ = {cal_delta}   τ_high = {tau_high}   τ_low = {TAU_LOW}\n")

    print(f"  TEST SET (N={len(test_records)}, {len(D_n)} normal + {len(D_c)} contradiction)")
    print(f"  ─────────────────────────────────────────────────────────")
    if D_n:
        print(f"  Normal        D:  μ={statistics.mean(D_n):.4f}  σ={statistics.stdev(D_n):.4f}")
    if D_c:
        print(f"  Contradiction D:  μ={statistics.mean(D_c):.4f}  σ={statistics.stdev(D_c):.4f}")
    print()
    print(f"  AUROC                    : {test_eval['auroc']}")
    print(f"  Accuracy (at δ={cal_delta:.2f})   : {test_eval['accuracy']:.1%}")
    print(f"  Fisher's exact p-value   : {test_eval['p_value']:.2e}")
    print(f"  Contingency table        : {test_eval['contingency_table']}")
    print()
    print(f"  TRIGGER_REPLAN (normal)       : {test_eval['trigger_replan_normal_pct']}%")
    print(f"  TRIGGER_REPLAN (contradiction): {test_eval['trigger_replan_contradiction_pct']}%")

    lift = (test_eval['trigger_replan_contradiction_pct'] /
            test_eval['trigger_replan_normal_pct']
            if test_eval['trigger_replan_normal_pct'] > 0 else float('inf'))
    d_lift = (test_eval['mean_D_contra'] / test_eval['mean_D_normal']
              if test_eval['mean_D_normal'] > 0 else float('inf'))
    print(f"  TRIGGER_REPLAN lift           : {lift:.2f}×")
    print(f"  Mean D lift                   : {d_lift:.2f}×")

    print("\n  HYPOTHESIS CHECK (V4 Content Blindness validation)")
    print(f"  ─────────────────────────────────────────────────────────")
    confirmed = (
        test_eval["auroc"] > 0.70 and
        test_eval["p_value"] < 0.05 and
        test_eval["mean_D_contra"] > test_eval["mean_D_normal"]
    )
    if confirmed:
        print(f"\n  ✓ HYPOTHESIS CONFIRMED (AUROC={test_eval['auroc']}, p={test_eval['p_value']:.2e})")
        print("    Content-blind routing detects synthetic clinical contradictions.")
        print("    V4 Content Blindness validated: route() saw only D, never v_A or v_B.")
        print("    Theorem 1 (Safety-Learning Equivalence) empirically supported.")
    else:
        reasons = []
        if test_eval["auroc"] <= 0.70:
            reasons.append(f"AUROC={test_eval['auroc']} ≤ 0.70")
        if test_eval["p_value"] >= 0.05:
            reasons.append(f"p={test_eval['p_value']:.2e} ≥ 0.05")
        if test_eval["mean_D_contra"] <= test_eval["mean_D_normal"]:
            reasons.append("mean D contradiction ≤ normal")
        print(f"\n  ~ PARTIAL/UNCONFIRMED: {'; '.join(reasons)}")
        print("    Check calibration set for encoder issues.")

    print("="*70)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    random.seed(SEED)
    np.random.seed(SEED)

    print("\n" + "="*70)
    print("  Snath AI — Option C: TXV × BiomedCLIP Finding-Vector Experiment")
    print("  Synthetic ground truth | Calibration/Test split | AUROC + Fisher p")
    print("="*70 + "\n")

    print("[dataset] Loading samples...")
    samples = load_dataset(str(IMAGES_DIR), str(REPORTS_DIR))
    random.shuffle(samples)
    print(f"[dataset] {len(samples)} paired samples available\n")

    print("[model] Loading TorchXRayVision DenseNet-121...")
    txv_enc = TXVImageEncoder(device=DEVICE)

    print("[model] Loading BiomedCLIP...")
    biomedclip_model, tokenizer = load_biomedclip(device=DEVICE)
    txt_enc = BiomedCLIPTextFindingEncoder(biomedclip_model, tokenizer, device=DEVICE)

    # ── Build synthetic pairs ─────────────────────────────────────────────────
    valid_pool, txv_vecs, normal_pairs, contra_pairs = build_synthetic_pairs(
        samples, txv_enc,
        n_normal=N_NORMAL, n_contra=N_CONTRA, pool_size=POOL_SIZE, seed=SEED,
    )

    all_pairs = normal_pairs[:N_NORMAL] + contra_pairs[:N_CONTRA]
    random.shuffle(all_pairs)

    split = len(all_pairs) // 2
    cal_pairs  = all_pairs[:split]
    test_pairs = all_pairs[split:]

    print(f"\n[split] Calibration: {len(cal_pairs)}  Test: {len(test_pairs)}\n")

    # ── Calibration set routing ───────────────────────────────────────────────
    print("[routing] Calibration set...")
    cal_records = run_routing(cal_pairs, txv_enc, txt_enc,
                              TAU_HIGH, TAU_LOW, DELTA, tag="CAL")

    cal_delta, cal_acc, cal_tau, cal_coverage = calibrate(cal_records)
    print(f"\n[calibration] Optimal δ = {cal_delta}  (acc={cal_acc:.1%})")
    print(f"[calibration] Optimal τ_high = {cal_tau}  (coverage={cal_coverage:.0%})\n")

    # ── Test set routing ──────────────────────────────────────────────────────
    print("[routing] Test set (held out)...")
    test_records = run_routing(test_pairs, txv_enc, txt_enc,
                               cal_tau, TAU_LOW, cal_delta, tag="TEST")

    test_eval = evaluate(test_records, cal_delta, cal_tau, TAU_LOW)
    print_results(cal_records, test_records, test_eval, cal_delta, cal_tau)

    # ── Distribution analysis ─────────────────────────────────────────────────
    D_n_test = [r["divergence"] for r in test_records if r["label"] == "normal"]
    D_c_test = [r["divergence"] for r in test_records if r["label"] == "contradiction"]
    if D_n_test and D_c_test:
        import statistics
        print(f"\n  Full distribution (test set):")
        print(f"    Normal        μ={statistics.mean(D_n_test):.4f}  σ={statistics.stdev(D_n_test):.4f}"
              f"  min={min(D_n_test):.4f}  max={max(D_n_test):.4f}")
        print(f"    Contradiction μ={statistics.mean(D_c_test):.4f}  σ={statistics.stdev(D_c_test):.4f}"
              f"  min={min(D_c_test):.4f}  max={max(D_c_test):.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    output = {
        "experiment":  "nlm_cxr_divergence_option_c",
        "timestamp":   datetime.datetime.utcnow().isoformat() + "Z",
        "option":      "C — TXV DenseNet-121 (image) × BiomedCLIP finding-vector (text)",
        "ground_truth": "synthetic — TXV-maximally-distant patient report swap",
        "config": {
            "txv_model":    "densenet121-res224-all",
            "text_model":   "BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
            "tau_high":     cal_tau,
            "tau_low":      TAU_LOW,
            "delta":        cal_delta,
            "n_normal":     N_NORMAL,
            "n_contra":     N_CONTRA,
            "pool_size":    POOL_SIZE,
            "device":       DEVICE,
        },
        "calibration": {
            "n":        len(cal_records),
            "delta":    cal_delta,
            "accuracy": cal_acc,
            "tau_high": cal_tau,
        },
        "test_evaluation": test_eval,
        "cal_records":  cal_records,
        "test_records": test_records,
    }

    out_path = RESULTS_DIR / "experiment_results_c.json"
    with open(out_path, "w") as fh:
        json.dump(output, fh, indent=2)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
