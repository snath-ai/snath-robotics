"""
Option C2 — BiomedCLIP Finding-Vector Divergence (18-finding, full protocol)
=============================================================================
Publication-grade validation of AbstractDivergenceRouter V1–V6 and Theorem 1
(Safety-Learning Equivalence) from "Disagreement as Signal" (Sajeev 2026).

Architecture:
  Stream A (image): BiomedCLIP image encoder + 18 finding templates
    → p_k = softmax([sim(image, "chest X-ray showing {f_k}"),
                     sim(image, "normal chest X-ray, no {f_k}")])[0]

  Stream B (text):  BiomedCLIP text encoder + same 18 finding templates
    → p_k = softmax([sim(report, "chest X-ray showing {f_k}"),
                     sim(report, "normal chest X-ray, no {f_k}")])[0]

Both streams produce an 18-dim [0,1] finding probability vector in BiomedCLIP's
shared biomedical space. L1 distance is the divergence signal.

Ground truth (synthetic, model-defined):
  Normal:       (image_i, text_i)  — image with its own report
  Contradiction:(image_i, text_j)  — image with a DIFFERENT patient's report,
                where j was selected to MAXIMISE L1(TXV_i, TXV_j) in
                TorchXRayVision's 18-dim space (independent ground truth oracle).

Protocol:
  - Build 100 normal + 100 synthetic contradiction pairs
  - 50% calibration / 50% held-out test split (random, stratified)
  - Calibration: grid-search optimal δ (divergence threshold)
  - Test: AUROC, accuracy, Fisher's exact test p-value

Key property (V4 Content Blindness):
  route() sees only the scalar D = L1(v_A, v_B), never v_A or v_B themselves.
  Detection is content-blind: the system catches clinical contradictions
  from a single scalar without reading either stream.

Run:
    cd /path/to/lar_divergence_exp
    USE_TF=0 TOKENIZERS_PARALLELISM=false python3 run_experiment_c2.py
"""

import os, sys, json, random, datetime
from pathlib import Path
from collections import Counter
from PIL import Image as PILImage

import numpy as np
import torch
import torch.nn.functional as F
import open_clip

_HERE     = Path(__file__).parent.resolve()
_PLAY     = _HERE.parent.parent.parent.parent  # DAS/lar_divergence_exp -> Snath Robotics/experiments/DAS -> experiments -> Snath Robotics -> JEPA_Playground
_LAR_JEPA = _PLAY / "lar_jepa"
_LAR_SRC  = _LAR_JEPA / "lar_jepa" / "src"

for _p in [str(_LAR_JEPA), str(_LAR_SRC), str(_HERE), str(_HERE.parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.types import RouteDecision
from data.dataset import load_dataset
import torchxrayvision as xrv
import torchvision.transforms as T

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE     = "mps"
SEED       = 42
N_NORMAL   = 100
N_CONTRA   = 100
POOL_SIZE  = 500   # samples to compute TXV oracle vectors on (for GT construction)
TAU_HIGH   = 0.70  # initial (will be calibrated)
TAU_LOW    = 0.10
DELTA_INIT = 2.00  # initial (will be calibrated)

MODEL_ID   = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

# 18-finding vocabulary — TXV pathology names, naturalised for BiomedCLIP templates
FINDINGS_18 = [
    "atelectasis",
    "consolidation",
    "infiltration",
    "pneumothorax",
    "pulmonary edema",
    "emphysema",
    "pulmonary fibrosis",
    "pleural effusion",
    "pneumonia",
    "pleural thickening",
    "cardiomegaly",
    "pulmonary nodule",
    "pulmonary mass",
    "diaphragmatic hernia",
    "lung lesion",
    "rib fracture",
    "lung opacity",
    "enlarged cardiomediastinum",
]

DATA_DIR    = _HERE / "data"
IMAGES_DIR  = DATA_DIR / "images"
REPORTS_DIR = DATA_DIR / "reports"
RESULTS_DIR = _HERE / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ── BiomedCLIP 18-finding encoder (shared for both streams) ───────────────────

class FindingVectorEncoder18:
    """
    BiomedCLIP zero-shot 18-dim finding probability encoder.
    Works for image inputs (Stream A) and text inputs (Stream B).

    For each finding f_k:
        present_template = "chest X-ray showing {f_k}"
        absent_template  = "normal chest X-ray, no {f_k}"
        p_k = softmax(logit_scale * [sim(input, present), sim(input, absent)])[0]

    Confidence = mean max-softmax across 18 findings (decisiveness), rescaled to [0,1].
    """

    def __init__(self, model, preprocess, tokenizer, device):
        self.model      = model
        self.preprocess = preprocess
        self.tokenizer  = tokenizer
        self.device     = device
        self.K          = len(FINDINGS_18)

        present_texts = [f"chest X-ray showing {f}"          for f in FINDINGS_18]
        absent_texts  = [f"normal chest X-ray, no {f}"       for f in FINDINGS_18]

        with torch.no_grad():
            tok_p = tokenizer(present_texts).to(device)
            tok_a = tokenizer(absent_texts).to(device)
            emb_p = F.normalize(model.encode_text(tok_p).float(), dim=-1)  # (18, 512)
            emb_a = F.normalize(model.encode_text(tok_a).float(), dim=-1)  # (18, 512)

        self._templates = torch.zeros(2 * self.K, 512, device=device)
        for k in range(self.K):
            self._templates[2 * k]     = emb_p[k]
            self._templates[2 * k + 1] = emb_a[k]

        print(f"[FindingVectorEncoder18] {self.K} findings, {2*self.K} templates.")

    @torch.no_grad()
    def encode_image(self, image: PILImage.Image) -> tuple[np.ndarray, float]:
        img_t = self.preprocess(image).unsqueeze(0).to(self.device)
        emb   = F.normalize(self.model.encode_image(img_t).float(), dim=-1)
        return self._project(emb)

    @torch.no_grad()
    def encode_text(self, text: str) -> tuple[np.ndarray, float]:
        tok = self.tokenizer([text]).to(self.device)
        emb = F.normalize(self.model.encode_text(tok).float(), dim=-1)
        return self._project(emb)

    def _project(self, emb: torch.Tensor) -> tuple[np.ndarray, float]:
        logit_scale = self.model.logit_scale.exp()
        sims = logit_scale * (emb @ self._templates.T)  # (1, 36)

        probs     = np.zeros(self.K)
        max_probs = []
        for k in range(self.K):
            pair = sims[0, 2 * k : 2 * k + 2].softmax(0)
            probs[k] = pair[0].item()
            max_probs.append(pair.max().item())

        conf = float(sum(max_probs) / len(max_probs))
        conf = (conf - 0.5) * 2.0
        return probs, min(max(conf, 0.0), 1.0)


# ── TXV oracle for ground-truth construction ──────────────────────────────────

class TXVOracle:
    """TorchXRayVision DenseNet-121 used ONLY for constructing synthetic GT labels.
    Not used in routing — V4 Content Blindness is preserved."""

    def __init__(self, device: str = "mps"):
        self.device = device
        self.model = xrv.models.DenseNet(weights="densenet121-res224-all").to(device).eval()
        self._transform = T.Compose([
            T.Grayscale(num_output_channels=1),
            T.Resize((224, 224)),
            T.ToTensor(),
        ])

    @torch.no_grad()
    def predict(self, image: PILImage.Image) -> np.ndarray:
        img_t = self._transform(image).unsqueeze(0).to(self.device)
        img_t = img_t * 2048 - 1024
        preds = self.model(img_t).squeeze(0).clamp(0, 1)
        preds[torch.isnan(preds)] = 0.5
        return preds.cpu().numpy()


# ── Synthetic pair construction using TXV oracle ──────────────────────────────

def build_synthetic_pairs(samples, txv_oracle, n_normal, n_contra, pool_size, seed):
    """
    Normal:       (image_i, text_i)  — own report
    Contradiction:(image_i, text_j)  — report from most TXV-distant patient j

    TXV oracle is used ONLY to select j (ground truth). It is never used in routing.
    """
    rng = random.Random(seed)
    pool = list(samples[:pool_size])
    rng.shuffle(pool)

    print(f"\n[GT oracle] Computing TXV vectors for {len(pool)} samples...")
    txv_vecs, valid_pool = [], []
    for i, s in enumerate(pool):
        try:
            img = PILImage.open(s.image_path).convert("RGB")
            txv_vecs.append(txv_oracle.predict(img))
            valid_pool.append(s)
        except Exception as e:
            print(f"  skip {s.cxr_id}: {e}")
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(pool)}")

    txv_vecs = np.array(txv_vecs)  # (N, 18)
    N = len(valid_pool)
    print(f"[GT oracle] Valid: {N}  TXV matrix: {txv_vecs.shape}")

    half = N // 2
    normal_idxs  = list(range(half))
    contra_idxs  = list(range(half, N))
    partner_pool = list(range(half))

    rng.shuffle(normal_idxs)
    rng.shuffle(contra_idxs)

    normal_pairs = []
    for idx in normal_idxs[:n_normal]:
        s = valid_pool[idx]
        text = (s.findings + " " + s.impression).strip()
        if text:
            normal_pairs.append({"img_cxr_id": s.cxr_id, "txt_cxr_id": s.cxr_id,
                                  "image_path": str(s.image_path), "text": text,
                                  "label": "normal", "txv_l1_gt": 0.0})

    contra_pairs = []
    for idx in contra_idxs:
        if len(contra_pairs) >= n_contra:
            break
        v_img = txv_vecs[idx]
        l1_dists = np.abs(txv_vecs[partner_pool] - v_img).sum(axis=1)
        best_partner = partner_pool[int(np.argmax(l1_dists))]
        s_img, s_txt = valid_pool[idx], valid_pool[best_partner]
        text = (s_txt.findings + " " + s_txt.impression).strip()
        if text:
            contra_pairs.append({
                "img_cxr_id": s_img.cxr_id, "txt_cxr_id": s_txt.cxr_id,
                "image_path": str(s_img.image_path), "text": text,
                "label": "contradiction",
                "txv_l1_gt": round(float(l1_dists.max()), 4),
            })

    print(f"[GT oracle] {len(normal_pairs)} normal + {len(contra_pairs)} contradiction pairs")
    if contra_pairs:
        print(f"  Mean TXV L1 (GT selection): {np.mean([p['txv_l1_gt'] for p in contra_pairs]):.3f}")

    return normal_pairs, contra_pairs


# ── Routing ───────────────────────────────────────────────────────────────────

def route(c_a, c_b, D, tau_h, tau_l, delta):
    """V4 Content Blindness: sees only scalars."""
    bh = c_a >= tau_h and c_b >= tau_h
    bl = c_a <  tau_l and c_b <  tau_l
    if bh and D < delta:  return RouteDecision.COMMIT_TRAJECTORY
    if bh and D >= delta: return RouteDecision.TRIGGER_REPLAN
    if bl:                return RouteDecision.STRUCTURAL_IMPASSE
    return RouteDecision.COMMIT_TRAJECTORY


def run_routing(pairs, enc, tau_h, tau_l, delta, tag=""):
    records = []
    for i, p in enumerate(pairs):
        try:
            img      = PILImage.open(p["image_path"]).convert("RGB")
            v_a, c_a = enc.encode_image(img)
            v_b, c_b = enc.encode_text(p["text"])
            D        = float(np.abs(v_a - v_b).sum())
            decision = route(c_a, c_b, D, tau_h, tau_l, delta)

            records.append({
                "img_cxr_id": p["img_cxr_id"],
                "txt_cxr_id": p["txt_cxr_id"],
                "label":      p["label"],
                "conf_a":     round(c_a, 4),
                "conf_b":     round(c_b, 4),
                "divergence": round(D, 4),
                "decision":   decision.value,
                "txv_l1_gt":  p.get("txv_l1_gt", 0.0),
                "v_a":        [round(x, 3) for x in v_a.tolist()],
                "v_b":        [round(x, 3) for x in v_b.tolist()],
            })

            if (i + 1) % 20 == 0:
                print(f"  [{tag}] {i+1}/{len(pairs)}  "
                      f"c_A={c_a:.3f} c_B={c_b:.3f} D={D:.3f} → {decision.value}")
        except Exception as e:
            print(f"  [{tag}] error {p.get('img_cxr_id','?')}: {e}")

    return records


# ── Calibration ───────────────────────────────────────────────────────────────

def calibrate(records):
    D_vals = [r["divergence"] for r in records]
    labels = [1 if r["label"] == "contradiction" else 0 for r in records]
    D_min, D_max = min(D_vals), max(D_vals)

    best_acc, best_delta = 0.0, (D_min + D_max) / 2
    for d100 in range(int(D_min * 100), int(D_max * 100) + 1, 2):
        d = d100 / 100
        acc = sum((D >= d) == bool(l) for D, l in zip(D_vals, labels)) / len(labels)
        if acc > best_acc:
            best_acc, best_delta = acc, d

    confs_a = [r["conf_a"] for r in records]
    confs_b = [r["conf_b"] for r in records]
    best_tau, best_cov = TAU_HIGH, 0.0
    for t100 in range(20, 95, 5):
        tau = t100 / 100
        cov = sum(1 for a, b in zip(confs_a, confs_b) if a >= tau and b >= tau) / len(records)
        if cov >= 0.50:
            best_tau = tau
            best_cov = cov

    return best_delta, best_acc, best_tau, best_cov


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(records, delta, tau_h, tau_l):
    from sklearn.metrics import roc_auc_score
    from scipy.stats import fisher_exact
    import statistics

    D_vals = [r["divergence"] for r in records]
    labels = [1 if r["label"] == "contradiction" else 0 for r in records]

    auroc   = float(roc_auc_score(labels, D_vals))
    preds   = [int(D >= delta) for D in D_vals]
    tp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 1)
    tn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 0)
    _, p_val = fisher_exact([[tp, fp], [fn, tn]], alternative="greater")
    acc = (tp + tn) / len(labels)

    D_n = [r["divergence"] for r in records if r["label"] == "normal"]
    D_c = [r["divergence"] for r in records if r["label"] == "contradiction"]
    n_n = len(D_n); n_c = len(D_c)

    tr_n = sum(1 for r in records if r["label"] == "normal"
               and r["conf_a"] >= tau_h and r["conf_b"] >= tau_h
               and r["divergence"] >= delta)
    tr_c = sum(1 for r in records if r["label"] == "contradiction"
               and r["conf_a"] >= tau_h and r["conf_b"] >= tau_h
               and r["divergence"] >= delta)

    return {
        "auroc":            round(auroc, 4),
        "accuracy":         round(acc,   4),
        "p_value":          float(p_val),
        "contingency":      {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "mean_D_normal":    round(statistics.mean(D_n), 4) if D_n else 0,
        "mean_D_contra":    round(statistics.mean(D_c), 4) if D_c else 0,
        "std_D_normal":     round(statistics.stdev(D_n), 4) if len(D_n) > 1 else 0,
        "std_D_contra":     round(statistics.stdev(D_c), 4) if len(D_c) > 1 else 0,
        "trigger_replan_normal_pct":      round(100 * tr_n / n_n, 1) if n_n else 0,
        "trigger_replan_contradiction_pct": round(100 * tr_c / n_c, 1) if n_c else 0,
    }


# ── Print results ──────────────────────────────────────────────────────────────

def print_results(test_eval, cal_delta, tau_h):
    print("\n" + "="*70)
    print("  EXPERIMENT C2 — BiomedCLIP 18-finding vectors (calibrated protocol)")
    print("  Encoder: BiomedCLIP (image + text, shared space)")
    print("  Ground truth: SYNTHETIC (TXV-oracle-defined, model-constructed)")
    print("="*70)
    print(f"  Calibrated δ = {cal_delta}   τ_high = {tau_h}   τ_low = {TAU_LOW}")
    print(f"  Findings: {len(FINDINGS_18)}\n")

    ev = test_eval
    print(f"  Normal        D:  μ={ev['mean_D_normal']:.4f}  σ={ev['std_D_normal']:.4f}")
    print(f"  Contradiction D:  μ={ev['mean_D_contra']:.4f}  σ={ev['std_D_contra']:.4f}")
    d_lift = ev['mean_D_contra'] / ev['mean_D_normal'] if ev['mean_D_normal'] > 0 else float('inf')
    print(f"  Mean D lift:      {d_lift:.2f}×\n")

    print(f"  AUROC                  : {ev['auroc']}")
    print(f"  Accuracy (δ={cal_delta:.2f})   : {ev['accuracy']:.1%}")
    print(f"  Fisher's exact p       : {ev['p_value']:.2e}")
    print(f"  Contingency            : {ev['contingency']}\n")

    print(f"  TRIGGER_REPLAN normal      : {ev['trigger_replan_normal_pct']}%")
    print(f"  TRIGGER_REPLAN contradiction: {ev['trigger_replan_contradiction_pct']}%")
    lift = (ev['trigger_replan_contradiction_pct'] / ev['trigger_replan_normal_pct']
            if ev['trigger_replan_normal_pct'] > 0 else float('inf'))
    print(f"  TRIGGER_REPLAN lift        : {lift:.2f}×\n")

    confirmed = ev["auroc"] > 0.70 and ev["p_value"] < 0.05 and ev["mean_D_contra"] > ev["mean_D_normal"]
    if confirmed:
        print(f"  ✓ HYPOTHESIS CONFIRMED (AUROC={ev['auroc']}, p={ev['p_value']:.2e})")
        print("    Content-blind routing (V4) detects synthetic clinical contradictions.")
        print("    route() saw only D — never v_A or v_B.")
        print("    Theorem 1 (Safety-Learning Equivalence) empirically validated.")
    else:
        issues = []
        if ev["auroc"] <= 0.70: issues.append(f"AUROC={ev['auroc']}")
        if ev["p_value"] >= 0.05: issues.append(f"p={ev['p_value']:.2e}")
        if ev["mean_D_contra"] <= ev["mean_D_normal"]: issues.append("D contra ≤ normal")
        print(f"  ~ NOT FULLY CONFIRMED: {'; '.join(issues)}")

    print("="*70)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    random.seed(SEED); np.random.seed(SEED)

    print("\n" + "="*70)
    print("  Option C2: BiomedCLIP 18-finding | Synthetic GT | Cal/Test split")
    print("="*70 + "\n")

    samples = load_dataset(str(IMAGES_DIR), str(REPORTS_DIR))
    random.shuffle(samples)
    print(f"[dataset] {len(samples)} paired samples\n")

    print("[oracle] Loading TXV DenseNet-121 (ground-truth construction only)...")
    txv_oracle = TXVOracle(device=DEVICE)

    print("[model] Loading BiomedCLIP...")
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(MODEL_ID)
    tokenizer = open_clip.get_tokenizer(MODEL_ID)
    model = model.to(DEVICE).eval()
    enc = FindingVectorEncoder18(model, preprocess_val, tokenizer, device=DEVICE)

    # Build pairs using TXV oracle for ground truth
    normal_pairs, contra_pairs = build_synthetic_pairs(
        samples, txv_oracle, N_NORMAL, N_CONTRA, POOL_SIZE, SEED
    )

    # Stratified 50/50 split
    rng = random.Random(SEED + 1)
    rng.shuffle(normal_pairs); rng.shuffle(contra_pairs)
    split_n = len(normal_pairs) // 2; split_c = len(contra_pairs) // 2
    cal_pairs  = normal_pairs[:split_n] + contra_pairs[:split_c]
    test_pairs = normal_pairs[split_n:] + contra_pairs[split_c:]
    rng.shuffle(cal_pairs); rng.shuffle(test_pairs)
    print(f"\n[split] Calibration: {len(cal_pairs)} | Test: {len(test_pairs)}")

    print("\n[routing] Calibration set...")
    cal_records = run_routing(cal_pairs, enc, TAU_HIGH, TAU_LOW, DELTA_INIT, "CAL")

    cal_delta, cal_acc, cal_tau, cal_cov = calibrate(cal_records)
    print(f"\n[calibration] δ={cal_delta}  acc={cal_acc:.1%}  τ_high={cal_tau}  cov={cal_cov:.0%}\n")

    print("[routing] Test set (held out)...")
    test_records = run_routing(test_pairs, enc, cal_tau, TAU_LOW, cal_delta, "TEST")

    test_eval = evaluate(test_records, cal_delta, cal_tau, TAU_LOW)
    print_results(test_eval, cal_delta, cal_tau)

    output = {
        "experiment": "nlm_cxr_divergence_option_c2",
        "timestamp":  datetime.datetime.utcnow().isoformat() + "Z",
        "option":     "C2 — BiomedCLIP 18-finding vectors, synthetic GT, cal/test split",
        "findings":   FINDINGS_18,
        "config": {
            "encoder":    MODEL_ID,
            "gt_oracle":  "TXV densenet121-res224-all (GT only, not in routing)",
            "tau_high":   cal_tau,   "tau_low": TAU_LOW,  "delta": cal_delta,
            "n_normal":   N_NORMAL,  "n_contra": N_CONTRA, "pool_size": POOL_SIZE,
            "device":     DEVICE,
        },
        "calibration":    {"n": len(cal_records), "delta": cal_delta, "accuracy": cal_acc, "tau": cal_tau},
        "test_evaluation": test_eval,
        "cal_records":    cal_records,
        "test_records":   test_records,
    }
    out_path = RESULTS_DIR / "experiment_results_c2.json"
    with open(out_path, "w") as fh:
        json.dump(output, fh, indent=2)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
