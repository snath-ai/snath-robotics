"""
DAS Paper — Tier 1 Experiments (run_tier1.py)
=============================================
Three directions from the DAS paper researcher review:

  Direction 11 — Vocabulary size ablation   (K = 5 → 9 → 18 → 36 findings)
  Direction 12 — Calibrated fusion baseline  (routing L1 vs CLS-cosine)
  Direction 13 — Geometry of Δ-space         (k-means + silhouette + UMAP on D_hard)

Architecture (same as Option C2):
  Ground truth (synthetic, model-defined):
    Normal:       (image_i, text_i)  — own report
    Contradiction:(image_i, text_j)  — text from most TXV-distant patient j
    TXV DenseNet-121 used ONLY for GT construction (V4 preserved — route() never sees it)

  Routing signals:
    v_A[k] = P(finding_k present | image)  — BiomedCLIP image zero-shot
    v_B[k] = P(finding_k present | text)   — BiomedCLIP text  zero-shot
    D_routing = L1(v_A, v_B) / K           — normalised per-finding disagreement

  Fusion signal (Direction 12 only):
    z_A = BiomedCLIP CLS embedding of image  (512-dim, L2-normalised)
    z_B = BiomedCLIP CLS embedding of text   (512-dim, L2-normalised)
    D_fusion = 1 − cosine_sim(z_A, z_B)      — global embedding distance

  All encoding is done once; vocab projections are fast matrix multiplies.
  Cache: data/tier1_cache.pt

Run:
    cd /path/to/lar_divergence_exp
    USE_TF=0 TOKENIZERS_PARALLELISM=false python3 run_tier1.py [--fast]

Author: Aadithya Vishnu Sajeev — May 2026
"""

import os, sys, json, random, datetime, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import open_clip
from PIL import Image as PILImage
from scipy.stats import fisher_exact
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.cluster import KMeans
import torchxrayvision as xrv
import torchvision.transforms as T

# ── Repo paths ────────────────────────────────────────────────────────────────
_HERE     = Path(__file__).parent.resolve()
_PLAY     = _HERE.parent.parent.parent.parent  # DAS/lar_divergence_exp -> Snath Robotics/experiments/DAS -> experiments -> Snath Robotics -> JEPA_Playground
_LAR_JEPA = _PLAY / "lar_jepa"
_LAR_SRC  = _LAR_JEPA / "lar_jepa" / "src"

for _p in [str(_LAR_JEPA), str(_LAR_SRC), str(_HERE), str(_HERE.parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data.dataset import load_dataset   # noqa: E402

DATA_DIR    = _HERE / "data"
IMAGES_DIR  = DATA_DIR / "images"
REPORTS_DIR = DATA_DIR / "reports"
RESULTS_DIR = _HERE / "results"
CACHE_FILE  = DATA_DIR / "tier1_cache.pt"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE     = "mps" if torch.backends.mps.is_available() else "cpu"
SEED       = 42
N_NORMAL   = 100
N_CONTRA   = 100
POOL_SIZE  = 800     # samples for TXV oracle GT construction
TAU_HIGH   = 0.70
TAU_LOW    = 0.10
DELTA_INIT = 2.0

MODEL_ID   = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

# ── Vocabularies ──────────────────────────────────────────────────────────────

VOCAB_5 = [
    "atelectasis",
    "cardiomegaly",
    "pleural effusion",
    "pneumothorax",
    "consolidation",
]

VOCAB_9 = VOCAB_5 + [
    "pneumonia",
    "pulmonary edema",
    "emphysema",
    "pulmonary fibrosis",
]

VOCAB_18 = [
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

_VOCAB_EXTRA_18 = [
    "ground glass opacity",
    "hyperinflation",
    "air trapping",
    "bronchiectasis",
    "pleural plaques",
    "hilar enlargement",
    "tracheal deviation",
    "mediastinal mass",
    "pulmonary hypertension",
    "aortic calcification",
    "diaphragm flattening",
    "costophrenic angle blunting",
    "peribronchial thickening",
    "air bronchogram",
    "bilateral infiltrates",
    "lobar collapse",
    "pulmonary congestion",
    "pericardial effusion",
]

VOCAB_36 = VOCAB_18 + _VOCAB_EXTRA_18

VOCABS = {5: VOCAB_5, 9: VOCAB_9, 18: VOCAB_18, 36: VOCAB_36}
VOCAB_SIZES = [5, 9, 18, 36]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. TXV ORACLE — ground truth construction only
# ═══════════════════════════════════════════════════════════════════════════════

class TXVOracle:
    def __init__(self, device=DEVICE):
        self.device = device
        self.model = xrv.models.DenseNet(
            weights="densenet121-res224-all"
        ).to(device).eval()
        self._tf = T.Compose([
            T.Grayscale(num_output_channels=1),
            T.Resize((224, 224)),
            T.ToTensor(),
        ])

    @torch.no_grad()
    def predict(self, image: PILImage.Image) -> np.ndarray:
        t = self._tf(image).unsqueeze(0).to(self.device)
        t = t * 2048 - 1024
        out = self.model(t).squeeze(0).clamp(0, 1)
        out[torch.isnan(out)] = 0.5
        return out.cpu().numpy()


def build_synthetic_pairs(samples, oracle, n_normal, n_contra, pool_size, seed):
    rng = random.Random(seed)
    pool = list(samples[:pool_size])
    rng.shuffle(pool)

    print(f"[GT oracle] TXV predictions for {len(pool)} samples...")
    txv_vecs, valid_pool = [], []
    for i, s in enumerate(pool):
        try:
            img = PILImage.open(s.image_path).convert("RGB")
            txv_vecs.append(oracle.predict(img))
            valid_pool.append(s)
        except Exception:
            pass
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(pool)}")

    txv_vecs = np.array(txv_vecs)
    N = len(valid_pool)
    print(f"[GT oracle] Valid: {N}  matrix: {txv_vecs.shape}")

    half = N // 2
    normal_pool  = list(range(half))
    contra_pool  = list(range(half, N))
    partner_pool = list(range(half))

    rng.shuffle(normal_pool)
    rng.shuffle(contra_pool)

    normal_pairs = []
    for idx in normal_pool[:n_normal]:
        s = valid_pool[idx]
        text = (s.findings + " " + s.impression).strip() or "normal chest radiograph"
        normal_pairs.append({
            "img_path": str(s.image_path), "text": text,
            "label": "normal", "txv_l1_gt": 0.0,
        })

    contra_pairs = []
    for idx in contra_pool:
        if len(contra_pairs) >= n_contra:
            break
        v_img = txv_vecs[idx]
        l1s   = np.abs(txv_vecs[partner_pool] - v_img).sum(axis=1)
        best  = partner_pool[int(np.argmax(l1s))]
        s_img, s_txt = valid_pool[idx], valid_pool[best]
        text  = (s_txt.findings + " " + s_txt.impression).strip() or "normal chest radiograph"
        contra_pairs.append({
            "img_path": str(s_img.image_path), "text": text,
            "label": "contradiction", "txv_l1_gt": round(float(l1s.max()), 4),
        })

    print(f"[GT oracle] {len(normal_pairs)} normal + {len(contra_pairs)} contra pairs")
    if contra_pairs:
        print(f"  Mean TXV L1 (oracle selection): "
              f"{np.mean([p['txv_l1_gt'] for p in contra_pairs]):.3f}")
    return normal_pairs, contra_pairs


# ═══════════════════════════════════════════════════════════════════════════════
# 2. BIOMEDCLIP ENCODING
# ═══════════════════════════════════════════════════════════════════════════════

def encode_all_pairs_cls(
    pairs,
    model,
    preprocess_val,
    tokenizer,
    device=DEVICE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Encode all pairs → raw CLS embeddings.
    Returns:
        img_cls : (N, 512)  image L2-normalised CLS
        txt_cls : (N, 512)  text  L2-normalised CLS
    """
    img_cls, txt_cls = [], []
    for i, p in enumerate(pairs):
        img = PILImage.open(p["img_path"]).convert("RGB")
        img_t = preprocess_val(img).unsqueeze(0).to(device)
        with torch.no_grad():
            img_emb = F.normalize(model.encode_image(img_t).float(), dim=-1)
        img_cls.append(img_emb.cpu().squeeze(0))

        tok = tokenizer([p["text"]]).to(device)
        with torch.no_grad():
            txt_emb = F.normalize(model.encode_text(tok).float(), dim=-1)
        txt_cls.append(txt_emb.cpu().squeeze(0))

        if (i + 1) % 50 == 0:
            print(f"  encoded {i+1}/{len(pairs)}")

    return torch.stack(img_cls), torch.stack(txt_cls)


def cls_to_finding_vectors(
    img_cls: torch.Tensor,   # (N, 512)
    txt_cls: torch.Tensor,   # (N, 512)
    vocabulary: list[str],
    model,
    tokenizer,
    device=DEVICE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Project CLS embeddings to K-dim finding-probability vectors.

    For each finding f_k:
        p_k = softmax(logit_scale × [sim(emb, present_k), sim(emb, absent_k)])[0]

    Fast: CLS already computed — only K template text encodings needed.
    Returns:
        v_A: (N, K)  image finding probabilities
        v_B: (N, K)  text  finding probabilities
    """
    K = len(vocabulary)
    present_texts = [f"chest X-ray showing {f}"    for f in vocabulary]
    absent_texts  = [f"normal chest X-ray, no {f}" for f in vocabulary]

    with torch.no_grad():
        tok_p = tokenizer(present_texts).to(device)
        tok_a = tokenizer(absent_texts).to(device)
        emb_p = F.normalize(model.encode_text(tok_p).float(), dim=-1).cpu()  # (K, 512)
        emb_a = F.normalize(model.encode_text(tok_a).float(), dim=-1).cpu()  # (K, 512)

    logit_scale = float(model.logit_scale.exp().cpu().item())

    def project(cls: torch.Tensor) -> torch.Tensor:
        # cls: (N, 512)  →  finding_probs: (N, K)
        sim_p = cls @ emb_p.T * logit_scale  # (N, K)
        sim_a = cls @ emb_a.T * logit_scale  # (N, K)
        logits = torch.stack([sim_p, sim_a], dim=-1)  # (N, K, 2)
        return logits.softmax(dim=-1)[:, :, 0]         # (N, K)  present prob

    v_A = project(img_cls)   # (N, K)
    v_B = project(txt_cls)   # (N, K)
    return v_A, v_B


# ═══════════════════════════════════════════════════════════════════════════════
# 3. ROUTING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def snr_confidence(v: torch.Tensor) -> np.ndarray:
    """(N, K) → (N,) confidence ∈ [0,1]. Far from 0.5 = high confidence."""
    dev = (v - 0.5).abs().mean(dim=1)
    return torch.sigmoid((dev - 0.15) * 10).clamp(0, 1).numpy()


def l1_divergence(v_A: torch.Tensor, v_B: torch.Tensor) -> np.ndarray:
    """Normalised L1: (N, K), (N, K) → (N,)  D ∈ [0,1]."""
    K = v_A.shape[1]
    return (v_A - v_B).abs().sum(dim=1).numpy() / K


def route_decision(c_a, c_b, D, tau_h, tau_l, delta) -> str:
    bh = c_a >= tau_h and c_b >= tau_h
    bl = c_a <  tau_l and c_b <  tau_l
    if bh and D >= delta: return "TRIGGER_REPLAN"
    if bh and D <  delta: return "COMMIT_TRAJECTORY"
    if bl:                return "STRUCTURAL_IMPASSE"
    return "COMMIT_TRAJECTORY"


# ═══════════════════════════════════════════════════════════════════════════════
# 4. CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════════

def calibrate_delta(
    D_calib: np.ndarray,
    y_calib: np.ndarray,
    n_steps: int = 200,
) -> tuple[float, float]:
    """Grid-search delta that maximises balanced accuracy on calib set."""
    best_delta, best_ba = float(np.percentile(D_calib, 50)), -1.0
    for pct in np.linspace(5, 95, n_steps):
        d = float(np.percentile(D_calib, pct))
        preds = (D_calib >= d).astype(float)
        if preds.sum() == 0 or preds.sum() == len(preds): continue
        pos = float((y_calib == 1).sum()); neg = float((y_calib == 0).sum())
        if pos == 0 or neg == 0: continue
        tp = float(((preds == 1) & (y_calib == 1)).sum())
        tn = float(((preds == 0) & (y_calib == 0)).sum())
        ba = 0.5 * (tp / pos + tn / neg)
        if ba > best_ba:
            best_ba = ba
            best_delta = d
    return best_delta, float(best_ba)


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    if labels.sum() < 2 or (labels == 0).sum() < 2: return float("nan")
    try:    return float(roc_auc_score(labels, scores))
    except: return float("nan")


def accuracy_at_threshold(D: np.ndarray, y: np.ndarray, delta: float) -> float:
    preds = (D >= delta).astype(float)
    return float((preds == y).mean())


# ═══════════════════════════════════════════════════════════════════════════════
# 5. DIRECTION 11 — Vocabulary Size Ablation
# ═══════════════════════════════════════════════════════════════════════════════

def run_direction_11(
    img_cls: torch.Tensor,   # (N, 512)
    txt_cls: torch.Tensor,   # (N, 512)
    y:       np.ndarray,     # (N,)
    calib_mask: np.ndarray,  # (N,) bool
    model, tokenizer,
) -> dict:
    print("\n" + "─" * 60)
    print("DIRECTION 11 — Vocabulary Size Ablation")
    print("─" * 60)

    test_mask = ~calib_mask
    y_test    = y[test_mask]
    results   = {}

    for K in VOCAB_SIZES:
        print(f"\n  K={K:2d}  ({VOCABS[K][0]}, {VOCABS[K][1]}, ...)")
        v_A, v_B = cls_to_finding_vectors(img_cls, txt_cls, VOCABS[K], model, tokenizer)

        D_all  = l1_divergence(v_A, v_B)
        D_cal  = D_all[calib_mask]
        D_test = D_all[test_mask]
        y_cal  = y[calib_mask]

        delta_cal, ba_cal = calibrate_delta(D_cal, y_cal)
        auc_test  = auroc(D_test, y_test)
        acc_test  = accuracy_at_threshold(D_test, y_test, delta_cal)

        print(f"    δ_calib={delta_cal:.4f}  bal_acc_cal={ba_cal:.3f}")
        print(f"    AUROC_test={auc_test:.4f}  acc_test={acc_test:.3f}")

        results[K] = {
            "K":             K,
            "example_findings": VOCABS[K][:5],
            "delta_calibrated": round(delta_cal, 4),
            "balanced_acc_calib": round(ba_cal, 4),
            "auroc_test":    round(auc_test, 4),
            "accuracy_test": round(acc_test, 4),
        }

    # Trend analysis
    aucs  = [results[K]["auroc_test"] for K in VOCAB_SIZES
             if not np.isnan(results[K]["auroc_test"])]
    valid_sizes = [K for K in VOCAB_SIZES
                   if not np.isnan(results[K]["auroc_test"])]
    trend = (
        "monotone_increase" if aucs == sorted(aucs) else
        "monotone_decrease" if aucs == sorted(aucs, reverse=True) else
        "non_monotone"
    )
    print(f"\n  Trend ({valid_sizes}): {trend}")
    for K in VOCAB_SIZES:
        a = results[K]["auroc_test"]
        print(f"    K={K:2d}: AUROC = {a:.4f}")

    return {
        "direction":         11,
        "description":       "Vocabulary size ablation: routing AUROC vs |vocabulary|",
        "n_calib":           int(calib_mask.sum()),
        "n_test":            int(test_mask.sum()),
        "n_positive_test":   int(y_test.sum()),
        "results_by_K":      results,
        "vocab_sizes_tested": VOCAB_SIZES,
        "trend":             trend,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 6. DIRECTION 12 — Calibrated Fusion Baseline
# ═══════════════════════════════════════════════════════════════════════════════

def run_direction_12(
    v_A18: torch.Tensor,   # (N, 18)  routing finding vectors
    v_B18: torch.Tensor,
    img_cls: torch.Tensor,  # (N, 512) CLS for fusion
    txt_cls: torch.Tensor,
    y: np.ndarray,
    calib_mask: np.ndarray,
) -> dict:
    print("\n" + "─" * 60)
    print("DIRECTION 12 — Calibrated Fusion Baseline")
    print("─" * 60)

    test_mask = ~calib_mask
    y_cal     = y[calib_mask]
    y_test    = y[test_mask]

    # Routing signal (normalised L1 of K=18 finding vectors)
    D_rout = l1_divergence(v_A18, v_B18)

    # Fusion signal (CLS cosine distance)
    D_fuse = (1 - F.cosine_similarity(img_cls, txt_cls, dim=-1)).numpy()

    # --- Calibrate on calib set ---
    d_rout_cal, ba_rout_cal = calibrate_delta(D_rout[calib_mask], y_cal)
    d_fuse_cal, ba_fuse_cal = calibrate_delta(D_fuse[calib_mask], y_cal)

    auroc_rout_cal = auroc(D_rout[calib_mask], y_cal)
    auroc_fuse_cal = auroc(D_fuse[calib_mask], y_cal)

    print(f"  Routing calib: AUROC={auroc_rout_cal:.4f}  δ={d_rout_cal:.4f}  ba={ba_rout_cal:.3f}")
    print(f"  Fusion  calib: AUROC={auroc_fuse_cal:.4f}  δ={d_fuse_cal:.4f}  ba={ba_fuse_cal:.3f}")

    # --- Evaluate on test set ---
    auroc_rout_test = auroc(D_rout[test_mask], y_test)
    auroc_fuse_test = auroc(D_fuse[test_mask], y_test)
    acc_rout_test   = accuracy_at_threshold(D_rout[test_mask], y_test, d_rout_cal)
    acc_fuse_test   = accuracy_at_threshold(D_fuse[test_mask], y_test, d_fuse_cal)

    delta_auroc = (auroc_rout_test - auroc_fuse_test
                   if not (np.isnan(auroc_rout_test) or np.isnan(auroc_fuse_test))
                   else float("nan"))

    print(f"\n  Test set: N={int(test_mask.sum())}  pos={int(y_test.sum())}")
    print(f"  Routing AUROC: {auroc_rout_test:.4f}  acc={acc_rout_test:.3f}")
    print(f"  Fusion  AUROC: {auroc_fuse_test:.4f}  acc={acc_fuse_test:.3f}")
    print(f"  ΔAUROC (routing − fusion): {delta_auroc:+.4f}")

    # Mean D by label on test set (signal strength)
    D_rout_test = D_rout[test_mask]
    D_fuse_test = D_fuse[test_mask]
    mean_d_rout = {
        "normal": float(D_rout_test[y_test == 0].mean()),
        "contra": float(D_rout_test[y_test == 1].mean()),
    }
    mean_d_fuse = {
        "normal": float(D_fuse_test[y_test == 0].mean()),
        "contra": float(D_fuse_test[y_test == 1].mean()),
    }
    lift_rout = (mean_d_rout["contra"] / mean_d_rout["normal"]
                 if mean_d_rout["normal"] > 0 else float("inf"))
    lift_fuse = (mean_d_fuse["contra"] / mean_d_fuse["normal"]
                 if mean_d_fuse["normal"] > 0 else float("inf"))

    print(f"  Routing D — normal: {mean_d_rout['normal']:.4f}  "
          f"contra: {mean_d_rout['contra']:.4f}  lift: {lift_rout:.2f}×")
    print(f"  Fusion  D — normal: {mean_d_fuse['normal']:.4f}  "
          f"contra: {mean_d_fuse['contra']:.4f}  lift: {lift_fuse:.2f}×")

    return {
        "direction":   12,
        "description": "Calibrated fusion baseline: routing L1 vs BiomedCLIP CLS cosine",
        "n_calib":     int(calib_mask.sum()),
        "n_test":      int(test_mask.sum()),
        "n_positive_test": int(y_test.sum()),
        "routing": {
            "signal":            "L1(v_A18, v_B18) / 18",
            "auroc_calib":       round(auroc_rout_cal, 4),
            "delta_calibrated":  round(d_rout_cal, 4),
            "balanced_acc_calib": round(ba_rout_cal, 4),
            "auroc_test":        round(auroc_rout_test, 4),
            "accuracy_test":     round(acc_rout_test, 4),
            "mean_D_normal":     round(mean_d_rout["normal"], 4),
            "mean_D_contra":     round(mean_d_rout["contra"], 4),
            "D_lift":            round(lift_rout, 3),
        },
        "fusion": {
            "signal":            "1 - cosine_sim(z_A, z_B)",
            "auroc_calib":       round(auroc_fuse_cal, 4),
            "tau_calibrated":    round(d_fuse_cal, 4),
            "balanced_acc_calib": round(ba_fuse_cal, 4),
            "auroc_test":        round(auroc_fuse_test, 4),
            "accuracy_test":     round(acc_fuse_test, 4),
            "mean_D_normal":     round(mean_d_fuse["normal"], 4),
            "mean_D_contra":     round(mean_d_fuse["contra"], 4),
            "D_lift":            round(lift_fuse, 3),
        },
        "delta_auroc_routing_minus_fusion": round(delta_auroc, 4),
        "interpretation": (
            "ΔAUROC > 0: routing (per-finding L1) detects disagreements more reliably "
            "than raw CLS-cosine distance — the central DAS claim. "
            "Both methods use the same BiomedCLIP model; only the scoring function differs."
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 7. DIRECTION 13 — Geometry of Δ-Space
# ═══════════════════════════════════════════════════════════════════════════════

def run_direction_13(
    v_A18: torch.Tensor,   # (N, 18)
    v_B18: torch.Tensor,
    y:     np.ndarray,
    pairs: list[dict],
    calib_mask: np.ndarray,
    delta_cal: float,
) -> dict:
    print("\n" + "─" * 60)
    print("DIRECTION 13 — Geometry of Δ-Space")
    print("─" * 60)

    c_A = snr_confidence(v_A18)
    c_B = snr_confidence(v_B18)
    D   = l1_divergence(v_A18, v_B18)

    # D_hard = TRIGGER_REPLAN cases using calibrated delta
    d_hard_mask = np.array([
        route_decision(float(c_A[i]), float(c_B[i]), float(D[i]),
                       TAU_HIGH, TAU_LOW, delta_cal) == "TRIGGER_REPLAN"
        for i in range(len(D))
    ])
    n_d_hard = int(d_hard_mask.sum())
    print(f"  D_hard (TRIGGER_REPLAN, δ={delta_cal:.4f}): {n_d_hard}/{len(D)}")

    # Ensure enough points — fallback to 70th percentile if too few
    _delta_used = delta_cal
    if n_d_hard < 20:
        _delta_used = float(np.percentile(D, 70))
        d_hard_mask = np.array([
            route_decision(float(c_A[i]), float(c_B[i]), float(D[i]),
                           TAU_HIGH, TAU_LOW, _delta_used) == "TRIGGER_REPLAN"
            for i in range(len(D))
        ])
        n_d_hard = int(d_hard_mask.sum())
        print(f"  Adjusted to 70th-pct δ={_delta_used:.4f}: {n_d_hard} D_hard cases")

    # Δ_i = v_A_i − v_B_i ∈ ℝ^18  (signed per-finding disagreement direction)
    delta_vecs = (v_A18 - v_B18)[d_hard_mask].numpy()   # (n_hard, 18)
    y_d_hard   = y[d_hard_mask]

    print(f"  Δ-space shape: {delta_vecs.shape}")
    print(f"  GT positive rate in D_hard: "
          f"{y_d_hard.mean():.3f}  (overall: {y.mean():.3f})")
    enrichment = (y_d_hard.mean() / y.mean() if y.mean() > 0 else float("nan"))
    print(f"  Enrichment: {enrichment:.2f}×")

    # k-means for k ∈ {3, 4, 5}
    kmeans_results = {}
    best_k, best_sil, best_labels = 3, -1.0, None
    for k in [3, 4, 5]:
        if n_d_hard < k + 2: continue
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
        labels = km.fit_predict(delta_vecs)
        sil = (float(silhouette_score(delta_vecs, labels))
               if len(np.unique(labels)) > 1 else -1.0)

        # Top-3 findings driving each cluster
        centroids = km.cluster_centers_   # (k, 18)
        cluster_info = {}
        for ci in range(k):
            top_pos = np.argsort(centroids[ci])[::-1][:3]  # image overestimates
            top_neg = np.argsort(centroids[ci])[:3]         # image underestimates
            cluster_info[f"c{ci}"] = {
                "size":               int((labels == ci).sum()),
                "image_overestimates":  [VOCAB_18[j] for j in top_pos],
                "image_underestimates": [VOCAB_18[j] for j in top_neg],
                "silhouette_k":        round(sil, 4),
            }
        kmeans_results[k] = {"silhouette": round(sil, 4), "clusters": cluster_info}
        print(f"  k={k}: silhouette={sil:.4f}")

        if sil > best_sil:
            best_sil = sil; best_k = k; best_labels = labels.copy()

    # UMAP projection
    umap_coords, umap_available = None, False
    try:
        import umap as umap_lib
        n_neighbors = min(15, n_d_hard - 1)
        reducer = umap_lib.UMAP(n_components=2, random_state=SEED,
                                n_neighbors=n_neighbors)
        umap_coords   = reducer.fit_transform(delta_vecs).tolist()
        umap_available = True
        print(f"  UMAP: 2D projection complete ({len(umap_coords)} points)")
    except Exception as e:
        print(f"  UMAP: not available ({e})")

    # Axis interpretation: mean |Δ_k| per finding (which findings cause replan?)
    mean_abs_delta = np.abs(delta_vecs).mean(axis=0)   # (18,)
    top_findings_by_delta = [VOCAB_18[j] for j in np.argsort(mean_abs_delta)[::-1][:5]]
    print(f"  Top-5 findings driving TRIGGER_REPLAN: {top_findings_by_delta}")

    return {
        "direction":   13,
        "description": "Geometry of Δ-space: k-means + silhouette + UMAP on D_hard",
        "delta_used":  round(_delta_used, 4),
        "n_total":     len(D),
        "n_d_hard":    n_d_hard,
        "gt_enrichment_factor": round(float(enrichment), 3),
        "gt_positive_rate_d_hard": round(float(y_d_hard.mean()), 3),
        "gt_positive_rate_overall": round(float(y.mean()), 3),
        "top_findings_driving_replan": top_findings_by_delta,
        "mean_abs_delta_per_finding": {
            VOCAB_18[j]: round(float(mean_abs_delta[j]), 4) for j in range(18)
        },
        "best_k":          best_k,
        "best_silhouette": round(float(best_sil), 4),
        "kmeans_by_k":     kmeans_results,
        "umap_available":  umap_available,
        "n_umap_points":   len(umap_coords) if umap_coords else 0,
        "umap_2d":         umap_coords,
        "interpretation": (
            "Δ_i = v_A_i − v_B_i is the signed per-finding disagreement vector. "
            "Positive Δ_k: image claims finding_k more than report. "
            "Clusters in Δ-space ≈ clinically coherent modes of disagreement. "
            "Enrichment > 1 confirms D_hard cases are genuinely contradictory pairs, "
            "not routing noise."
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 8. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main(fast: bool = False):
    t0 = time.time()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    pool_size = 300 if fast else POOL_SIZE
    n_normal  = 50  if fast else N_NORMAL
    n_contra  = 50  if fast else N_CONTRA

    print("\n" + "=" * 70)
    print("  DAS Tier 1 Experiments — Directions 11, 12, 13")
    print(f"  Device: {DEVICE}   |   Fast mode: {fast}")
    print("=" * 70 + "\n")

    # ── Step 1: Load dataset ─────────────────────────────────────────────────
    print("[data] Loading NLM CXR dataset...")
    samples = load_dataset(str(IMAGES_DIR), str(REPORTS_DIR))
    print(f"[data] {len(samples)} paired samples\n")

    # ── Step 2: Ground truth via TXV oracle ──────────────────────────────────
    print("[oracle] Loading TXV DenseNet-121 (GT construction only)...")
    oracle = TXVOracle(device=DEVICE)
    normal_pairs, contra_pairs = build_synthetic_pairs(
        samples, oracle, n_normal, n_contra, pool_size, SEED
    )
    del oracle  # free memory

    # Stratified 50/50 cal/test split
    rng = random.Random(SEED + 1)
    rng.shuffle(normal_pairs); rng.shuffle(contra_pairs)
    sn = len(normal_pairs) // 2; sc = len(contra_pairs) // 2
    cal_pairs  = normal_pairs[:sn] + contra_pairs[:sc]
    test_pairs = normal_pairs[sn:] + contra_pairs[sc:]
    rng.shuffle(cal_pairs); rng.shuffle(test_pairs)

    all_pairs = cal_pairs + test_pairs
    N = len(all_pairs)
    y = np.array([1.0 if p["label"] == "contradiction" else 0.0
                  for p in all_pairs], dtype=np.float32)
    calib_mask = np.zeros(N, dtype=bool)
    calib_mask[:len(cal_pairs)] = True

    print(f"\n[split] Cal: {len(cal_pairs)}  Test: {len(test_pairs)}")
    print(f"[split] y=1 (contradiction): {int(y.sum())} / {N}\n")

    # ── Step 3: BiomedCLIP encoding (with cache) ─────────────────────────────
    cache_valid = False
    if CACHE_FILE.exists() and not fast:
        try:
            cache = torch.load(CACHE_FILE, map_location="cpu", weights_only=False)
            if cache.get("N") == N and cache.get("seed") == SEED:
                img_cls = cache["img_cls"]
                txt_cls = cache["txt_cls"]
                cache_valid = True
                print(f"[cache] Loaded CLS embeddings from {CACHE_FILE}")
        except Exception as e:
            print(f"[cache] Could not load ({e}) — re-encoding")

    if not cache_valid:
        print("[model] Loading BiomedCLIP...")
        model, _, preprocess_val = open_clip.create_model_and_transforms(MODEL_ID)
        tokenizer = open_clip.get_tokenizer(MODEL_ID)
        model = model.to(DEVICE).eval()
        print("[model] BiomedCLIP loaded.\n")

        print(f"[encoding] {N} pairs (image + text)...")
        img_cls, txt_cls = encode_all_pairs_cls(
            all_pairs, model, preprocess_val, tokenizer, DEVICE
        )
        print(f"[encoding] img_cls: {img_cls.shape}  txt_cls: {txt_cls.shape}\n")

        if not fast:
            torch.save({"img_cls": img_cls, "txt_cls": txt_cls, "N": N, "seed": SEED},
                       CACHE_FILE)
            print(f"[cache] Saved CLS embeddings to {CACHE_FILE}")
    else:
        # Load model just for template projection (fast, no image/text encoding)
        print("[model] Loading BiomedCLIP for template projection...")
        model, _, _ = open_clip.create_model_and_transforms(MODEL_ID)
        tokenizer   = open_clip.get_tokenizer(MODEL_ID)
        model = model.to(DEVICE).eval()
        print("[model] BiomedCLIP loaded.\n")

    # ── Step 4: Pre-compute K=18 finding vectors (used by D12 and D13) ───────
    print("[finding_vecs] Computing K=18 finding vectors...")
    v_A18, v_B18 = cls_to_finding_vectors(
        img_cls, txt_cls, VOCAB_18, model, tokenizer
    )
    print(f"  v_A18: {v_A18.shape}  v_B18: {v_B18.shape}\n")

    # ── Step 5: Direction 12 (fusion baseline, uses K=18 + CLS) ──────────────
    result_12 = run_direction_12(v_A18, v_B18, img_cls, txt_cls, y, calib_mask)

    # ── Step 6: Direction 11 (vocab ablation) ────────────────────────────────
    result_11 = run_direction_11(img_cls, txt_cls, y, calib_mask, model, tokenizer)

    # ── Step 7: Direction 13 (Δ-space geometry) ──────────────────────────────
    # Use calibrated delta from Direction 12
    delta_cal = result_12["routing"]["delta_calibrated"]
    result_13 = run_direction_13(v_A18, v_B18, y, all_pairs, calib_mask, delta_cal)

    # ── Save results ──────────────────────────────────────────────────────────
    ts = datetime.datetime.utcnow().isoformat() + "Z"
    for direction, result in [(11, result_11), (12, result_12), (13, result_13)]:
        result["timestamp"] = ts
        result["fast_mode"] = fast
        outpath = RESULTS_DIR / f"tier1_dir{direction}.json"
        with open(outpath, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"\n[saved] {outpath}")

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print("  DAS Tier 1 Summary")
    print("=" * 70)
    print(f"  Completed in {elapsed/60:.1f} min  |  N={N}  pos={int(y.sum())}")
    print()
    print("  Direction 12 — Calibrated Fusion Baseline")
    print(f"    Routing AUROC (test):  {result_12['routing']['auroc_test']:.4f}")
    print(f"    Fusion  AUROC (test):  {result_12['fusion']['auroc_test']:.4f}")
    print(f"    ΔAUROC:                {result_12['delta_auroc_routing_minus_fusion']:+.4f}")
    print(f"    D lift — routing: {result_12['routing']['D_lift']:.2f}×  "
          f"fusion: {result_12['fusion']['D_lift']:.2f}×")
    print()
    print("  Direction 11 — Vocabulary Ablation")
    for K in VOCAB_SIZES:
        a = result_11["results_by_K"][K]["auroc_test"]
        print(f"    K={K:2d}: AUROC = {a:.4f}")
    print(f"    Trend: {result_11['trend']}")
    print()
    print("  Direction 13 — Δ-Space Geometry")
    print(f"    D_hard cases:     {result_13['n_d_hard']} / {result_13['n_total']}")
    print(f"    Best silhouette:  {result_13['best_silhouette']:.4f}  "
          f"(k={result_13['best_k']})")
    print(f"    GT enrichment:    {result_13['gt_enrichment_factor']:.2f}×")
    print(f"    UMAP:             {'available' if result_13['umap_available'] else 'skipped'}")
    print(f"    Top findings:     {result_13['top_findings_driving_replan'][:3]}")
    print()
    print(f"  Results → {RESULTS_DIR}/tier1_dir{{11,12,13}}.json")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true",
                    help="Smoke-test: 300-sample pool, 50+50 pairs (under 5 min)")
    args = ap.parse_args()
    main(fast=args.fast)
