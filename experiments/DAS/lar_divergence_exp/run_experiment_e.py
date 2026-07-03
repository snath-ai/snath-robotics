"""
Experiment E — Fusion vs Routing: Same Model, Same Data, Different Scoring
===========================================================================
Uses the EXACT same 200 pairs as Experiment C2 (NLM CXR, synthetic GT).
Uses the EXACT same model (BiomedCLIP).
The ONLY difference: how we score each pair.

  Fusion    : 1 - cosine_similarity(CLS_image, CLS_text)
              This is what contrastive training (CLIP-style) optimises.
              Both encoders trained to pull matched pairs together → the
              raw CLS cosine encodes "do these look like they belong together?"
              It does NOT encode "which specific findings does one stream
              claim that the other denies?"

  Routing   : L1(finding_vector_image, finding_vector_text)
              The 18-dim finding vectors already computed in C2.
              Preserves directional disagreement per finding.
              route() never sees the vectors — only the scalar D.

Expected result: Routing AUROC >> Fusion AUROC, zero-shot, same data, same model.
This directly demonstrates that the routing scoring function — not the encoder,
not the dataset — is what provides the separation signal.

Run:
    cd /path/to/lar_divergence_exp
    USE_TF=0 TOKENIZERS_PARALLELISM=false python3 run_experiment_e.py
"""

import os, sys, json, datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import open_clip
from PIL import Image as PILImage
from sklearn.metrics import roc_auc_score
from scipy.stats import fisher_exact

_HERE     = Path(__file__).parent.resolve()
_PLAY     = _HERE.parent.parent.parent.parent  # DAS/lar_divergence_exp -> Snath Robotics/experiments/DAS -> experiments -> Snath Robotics -> JEPA_Playground
_LAR_JEPA = _PLAY / "lar_jepa"
_LAR_SRC  = _LAR_JEPA / "lar_jepa" / "src"
for _p in [str(_LAR_JEPA), str(_LAR_SRC), str(_HERE), str(_HERE.parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data.dataset import load_dataset

DEVICE     = "mps"
MODEL_ID   = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
DATA_DIR   = _HERE / "data"
IMAGES_DIR = DATA_DIR / "images"
REPORTS_DIR= DATA_DIR / "reports"
RESULTS_DIR= _HERE / "results"
C2_PATH    = RESULTS_DIR / "experiment_results_c2.json"
OUT_PATH   = RESULTS_DIR / "experiment_results_e.json"

FINDINGS_18 = [
    "atelectasis","consolidation","infiltration","pneumothorax",
    "pulmonary edema","emphysema","pulmonary fibrosis","pleural effusion",
    "pneumonia","pleural thickening","cardiomegaly","pulmonary nodule",
    "pulmonary mass","diaphragmatic hernia","lung lesion","rib fracture",
    "lung opacity","enlarged cardiomediastinum",
]


def evaluate(scores, labels, name, higher_is_contradiction=True):
    if not higher_is_contradiction:
        scores = [-s for s in scores]
    scores = np.array(scores); labels = np.array(labels)
    auroc = float(roc_auc_score(labels, scores))
    thresholds = np.unique(scores)
    best_acc, best_thr = 0.0, np.median(scores)
    for thr in thresholds:
        acc = ((scores >= thr) == labels).mean()
        if acc > best_acc:
            best_acc, best_thr = acc, thr
    preds = (scores >= best_thr).astype(int)
    tp = int(((preds==1)&(labels==1)).sum()); fp = int(((preds==1)&(labels==0)).sum())
    fn = int(((preds==0)&(labels==1)).sum()); tn = int(((preds==0)&(labels==0)).sum())
    _, p_val = fisher_exact([[tp,fp],[fn,tn]], alternative="greater")
    return {
        "method": name, "auroc": round(auroc,4), "accuracy": round(best_acc,4),
        "p_value": float(p_val),
        "sensitivity": round(tp/(tp+fn),3) if (tp+fn)>0 else 0,
        "specificity": round(tn/(tn+fp),3) if (tn+fp)>0 else 0,
        "contingency": {"tp":tp,"fp":fp,"fn":fn,"tn":tn},
    }


def main():
    print("\n" + "="*68)
    print("  Experiment E — BiomedCLIP: Fusion (CLS cosine) vs Routing (L1)")
    print("  Same model · Same data · Different scoring function only")
    print("="*68 + "\n")

    # ── Load C2 test records ──────────────────────────────────────────────────
    with open(C2_PATH) as f:
        c2 = json.load(f)
    test_records = c2["test_records"]
    cal_records  = c2["cal_records"]

    # Routing scores already in C2
    routing_scores_test = [r["divergence"] for r in test_records]
    routing_labels_test = [1 if r["label"]=="contradiction" else 0 for r in test_records]
    routing_scores_cal  = [r["divergence"] for r in cal_records]
    routing_labels_cal  = [1 if r["label"]=="contradiction" else 0 for r in cal_records]

    # Build lookup from CXR ID to sample
    print("[dataset] Loading paired samples for image path lookup...")
    all_samples = load_dataset(str(IMAGES_DIR), str(REPORTS_DIR))
    by_id = {s.cxr_id: s for s in all_samples}

    # ── Load BiomedCLIP ───────────────────────────────────────────────────────
    print("[model] Loading BiomedCLIP...")
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_ID)
    tokenizer = open_clip.get_tokenizer(MODEL_ID)
    model = model.to(DEVICE).eval()
    print("[model] Loaded.\n")

    def encode_image_cls(image):
        img_t = preprocess(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            emb = F.normalize(model.encode_image(img_t).float(), dim=-1)
        return emb.squeeze(0)  # (512,)

    def encode_text_cls(text):
        tok = tokenizer([text]).to(DEVICE)
        with torch.no_grad():
            emb = F.normalize(model.encode_text(tok).float(), dim=-1)
        return emb.squeeze(0)  # (512,)

    def cosine_distance(e_a, e_b):
        return float(1.0 - (e_a * e_b).sum().item())

    # ── Encode all pairs ─────────────────────────────────────────────────────
    all_records_to_encode = list(enumerate(cal_records + test_records))
    total = len(all_records_to_encode)
    fusion_scores_all = []
    failures = 0

    print(f"[encoding] Computing CLS cosine distances for {total} pairs...")
    for i, (_, rec) in enumerate(all_records_to_encode):
        try:
            s_img = by_id.get(rec["img_cxr_id"])
            s_txt = by_id.get(rec["txt_cxr_id"])
            if s_img is None or s_txt is None:
                raise ValueError(f"CXR ID not found: {rec['img_cxr_id']} / {rec['txt_cxr_id']}")

            img  = PILImage.open(s_img.image_path).convert("RGB")
            text = (s_txt.findings + " " + s_txt.impression).strip()
            if not text:
                raise ValueError("Empty text")

            e_a   = encode_image_cls(img)
            e_b   = encode_text_cls(text)
            score = cosine_distance(e_a, e_b)
            fusion_scores_all.append(score)

            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{total}  cosine_dist={score:.4f}  label={rec['label']}")
        except Exception as ex:
            print(f"  ERROR {rec['img_cxr_id']}: {ex}")
            fusion_scores_all.append(None)
            failures += 1

    print(f"[encoding] Done. Failures: {failures}")

    # Split back into cal / test
    n_cal  = len(cal_records)
    fusion_cal  = fusion_scores_all[:n_cal]
    fusion_test = fusion_scores_all[n_cal:]

    # Drop failures
    cal_pairs  = [(s, l) for s, l in zip(fusion_cal,  routing_labels_cal)  if s is not None]
    test_pairs = [(s, l) for s, l in zip(fusion_test, routing_labels_test) if s is not None]

    fc_scores = [p[0] for p in test_pairs]; fc_labels = [p[1] for p in test_pairs]
    rt_scores = [routing_scores_test[i] for i, (s, _) in enumerate(zip(fusion_test, routing_labels_test)) if s is not None]
    rt_labels = fc_labels

    # ── Distribution summary ──────────────────────────────────────────────────
    fc_n = [s for s, l in test_pairs if l == 0]
    fc_c = [s for s, l in test_pairs if l == 1]
    rt_n = [routing_scores_test[i] for i, r in enumerate(test_records) if r["label"]=="normal"]
    rt_c = [routing_scores_test[i] for i, r in enumerate(test_records) if r["label"]=="contradiction"]

    print(f"\n  Fusion CLS cosine — normal μ={np.mean(fc_n):.4f}  contra μ={np.mean(fc_c):.4f}")
    print(f"  Routing L1       — normal μ={np.mean(rt_n):.4f}  contra μ={np.mean(rt_c):.4f}")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    res_fusion  = evaluate(fc_scores, fc_labels, "Fusion — BiomedCLIP CLS cosine (zero-shot)")
    res_routing = evaluate(rt_scores, rt_labels, "Routing — BiomedCLIP finding-vector L1 (zero-shot)")

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n" + "="*68)
    print("  EXPERIMENT E RESULTS — BiomedCLIP: Fusion vs Routing")
    print("  Dataset: NLM CXR (Indiana)  |  Ground truth: Synthetic (TXV oracle)")
    print("  Same model (BiomedCLIP). Same data. Different scoring only.")
    print("="*68)
    print(f"\n  {'Method':<50} {'AUROC':>6}  {'Acc':>6}  {'p':>12}")
    print("  " + "-"*74)
    for r in [res_fusion, res_routing]:
        print(f"  {r['method']:<50} {r['auroc']:>6.4f}  {r['accuracy']:>6.1%}  {r['p_value']:>12.2e}")

    delta = res_routing["auroc"] - res_fusion["auroc"]
    print(f"\n  Routing advantage: AUROC {delta:+.4f}  ({'+' if delta>0 else ''}{delta/max(res_fusion['auroc'],1e-6)*100:.0f}% relative)")

    print(f"\n  Mean D — fusion normal  : {np.mean(fc_n):.4f}")
    print(f"  Mean D — fusion contra  : {np.mean(fc_c):.4f}  (lift: {np.mean(fc_c)/np.mean(fc_n):.2f}×)")
    print(f"  Mean D — routing normal : {np.mean(rt_n):.4f}")
    print(f"  Mean D — routing contra : {np.mean(rt_c):.4f}  (lift: {np.mean(rt_c)/np.mean(rt_n):.2f}×)")

    if delta > 0.05:
        print(f"\n  ✓ ROUTING OUTPERFORMS FUSION (+{delta:.3f} AUROC, zero-shot)")
        print("    Same model. Same data. The architectural choice — routing vs fusion —")
        print("    is the source of the performance gap.")
        print("    Fusion CLS cosine discards directional disagreement per finding.")
        print("    Routing L1 preserves and amplifies it.")
    else:
        print(f"\n  ~ Gap < 0.05 AUROC. Check distribution overlap.")

    print("="*68)

    # ── Save ──────────────────────────────────────────────────────────────────
    output = {
        "experiment":    "fusion_vs_routing_same_model_nlm_cxr",
        "timestamp":     datetime.datetime.utcnow().isoformat() + "Z",
        "model":         MODEL_ID,
        "dataset":       "NLM Indiana University CXR",
        "ground_truth":  "Synthetic (TXV oracle, same as C2)",
        "note":          "Same model and data as C2. Only scoring function differs.",
        "results":       [res_fusion, res_routing],
        "distributions": {
            "fusion_normal_mean": round(float(np.mean(fc_n)), 4),
            "fusion_contra_mean": round(float(np.mean(fc_c)), 4),
            "routing_normal_mean": round(float(np.mean(rt_n)), 4),
            "routing_contra_mean": round(float(np.mean(rt_c)), 4),
        }
    }
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[saved] {OUT_PATH}")


if __name__ == "__main__":
    main()
