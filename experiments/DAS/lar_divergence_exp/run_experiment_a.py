"""
Option A — NLM CXR AbstractDivergenceRouter Experiment
=======================================================
BiomedCLIP (microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)

Stream A: BiomedCLIP image encoder on chest X-ray PNG  → 512-dim shared space
Stream B: BiomedCLIP text encoder on radiology report  → 512-dim shared space

Trained on 15M biomedical image-text pairs — cosine distance is semantically
calibrated. Agreeing pairs: D ≈ 0.2–0.5. Contradicting pairs: D ≈ 0.7–1.2.

This is the publication-grade validation of V1–V6 and Theorem 1 (Safety-Learning
Equivalence) from Disagreement as Signal (Sajeev 2026).

Run:
    cd /path/to/lar_divergence_exp
    USE_TF=0 TOKENIZERS_PARALLELISM=false python3 run_experiment_a.py

Results saved to results/experiment_results_a.json
"""

import os, sys, json, random, datetime
from pathlib import Path
from collections import Counter
from PIL import Image as PILImage

_HERE     = Path(__file__).parent.resolve()
_PLAY     = _HERE.parent.parent.parent.parent  # DAS/lar_divergence_exp -> Snath Robotics/experiments/DAS -> experiments -> Snath Robotics -> JEPA_Playground
_LAR_JEPA = _PLAY / "lar_jepa"
_LAR_SRC  = _LAR_JEPA / "lar_jepa" / "src"

for _p in [str(_LAR_JEPA), str(_LAR_SRC), str(_HERE), str(_HERE.parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.types import RouteDecision
from data.dataset import load_dataset, build_contradiction_subset
from encoders.cxr_encoders_a import (
    BiomedCLIPImageEncoder, BiomedCLIPTextEncoder, load_biomedclip
)
from router.divergence_router_a import BiomedCLIPDivergenceRouter

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE   = "mps"
N_SAMPLE = 100
SEED     = 42
# BiomedCLIP logit_scale is learned (~14 after softmax). Cosine distances are
# expected in [0.2, 0.5] for agreeing pairs and [0.5, 1.2] for contradicting.
TAU_HIGH = 0.75
TAU_LOW  = 0.20
DELTA    = 0.50   # separating threshold in BiomedCLIP cosine-distance space

DATA_DIR    = _HERE / "data"
IMAGES_DIR  = DATA_DIR / "images"
REPORTS_DIR = DATA_DIR / "reports"
RESULTS_DIR = _HERE / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ── Routing loop ──────────────────────────────────────────────────────────────

def run_routing(samples, label, router, img_enc, txt_enc, n):
    records = []
    for i, sample in enumerate(samples[:n]):
        try:
            image  = PILImage.open(sample.image_path).convert("RGB")
            text   = (sample.findings + " " + sample.impression).strip()
            if not text:
                continue

            z_a, c_a = img_enc.encode(image)
            z_b, c_b = txt_enc.encode(text)
            D        = router.divergence(z_a, z_b)
            decision = router.route(c_a, c_b, D)

            records.append({
                "cxr_id":     sample.cxr_id,
                "label":      label,
                "conf_a":     round(c_a, 4),
                "conf_b":     round(c_b, 4),
                "divergence": round(D, 4),
                "decision":   decision.value,
            })

            if (i + 1) % 10 == 0:
                print(f"  [{label}] {i+1}/{n}  "
                      f"c_A={c_a:.3f} c_B={c_b:.3f} D={D:.3f} → {decision.value}")

        except Exception as e:
            print(f"  [{label}] error on {sample.cxr_id}: {e}")

    return records


def summarise(records):
    by_label = {}
    for r in records:
        by_label.setdefault(r["label"], []).append(r)

    summary = {}
    for label, recs in by_label.items():
        decisions = Counter(r["decision"] for r in recs)
        summary[label] = {
            "n":                  len(recs),
            "mean_divergence":    round(sum(r["divergence"] for r in recs) / len(recs), 4),
            "mean_conf_a":        round(sum(r["conf_a"]     for r in recs) / len(recs), 4),
            "mean_conf_b":        round(sum(r["conf_b"]     for r in recs) / len(recs), 4),
            "decisions":          dict(decisions),
            "trigger_replan_pct": round(100 * decisions.get("TRIGGER_REPLAN",     0) / len(recs), 1),
            "commit_pct":         round(100 * decisions.get("COMMIT_TRAJECTORY",  0) / len(recs), 1),
            "impasse_pct":        round(100 * decisions.get("STRUCTURAL_IMPASSE", 0) / len(recs), 1),
        }
    return summary


def print_results(summary):
    print("\n" + "="*65)
    print("  EXPERIMENT A RESULTS — AbstractDivergenceRouter on NLM CXR")
    print("  (BiomedCLIP: true vision + language in shared space)")
    print("="*65)
    print(f"  τ_high={TAU_HIGH}  τ_low={TAU_LOW}  δ={DELTA}\n")

    for label, s in summary.items():
        print(f"  [{label.upper()}]  n={s['n']}")
        print(f"    Mean divergence D : {s['mean_divergence']}")
        print(f"    Mean conf_A (img) : {s['mean_conf_a']}")
        print(f"    Mean conf_B (txt) : {s['mean_conf_b']}")
        print(f"    Decisions         : {s['decisions']}")
        print(f"    TRIGGER_REPLAN %  : {s['trigger_replan_pct']}%")
        print(f"    COMMIT %          : {s['commit_pct']}%")
        print(f"    IMPASSE %         : {s['impasse_pct']}%\n")

    if "normal" in summary and "contradiction" in summary:
        n_rp = summary["normal"]["trigger_replan_pct"]
        c_rp = summary["contradiction"]["trigger_replan_pct"]
        n_D  = summary["normal"]["mean_divergence"]
        c_D  = summary["contradiction"]["mean_divergence"]
        print("  HYPOTHESIS CHECK")
        print(f"  Contradiction TRIGGER_REPLAN: {c_rp}%  (expected >> Normal)")
        print(f"  Normal        TRIGGER_REPLAN: {n_rp}%")
        print(f"  Mean D contradiction: {c_D}  >?  Normal: {n_D}")
        print()
        if c_rp > n_rp and c_D > n_D:
            print("  ✓ HYPOTHESIS CONFIRMED — content-blind routing detects")
            print("    structural contradiction without reading stream content.")
            print("    Safety-Learning Equivalence (V6) empirically validated.")
        elif c_rp > n_rp:
            print("  ~ PARTIAL — TRIGGER_REPLAN higher for contradictions.")
            print("    Consider tuning δ using distribution analysis.")
        else:
            print("  ✗ NOT CONFIRMED — check encoder output distributions.")
    print("="*65)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    random.seed(SEED)

    print("\n" + "="*65)
    print("  Snath AI — Option A: BiomedCLIP DivergenceRouter Experiment")
    print("="*65 + "\n")

    # Data (already extracted)
    print("[dataset] Loading paired samples...")
    samples = load_dataset(str(IMAGES_DIR), str(REPORTS_DIR))
    normal, contradiction = build_contradiction_subset(samples)
    random.shuffle(normal)
    random.shuffle(contradiction)
    print(f"[dataset] {N_SAMPLE} normal + {N_SAMPLE} contradiction\n")

    # Load BiomedCLIP (downloads ~800MB on first run)
    model, preprocess, tokenizer = load_biomedclip(device=DEVICE)

    img_enc = BiomedCLIPImageEncoder(model, preprocess, tokenizer, device=DEVICE)
    txt_enc = BiomedCLIPTextEncoder(model, tokenizer, device=DEVICE)
    router  = BiomedCLIPDivergenceRouter(
        tau_high=TAU_HIGH, tau_low=TAU_LOW, delta=DELTA, device=DEVICE
    )

    print("[routing] Normal cases...")
    normal_records = run_routing(normal, "normal", router, img_enc, txt_enc, N_SAMPLE)

    print("\n[routing] Contradiction cases...")
    contra_records = run_routing(contradiction, "contradiction", router, img_enc, txt_enc, N_SAMPLE)

    all_records = normal_records + contra_records
    summary = summarise(all_records)
    print_results(summary)

    output = {
        "experiment": "nlm_cxr_divergence_option_a",
        "timestamp":  datetime.datetime.utcnow().isoformat() + "Z",
        "option":     "A — BiomedCLIP shared vision-language space",
        "config": {
            "model":    "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
            "tau_high": TAU_HIGH, "tau_low": TAU_LOW,
            "delta":    DELTA, "n_sample": N_SAMPLE, "device": DEVICE,
        },
        "summary":  summary,
        "records":  all_records,
    }
    out_path = RESULTS_DIR / "experiment_results_a.json"
    with open(out_path, "w") as fh:
        json.dump(output, fh, indent=2)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
