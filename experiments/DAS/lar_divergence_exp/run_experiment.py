"""
NLM CXR Divergence Experiment
==============================
Empirical validation of AbstractDivergenceRouter V1-V6 on real biomedical
multi-modal data.

Hypothesis
----------
When a chest X-ray image and its radiology report CONTRADICT each other
(image auto-tags show a finding; report text explicitly negates it),
both the ViT vision encoder and BioBERT language encoder will be highly
confident in their respective representations — but their latent embeddings
will be geometrically DISTANT. The content-blind router (V4) should fire
TRIGGER_REPLAN on these cases.

When image and report AGREE, latent embeddings should be geometrically
CLOSE and the router should fire COMMIT_TRAJECTORY.

If this holds: content-blind divergence routing (reading only confidence
scalars and a single geometric distance) can detect genuine clinical
contradictions that a fused model would average away. This is Theorem 1
(Safety-Learning Equivalence) in action on real data.

Run
---
    cd /path/to/lar_divergence_exp
    python run_experiment.py

Results saved to results/experiment_results.json and printed to stdout.
"""

import os
import sys
import json
import tarfile
import random
import datetime
from pathlib import Path
from collections import Counter
from typing import Optional

# ---------------------------------------------------------------------------
# Path bootstrap — top-level JEPA_Playground/lar_jepa/
# ---------------------------------------------------------------------------
_HERE     = Path(__file__).parent.resolve()
_EXP_DIR  = _HERE
_PLAY     = _HERE.parent.parent.parent.parent  # DAS/lar_divergence_exp -> Snath Robotics/experiments/DAS -> experiments -> Snath Robotics -> JEPA_Playground
_LAR_JEPA = _PLAY / "lar_jepa"
_LAR_SRC  = _LAR_JEPA / "lar_jepa" / "src"

for _p in [str(_LAR_JEPA), str(_LAR_SRC), str(_HERE), str(_HERE.parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.types import RouteDecision
from router.divergence_router import CXRDivergenceRouter
from data.dataset import load_dataset, build_contradiction_subset, CXRSample
from encoders.cxr_encoders import ViTStreamEncoder, BioBERTStreamEncoder

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEVICE       = "mps"
N_SAMPLE     = 100     # samples per class (normal / contradiction)
RANDOM_SEED  = 42
TAU_HIGH     = 0.75    # confidence threshold — high
TAU_LOW      = 0.30    # confidence threshold — low
DELTA        = 0.45    # divergence threshold

DATA_DIR     = _EXP_DIR / "data"
IMAGES_DIR   = DATA_DIR / "images"
REPORTS_DIR  = DATA_DIR / "reports"
RESULTS_DIR  = _EXP_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

TARBALLS = {
    "images":  DATA_DIR / "NLMCXR_png.tgz",
    "reports": DATA_DIR / "NLMCXR_reports.tar",
}


# ---------------------------------------------------------------------------
# Step 1 — Extract data if needed
# ---------------------------------------------------------------------------

def extract_data():
    if IMAGES_DIR.exists() and any(IMAGES_DIR.glob("*.png")):
        n = len(list(IMAGES_DIR.glob("*.png")))
        print(f"[data] Images already extracted: {n} PNGs")
    else:
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[data] Extracting {TARBALLS['images']} → {IMAGES_DIR} ...")
        with tarfile.open(TARBALLS["images"], "r:gz") as tf:
            for member in tf.getmembers():
                if member.name.endswith(".png"):
                    member.name = Path(member.name).name
                    tf.extract(member, path=IMAGES_DIR)
        n = len(list(IMAGES_DIR.glob("*.png")))
        print(f"[data] Extracted {n} PNGs")

    if REPORTS_DIR.exists() and any(REPORTS_DIR.glob("*.xml")):
        n = len(list(REPORTS_DIR.glob("*.xml")))
        print(f"[data] Reports already extracted: {n} XMLs")
    else:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[data] Extracting {TARBALLS['reports']} → {REPORTS_DIR} ...")
        with tarfile.open(TARBALLS["reports"], "r:") as tf:
            for member in tf.getmembers():
                if member.name.endswith(".xml"):
                    member.name = Path(member.name).name
                    tf.extract(member, path=REPORTS_DIR)
        n = len(list(REPORTS_DIR.glob("*.xml")))
        print(f"[data] Extracted {n} XMLs")


# ---------------------------------------------------------------------------
# Step 2 — Run routing on a sample
# ---------------------------------------------------------------------------

def run_routing(
    samples:  list[CXRSample],
    label:    str,
    router:   CXRDivergenceRouter,
    vit:      ViTStreamEncoder,
    biobert:  BioBERTStreamEncoder,
    n:        int,
) -> list[dict]:
    from PIL import Image as PILImage

    records = []
    for i, sample in enumerate(samples[:n]):
        try:
            image = PILImage.open(sample.image_path).convert("RGB")
            text  = (sample.findings + " " + sample.impression).strip()
            if not text:
                continue

            z_a, c_a = vit.encode(image)
            z_b, c_b = biobert.encode(text)
            D        = router.divergence(z_a, z_b)
            decision = router.route(c_a, c_b, D)

            record = {
                "cxr_id":          sample.cxr_id,
                "label":           label,
                "confidence_vit":  round(c_a, 4),
                "confidence_bert": round(c_b, 4),
                "divergence":      round(D, 4),
                "decision":        decision.value,
            }
            records.append(record)

            if (i + 1) % 10 == 0:
                print(f"  [{label}] {i+1}/{n}  "
                      f"c_A={c_a:.3f} c_B={c_b:.3f} D={D:.3f} → {decision.value}")

        except Exception as e:
            print(f"  [{label}] error on {sample.cxr_id}: {e}")
            continue

    return records


# ---------------------------------------------------------------------------
# Step 3 — Summarise results
# ---------------------------------------------------------------------------

def summarise(records: list[dict]) -> dict:
    by_label: dict[str, list] = {}
    for r in records:
        by_label.setdefault(r["label"], []).append(r)

    summary = {}
    for label, recs in by_label.items():
        decisions = Counter(r["decision"] for r in recs)
        mean_D    = sum(r["divergence"] for r in recs) / len(recs)
        mean_ca   = sum(r["confidence_vit"] for r in recs) / len(recs)
        mean_cb   = sum(r["confidence_bert"] for r in recs) / len(recs)
        summary[label] = {
            "n":             len(recs),
            "mean_divergence":  round(mean_D, 4),
            "mean_conf_vit":    round(mean_ca, 4),
            "mean_conf_biobert": round(mean_cb, 4),
            "decisions":     dict(decisions),
            "trigger_replan_pct": round(
                100 * decisions.get("TRIGGER_REPLAN", 0) / len(recs), 1
            ),
            "commit_pct": round(
                100 * decisions.get("COMMIT_TRAJECTORY", 0) / len(recs), 1
            ),
        }
    return summary


def print_results(summary: dict):
    print("\n" + "="*65)
    print("  EXPERIMENT RESULTS — AbstractDivergenceRouter on NLM CXR")
    print("="*65)
    print(f"  τ_high={TAU_HIGH}  τ_low={TAU_LOW}  δ={DELTA}\n")

    for label, s in summary.items():
        print(f"  [{label.upper()}]  n={s['n']}")
        print(f"    Mean divergence D : {s['mean_divergence']}")
        print(f"    Mean conf ViT     : {s['mean_conf_vit']}")
        print(f"    Mean conf BioBERT : {s['mean_conf_biobert']}")
        print(f"    Decisions         : {s['decisions']}")
        print(f"    TRIGGER_REPLAN %  : {s['trigger_replan_pct']}%")
        print(f"    COMMIT %          : {s['commit_pct']}%")
        print()

    # Key metric: does contradiction → higher TRIGGER_REPLAN?
    if "contradiction" in summary and "normal" in summary:
        contra_replan = summary["contradiction"]["trigger_replan_pct"]
        normal_replan = summary["normal"]["trigger_replan_pct"]
        contra_D      = summary["contradiction"]["mean_divergence"]
        normal_D      = summary["normal"]["mean_divergence"]
        print(f"  HYPOTHESIS CHECK")
        print(f"  Contradiction TRIGGER_REPLAN: {contra_replan}%  "
              f"(expected >> Normal)")
        print(f"  Normal        TRIGGER_REPLAN: {normal_replan}%")
        print(f"  Mean D contradiction: {contra_D}  >?  Normal: {normal_D}")
        if contra_replan > normal_replan and contra_D > normal_D:
            print("\n  ✓ HYPOTHESIS CONFIRMED — content-blind routing detects")
            print("    structural contradiction without reading image or text.")
        else:
            print("\n  ~ Partial / threshold-sensitive — tune τ_high / δ.")
    print("="*65)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    random.seed(RANDOM_SEED)

    print("\n" + "="*65)
    print("  Snath AI — NLM CXR AbstractDivergenceRouter Experiment")
    print("="*65 + "\n")

    # 1. Data
    extract_data()

    # 2. Load dataset
    print("\n[dataset] Loading paired samples...")
    samples = load_dataset(str(IMAGES_DIR), str(REPORTS_DIR))
    normal, contradiction = build_contradiction_subset(samples)

    random.shuffle(normal)
    random.shuffle(contradiction)
    print(f"[dataset] Sampling {N_SAMPLE} normal + {N_SAMPLE} contradiction cases\n")

    # 3. Load encoders
    vit     = ViTStreamEncoder(device=DEVICE)
    biobert = BioBERTStreamEncoder(device=DEVICE)

    # 4. Build router (encoders passed for type-checking; route() is content-blind)
    router  = CXRDivergenceRouter(
        vision_encoder=vit.model,
        language_encoder=biobert.model,
        tau_high=TAU_HIGH,
        tau_low=TAU_LOW,
        delta=DELTA,
        device=DEVICE,
    )

    # 5. Run routing
    print("[routing] Normal cases...")
    normal_records = run_routing(normal, "normal", router, vit, biobert, N_SAMPLE)

    print("\n[routing] Contradiction cases...")
    contra_records = run_routing(contradiction, "contradiction", router, vit, biobert, N_SAMPLE)

    all_records = normal_records + contra_records

    # 6. Summarise
    summary = summarise(all_records)
    print_results(summary)

    # 7. Save
    output = {
        "experiment":   "nlm_cxr_divergence_v1",
        "timestamp":    datetime.datetime.utcnow().isoformat() + "Z",
        "config": {
            "tau_high": TAU_HIGH,
            "tau_low":  TAU_LOW,
            "delta":    DELTA,
            "n_sample": N_SAMPLE,
            "device":   DEVICE,
        },
        "summary":  summary,
        "records":  all_records,
    }
    out_path = RESULTS_DIR / "experiment_results.json"
    with open(out_path, "w") as fh:
        json.dump(output, fh, indent=2)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
