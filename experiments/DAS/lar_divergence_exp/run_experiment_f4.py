"""
Experiment F4 — COCO / OpenCLIP: 80-class Softmax Routing (Final)
=================================================================
Fixes the calibration failure in F3 where per-concept binary softmax
gave ~0.5 for every category (uninformative).

THE KEY FIX: use CLIP's standard 80-class softmax over all COCO categories:
  concept_vec_i = softmax(τ · [sim(img, template_1), ..., sim(img, template_80)])

This is CLIP zero-shot classification. It gives sparse, discriminative
concept vectors:  dog_image → [dog=0.82, others≈0.01]
                  person+bike → [person=0.41, bicycle=0.37, others≈0.01]

This mirrors exactly how BiomedCLIP finding vectors work in C2/E:
  low probability  ≈ finding absent   (~0.02)
  high probability ≈ finding present  (~0.90)

Text concept: keyword binary matching (same as F3).

Oracle / routing design (same as F2/F3):
  j_normal = argmin L1(softmax_concept_i, softmax_concept_j)
  j_contra = argmax CLS_sim(img_i, img_j) × L1(softmax_concept_i, softmax_concept_j)

Routing   : L1(softmax_concept_img, keyword_concept_txt)
Fusion    : 1 - CLS cosine(img_CLS, txt_CLS)
Supervised: concat logistic (N=100 labels)
"""

import os, sys, json, datetime, random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import open_clip
from PIL import Image as PILImage
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import fisher_exact

_HERE       = Path(__file__).parent.resolve()
RESULTS_DIR = _HERE / "results"
RESULTS_DIR.mkdir(exist_ok=True)
OUT_PATH    = RESULTS_DIR / "experiment_results_f4.json"

DEVICE      = "mps"
MODEL_ID    = ("ViT-B-32", "openai")
DATASET_ID  = "clip-benchmark/wds_mscoco_captions"
POOL_SIZE   = 800
SEED        = 42
TEMPERATURE = 100.0    # matches CLIP's learned temperature (~1/0.01)

COCO_CATS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]
N_CONCEPTS = len(COCO_CATS)    # 80

_SYNONYMS = {
    "person": ["person", "people", "man", "woman", "boy", "girl", "child", "children", "kid", "kids", "adult", "human"],
    "motorcycle": ["motorcycle", "motorbike", "scooter"],
    "airplane": ["airplane", "plane", "aircraft", "jet"],
    "bus": ["bus", "coach"],
    "truck": ["truck", "lorry", "pickup"],
    "boat": ["boat", "ship", "vessel", "canoe", "kayak"],
    "traffic light": ["traffic light", "stoplight"],
    "bench": ["bench", "seat"],
    "bird": ["bird", "pigeon", "seagull", "parrot", "crow", "duck"],
    "cat": ["cat", "kitten", "feline"],
    "dog": ["dog", "puppy", "hound", "canine"],
    "horse": ["horse", "pony"],
    "sheep": ["sheep", "lamb"],
    "cow": ["cow", "cattle", "bull"],
    "elephant": ["elephant"], "bear": ["bear"], "zebra": ["zebra"], "giraffe": ["giraffe"],
    "backpack": ["backpack", "rucksack", "bag"],
    "umbrella": ["umbrella"],
    "handbag": ["handbag", "purse"],
    "tie": ["tie", "necktie"],
    "suitcase": ["suitcase", "luggage"],
    "frisbee": ["frisbee"],
    "skis": ["ski", "skis", "skiing"],
    "snowboard": ["snowboard"],
    "sports ball": ["ball", "basketball", "football", "soccer ball", "baseball", "volleyball"],
    "kite": ["kite"],
    "baseball bat": ["bat", "baseball bat"],
    "baseball glove": ["glove", "baseball glove", "mitt"],
    "skateboard": ["skateboard"],
    "surfboard": ["surfboard"],
    "tennis racket": ["racket", "racquet", "tennis"],
    "bottle": ["bottle"],
    "wine glass": ["wine", "wine glass"],
    "cup": ["cup", "mug"],
    "fork": ["fork"], "knife": ["knife"], "spoon": ["spoon"], "bowl": ["bowl"],
    "banana": ["banana"], "apple": ["apple"],
    "sandwich": ["sandwich", "sub"],
    "orange": ["orange"], "broccoli": ["broccoli"], "carrot": ["carrot"],
    "hot dog": ["hot dog", "hotdog", "sausage"],
    "pizza": ["pizza"],
    "donut": ["donut", "doughnut"],
    "cake": ["cake", "cupcake"],
    "chair": ["chair", "stool"],
    "couch": ["couch", "sofa"],
    "potted plant": ["plant", "potted", "flower"],
    "bed": ["bed"],
    "dining table": ["table", "dining"],
    "toilet": ["toilet"],
    "tv": ["tv", "television", "monitor", "screen"],
    "laptop": ["laptop", "computer"],
    "mouse": ["mouse"],
    "remote": ["remote", "controller"],
    "keyboard": ["keyboard"],
    "cell phone": ["phone", "smartphone", "cell phone", "mobile"],
    "microwave": ["microwave"],
    "oven": ["oven", "stove"],
    "toaster": ["toaster"],
    "sink": ["sink", "basin"],
    "refrigerator": ["refrigerator", "fridge"],
    "book": ["book", "books"],
    "clock": ["clock", "watch"],
    "vase": ["vase"],
    "scissors": ["scissors"],
    "teddy bear": ["teddy", "stuffed animal", "plush"],
    "hair drier": ["hair dryer", "hair drier", "dryer"],
    "toothbrush": ["toothbrush"],
}

def keyword_concept_vec(caption):
    text = caption.lower()
    vec  = np.zeros(N_CONCEPTS, dtype=np.float32)
    for i, cat in enumerate(COCO_CATS):
        syns = _SYNONYMS.get(cat, [cat])
        if any(s in text for s in syns):
            vec[i] = 1.0
    return vec


# ── Model ─────────────────────────────────────────────────────────────────────
def load_model():
    model, _, preprocess = open_clip.create_model_and_transforms(*MODEL_ID)
    tokenizer = open_clip.get_tokenizer(MODEL_ID[0])
    model = model.to(DEVICE).eval()
    return model, preprocess, tokenizer


def build_pos_templates(model, tokenizer):
    """Positive templates only — used for 80-class global softmax."""
    templates = [f"a photo of a {c}" for c in COCO_CATS]
    tok = tokenizer(templates).to(DEVICE)
    with torch.no_grad():
        embs = F.normalize(model.encode_text(tok).float(), dim=-1)
    return embs   # (80, D)


def encode_image(img_pil, model, preprocess, pos_embs):
    """Returns (cls_np, softmax_concept_80d) using 80-class global softmax."""
    img_t = preprocess(img_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        img_emb = F.normalize(model.encode_image(img_t).float(), dim=-1).squeeze(0)
    cls_np  = img_emb.cpu().numpy()
    sims    = (pos_embs @ img_emb).cpu()            # (80,) cosine similarities
    probs   = torch.softmax(sims * TEMPERATURE, dim=0).numpy()  # (80,) sparse
    return cls_np, probs.astype(np.float32)


def encode_text_cls(text, model, tokenizer):
    tok = tokenizer([text]).to(DEVICE)
    with torch.no_grad():
        emb = F.normalize(model.encode_text(tok).float(), dim=-1).squeeze(0)
    return emb.cpu().numpy()


# ── Dataset ───────────────────────────────────────────────────────────────────
def stream_coco(n_max):
    from datasets import load_dataset as hf_load
    print(f"[data] Streaming up to {n_max} COCO captions...")
    ds = hf_load(DATASET_ID, split="test", streaming=True, trust_remote_code=True)
    samples = []
    for item in ds:
        try:
            jpg = item.get("jpg") or item.get("image")
            txt = item.get("txt") or item.get("caption") or item.get("text")
            if jpg is None or txt is None:
                continue
            if isinstance(jpg, bytes):
                import io
                img = PILImage.open(io.BytesIO(jpg)).convert("RGB")
            elif hasattr(jpg, "convert"):
                img = jpg.convert("RGB")
            else:
                continue
            if isinstance(txt, bytes):
                caption = txt.decode("utf-8", errors="replace").strip()
            elif isinstance(txt, list):
                caption = txt[0].strip() if txt else ""
            else:
                caption = str(txt).strip()
            if len(caption) < 5:
                continue
            kw = keyword_concept_vec(caption)
            if kw.sum() == 0:
                continue
            samples.append({"image": img, "caption": caption, "kw_vec": kw})
            if len(samples) >= n_max:
                break
        except Exception:
            pass
    print(f"[data] Collected {len(samples)} samples.")
    return samples


# ── Pool encoding ─────────────────────────────────────────────────────────────
def encode_pool(samples, model, preprocess, tokenizer, pos_embs):
    N = len(samples)
    D = 512
    cls_vecs     = np.zeros((N, D),          dtype=np.float32)
    concept_vecs = np.zeros((N, N_CONCEPTS), dtype=np.float32)
    print(f"[pool] Encoding image CLS + 80-class softmax for {N} images...")
    for i, s in enumerate(samples):
        c, cv = encode_image(s["image"], model, preprocess, pos_embs)
        cls_vecs[i]     = c
        concept_vecs[i] = cv
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{N}")

    # Diagnostic: check that concept vectors are actually sparse
    top1_prob = concept_vecs.max(axis=1)
    print(f"  [diag] mean max-prob per image: {top1_prob.mean():.4f}  "
          f"(should be ~0.2–0.9 for sparse distribution)")
    # Top-1 categories
    top1_cats = concept_vecs.argmax(axis=1)
    unique, counts = np.unique(top1_cats, return_counts=True)
    top5_idx = np.argsort(-counts)[:5]
    print(f"  [diag] top-5 dominant categories: "
          f"{[(COCO_CATS[unique[i]], counts[i]) for i in top5_idx]}")
    return cls_vecs, concept_vecs


# ── Oracle ────────────────────────────────────────────────────────────────────
def find_oracle_partners(cls_vecs, concept_vecs):
    N = len(cls_vecs)
    # CLS similarity
    norms   = np.linalg.norm(cls_vecs, axis=1, keepdims=True)
    cls_n   = cls_vecs / (norms + 1e-9)
    cls_sim = cls_n @ cls_n.T
    np.fill_diagonal(cls_sim, -2.0)

    # Concept L1
    l1 = np.abs(concept_vecs[:, None, :] - concept_vecs[None, :, :]).sum(axis=2)

    # Normal: concept-nearest
    l1_n = l1.copy(); np.fill_diagonal(l1_n, np.inf)
    j_normal = l1_n.argmin(axis=1)

    # Hard negative: argmax CLS_sim × L1
    cls_pos = np.clip(cls_sim, 0.0, 1.0)
    l1_pos  = np.clip(l1, 0.0, None); np.fill_diagonal(l1_pos, -1.0)
    score   = cls_pos * l1_pos; np.fill_diagonal(score, -1.0)
    j_contra = score.argmax(axis=1)

    mn_l1 = np.mean([l1_n[i, j_normal[i]] for i in range(N)])
    mc_l1 = np.mean([l1_pos[i, j_contra[i]] for i in range(N)])
    mn_cs = np.mean([max(0, cls_sim[i, j_normal[i]]) for i in range(N)])
    mc_cs = np.mean([max(0, cls_sim[i, j_contra[i]]) for i in range(N)])
    print(f"\n  Oracle diagnostics (80-class softmax concept):")
    print(f"    Normal  — mean L1: {mn_l1:.4f}  CLS_sim: {mn_cs:.4f}")
    print(f"    Contra  — mean L1: {mc_l1:.4f}  CLS_sim: {mc_cs:.4f}  ← hard negative")
    return j_normal, j_contra


# ── Build pairs ───────────────────────────────────────────────────────────────
def build_pairs(samples, cls_vecs, concept_vecs,
                model, preprocess, tokenizer, pos_embs, rng):
    N = len(samples)
    j_normal, j_contra = find_oracle_partners(cls_vecs, concept_vecs)

    idxs = list(range(N)); rng.shuffle(idxs); sel = idxs[:200]
    normal_pairs, contra_pairs = [], []
    print(f"\n[pairs] Encoding text CLS for {len(sel)*2} pairs...")
    for rank, i in enumerate(sel):
        jn, jc = int(j_normal[i]), int(j_contra[i])
        cls_n = encode_text_cls(samples[jn]["caption"], model, tokenizer)
        normal_pairs.append({
            "label":   "normal",
            "caption": samples[jn]["caption"][:200],
            "cv_img":  concept_vecs[i],
            "cv_txt":  samples[jn]["kw_vec"],
            "cls_img": cls_vecs[i],
            "cls_txt": cls_n,
        })
        cls_c = encode_text_cls(samples[jc]["caption"], model, tokenizer)
        contra_pairs.append({
            "label":   "contradiction",
            "caption": samples[jc]["caption"][:200],
            "cv_img":  concept_vecs[i],
            "cv_txt":  samples[jc]["kw_vec"],
            "cls_img": cls_vecs[i],
            "cls_txt": cls_c,
        })
        if (rank + 1) % 50 == 0:
            print(f"  {rank+1}/{len(sel)}")

    rng.shuffle(normal_pairs); rng.shuffle(contra_pairs)
    cal  = normal_pairs[:50]  + contra_pairs[:50]
    test = normal_pairs[50:100] + contra_pairs[50:100]
    rng.shuffle(cal); rng.shuffle(test)
    print(f"[pairs] Cal: {len(cal)}  Test: {len(test)}")
    return cal, test


# ── Scoring ───────────────────────────────────────────────────────────────────
def score_routing(pairs):
    return np.array([float(np.abs(p["cv_img"] - p["cv_txt"]).sum()) for p in pairs])

def score_fusion(pairs):
    return np.array([float(1.0 - np.dot(p["cls_img"], p["cls_txt"])) for p in pairs])

def score_supervised(cal, test):
    X_c = np.array([np.concatenate([p["cv_img"], p["cv_txt"]]) for p in cal])
    y_c = np.array([1 if p["label"]=="contradiction" else 0 for p in cal])
    X_t = np.array([np.concatenate([p["cv_img"], p["cv_txt"]]) for p in test])
    sc  = StandardScaler(); X_c = sc.fit_transform(X_c); X_t = sc.transform(X_t)
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
    clf.fit(X_c, y_c)
    return clf.predict_proba(X_t)[:, 1]

def get_labels(pairs):
    return np.array([1 if p["label"]=="contradiction" else 0 for p in pairs])


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(scores, labels, name):
    scores = np.array(scores, dtype=float); labels = np.array(labels, dtype=int)
    auroc  = float(roc_auc_score(labels, scores))
    thr    = np.unique(scores)
    best_acc, best_thr = 0.0, np.median(scores)
    for t in thr:
        acc = ((scores >= t) == labels).mean()
        if acc > best_acc:
            best_acc, best_thr = acc, t
    preds = (scores >= best_thr).astype(int)
    tp = int(((preds==1)&(labels==1)).sum()); fp = int(((preds==1)&(labels==0)).sum())
    fn = int(((preds==0)&(labels==1)).sum()); tn = int(((preds==0)&(labels==0)).sum())
    _, p_val = fisher_exact([[tp,fp],[fn,tn]], alternative="greater")
    return {
        "method": name, "auroc": round(auroc, 4), "accuracy": round(best_acc, 4),
        "p_value": float(p_val),
        "sensitivity": round(tp/(tp+fn), 3) if (tp+fn)>0 else 0,
        "specificity": round(tn/(tn+fp), 3) if (tn+fp)>0 else 0,
        "contingency": {"tp":tp,"fp":fp,"fn":fn,"tn":tn},
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    rng = random.Random(SEED); np.random.seed(SEED)

    print("\n" + "="*76)
    print("  Experiment F4 — COCO / OpenCLIP: 80-class Softmax + Hard Negatives")
    print("  Image concept: sparse 80-class softmax  |  Text: keyword binary")
    print("="*76 + "\n")

    model, preprocess, tokenizer = load_model()
    pos_embs = build_pos_templates(model, tokenizer)
    print(f"[model] Loaded. {N_CONCEPTS} concept templates encoded (positive only).\n")

    samples = stream_coco(POOL_SIZE)
    if len(samples) < 400:
        raise RuntimeError(f"Need ≥400 samples, got {len(samples)}")

    cls_vecs, concept_vecs = encode_pool(
        samples, model, preprocess, tokenizer, pos_embs)

    cal, test = build_pairs(
        samples, cls_vecs, concept_vecs,
        model, preprocess, tokenizer, pos_embs, rng)

    print("\n[scoring] Computing all method scores...")
    y_test    = get_labels(test)
    s_routing = score_routing(test)
    s_fusion  = score_fusion(test)
    s_super   = score_supervised(cal, test)

    rt_n = s_routing[y_test==0]; rt_c = s_routing[y_test==1]
    fc_n = s_fusion[y_test==0];  fc_c = s_fusion[y_test==1]

    print(f"\n  Routing (80-class softmax img × keyword txt):")
    print(f"    normal μ={rt_n.mean():.4f}  contra μ={rt_c.mean():.4f}  "
          f"lift={rt_c.mean()/max(rt_n.mean(),1e-6):.2f}×")
    print(f"  Fusion (CLS cosine):")
    print(f"    normal μ={fc_n.mean():.4f}  contra μ={fc_c.mean():.4f}  "
          f"lift={fc_c.mean()/max(fc_n.mean(),1e-6):.2f}×")

    res_routing = evaluate(s_routing, y_test,
        "Routing — 80-class softmax image × keyword text L1 (zero-shot)")
    res_fusion  = evaluate(s_fusion,  y_test,
        "Fusion  — OpenCLIP CLS cosine (zero-shot)")
    res_super   = evaluate(s_super,   y_test,
        "Supervised — concat logistic (N=100 labels)")
    results = [res_routing, res_fusion, res_super]

    print("\n" + "="*76)
    print("  EXPERIMENT F4 RESULTS — COCO / OpenCLIP (80-class softmax)")
    print("="*76)
    print(f"\n  {'Method':<55} {'AUROC':>6}  {'Acc':>6}  {'p':>12}  {'Labels?':>10}")
    print("  " + "-"*83)
    for r in results:
        flag = "No" if "zero-shot" in r["method"] else "Yes (N=100)"
        print(f"  {r['method']:<55} {r['auroc']:>6.4f}  {r['accuracy']:>6.1%}  "
              f"{r['p_value']:>12.2e}  {flag:>10}")

    delta = res_routing["auroc"] - res_fusion["auroc"]
    print(f"\n  Routing vs Fusion: AUROC {delta:+.4f}  "
          f"(routing lift {rt_c.mean()/max(rt_n.mean(),1e-6):.2f}× "
          f"vs fusion lift {fc_c.mean()/max(fc_n.mean(),1e-6):.2f}×)")

    if delta > 0.05:
        print(f"\n  ✓ ROUTING OUTPERFORMS FUSION (Δ AUROC = {delta:.3f})")
        print("    Sparse 80-class softmax gives discriminative concept vectors.")
        print("    Hard-negative oracle specifically blinds CLS cosine signal.")
        print("    Domain isomorphism: routing structural advantage is general.")
    elif delta > 0:
        print(f"\n  ~ Routing marginally outperforms fusion (+{delta:.4f} AUROC).")
    else:
        print(f"\n  ~ Routing did not outperform fusion (Δ={delta:.4f}).")

    print("="*76)

    # Illustrative example
    contra_idx = np.where(y_test == 1)[0]
    if len(contra_idx) > 0:
        idx = contra_idx[0]
        p   = test[idx]
        top3_img  = np.argsort(-p["cv_img"])[:3]
        nonzero_txt = np.where(p["cv_txt"] > 0.5)[0]
        print(f"\n  ILLUSTRATIVE EXAMPLE (test set, contradiction):")
        print(f"    Caption: '{p['caption'][:100]}'")
        print(f"    Image top-3 (softmax): "
              f"{[COCO_CATS[i]+'=%.3f'%p['cv_img'][i] for i in top3_img]}")
        print(f"    Text concepts (keyword): "
              f"{[COCO_CATS[i] for i in nonzero_txt[:5]]}")
        top3_diff = np.argsort(-np.abs(p["cv_img"] - p["cv_txt"]))[:3]
        print(f"    Biggest disagreements: "
              f"{[COCO_CATS[i]+'|img=%.3f,txt=%.0f'%(p['cv_img'][i],p['cv_txt'][i]) for i in top3_diff]}")

    output = {
        "experiment":   "fusion_vs_routing_coco_80class_softmax",
        "timestamp":    datetime.datetime.utcnow().isoformat() + "Z",
        "model":        f"OpenCLIP {MODEL_ID[0]} / {MODEL_ID[1]}",
        "dataset":      "MSCOCO (clip-benchmark/wds_mscoco_captions)",
        "ground_truth": "Hard contradiction: argmax[CLS_sim × L1_softmax_concept]",
        "n_concepts":   N_CONCEPTS,
        "concept_type": "80-class softmax (sparse) image × keyword text (binary)",
        "temperature":  TEMPERATURE,
        "results":      results,
        "distributions": {
            "routing_normal_mean": round(float(rt_n.mean()), 4),
            "routing_contra_mean": round(float(rt_c.mean()), 4),
            "routing_lift":        round(float(rt_c.mean()/max(rt_n.mean(),1e-6)), 3),
            "fusion_normal_mean":  round(float(fc_n.mean()), 4),
            "fusion_contra_mean":  round(float(fc_c.mean()), 4),
            "fusion_lift":         round(float(fc_c.mean()/max(fc_n.mean(),1e-6)), 3),
        },
    }
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[saved] {OUT_PATH}")


if __name__ == "__main__":
    main()
