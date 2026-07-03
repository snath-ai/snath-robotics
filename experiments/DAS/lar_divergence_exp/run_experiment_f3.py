"""
Experiment F3 — COCO / OpenCLIP: Hard Contradiction with 80 Categories
========================================================================
Third iteration of Experiment F. Fixes the calibration mismatch of F2 by:

  1. Using MS-COCO captions (clip-benchmark/wds_mscoco_captions).
     COCO captions explicitly name the objects present — keyword matching
     gives CLEAN BINARY text concept vectors.

  2. Using ALL 80 COCO object categories as the concept vocabulary.
     Much more discriminative than 25 generic concepts.

  3. Hybrid concept vectors (same approach as C2's TXV oracle):
       Image concept : CLIP image template probabilities (80-dim, continuous)
       Text concept  : keyword matching on caption  (80-dim, binary)
     These are correlated but NOT identical — the same controlled independence
     as C2's TXV (image oracle, different model) vs BiomedCLIP routing.

  Hard contradiction oracle (same as F2):
    j_contra = argmax_k [ CLS_sim(img_i, img_k) × L1_image_concept(i, k) ]
    j_normal = argmin_k L1_image_concept(i, k),  k ≠ i

  Normal  pair : (image_i, caption of concept-nearest image j)
  Contra  pair : (image_i, caption of CLS-similar & concept-different image k)

  Routing   : L1(CLIP_image_concept, keyword_text_concept)   [zero-shot]
  Fusion    : 1 - cosine(CLS_image, CLS_text)                [zero-shot]
  Supervised: Logistic regression on [v_A ; v_B]             [N=100 labels]

Expected: Routing > Fusion
  Hard negatives look globally similar → CLS cosine cannot separate them.
  COCO category concept vectors are specific → L1 clearly separates them.
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
OUT_PATH    = RESULTS_DIR / "experiment_results_f3.json"

DEVICE      = "mps"
MODEL_ID    = ("ViT-B-32", "openai")
DATASET_ID  = "clip-benchmark/wds_mscoco_captions"
POOL_SIZE   = 800
SEED        = 42

# ── 80 COCO object categories ─────────────────────────────────────────────────
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
N_CONCEPTS = len(COCO_CATS)   # 80

# ── Keyword synonyms (extend the bare category names) ─────────────────────────
_SYNONYMS = {
    "person": ["person", "people", "man", "woman", "boy", "girl", "child", "children", "kid", "kids", "adult", "human"],
    "motorcycle": ["motorcycle", "motorbike", "scooter"],
    "airplane": ["airplane", "plane", "aircraft", "jet"],
    "bus": ["bus", "coach"],
    "truck": ["truck", "lorry", "pickup"],
    "boat": ["boat", "ship", "vessel", "canoe", "kayak"],
    "traffic light": ["traffic light", "stoplight", "signal"],
    "bench": ["bench", "seat"],
    "bird": ["bird", "pigeon", "seagull", "parrot", "crow", "robin", "duck"],
    "cat": ["cat", "kitten", "feline"],
    "dog": ["dog", "puppy", "hound", "canine"],
    "horse": ["horse", "pony"],
    "sheep": ["sheep", "lamb"],
    "cow": ["cow", "cattle", "bull"],
    "elephant": ["elephant"],
    "bear": ["bear"],
    "zebra": ["zebra"],
    "giraffe": ["giraffe"],
    "backpack": ["backpack", "rucksack", "bag"],
    "umbrella": ["umbrella"],
    "handbag": ["handbag", "purse", "bag"],
    "tie": ["tie", "necktie"],
    "suitcase": ["suitcase", "luggage", "bag"],
    "frisbee": ["frisbee"],
    "skis": ["ski", "skis", "skiing"],
    "snowboard": ["snowboard"],
    "sports ball": ["ball", "basketball", "football", "soccer ball", "baseball", "volleyball", "rugby"],
    "kite": ["kite"],
    "baseball bat": ["bat", "baseball bat"],
    "baseball glove": ["glove", "baseball glove", "mitt"],
    "skateboard": ["skateboard"],
    "surfboard": ["surfboard"],
    "tennis racket": ["racket", "racquet", "tennis"],
    "bottle": ["bottle"],
    "wine glass": ["wine", "glass", "wine glass"],
    "cup": ["cup", "mug", "glass"],
    "fork": ["fork"],
    "knife": ["knife"],
    "spoon": ["spoon"],
    "bowl": ["bowl"],
    "banana": ["banana"],
    "apple": ["apple"],
    "sandwich": ["sandwich", "sub", "burger"],
    "orange": ["orange"],
    "broccoli": ["broccoli"],
    "carrot": ["carrot"],
    "hot dog": ["hot dog", "hotdog", "sausage"],
    "pizza": ["pizza"],
    "donut": ["donut", "doughnut"],
    "cake": ["cake", "cupcake"],
    "chair": ["chair", "stool"],
    "couch": ["couch", "sofa", "settee"],
    "potted plant": ["plant", "potted", "flower"],
    "bed": ["bed"],
    "dining table": ["table", "dining"],
    "toilet": ["toilet"],
    "tv": ["tv", "television", "monitor", "screen"],
    "laptop": ["laptop", "computer", "notebook"],
    "mouse": ["mouse", "mice"],
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
    "teddy bear": ["teddy", "bear", "stuffed animal"],
    "hair drier": ["hair dryer", "hair drier", "dryer"],
    "toothbrush": ["toothbrush"],
}

def keyword_concept_vec(caption):
    """80-dim binary concept vector: 1 if category keyword found in caption."""
    text = caption.lower()
    vec  = np.zeros(N_CONCEPTS, dtype=np.float32)
    for i, cat in enumerate(COCO_CATS):
        syns = _SYNONYMS.get(cat, [cat])
        if any(s in text for s in syns):
            vec[i] = 1.0
    return vec


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model():
    model, _, preprocess = open_clip.create_model_and_transforms(*MODEL_ID)
    tokenizer = open_clip.get_tokenizer(MODEL_ID[0])
    model = model.to(DEVICE).eval()
    return model, preprocess, tokenizer


def build_image_templates(model, tokenizer):
    """Pre-encode CLIP image template embeddings for all 80 COCO categories."""
    pos_texts = [f"a photo of a {c}" for c in COCO_CATS]
    neg_texts  = [f"a photo without a {c}" for c in COCO_CATS]
    tok = tokenizer(pos_texts + neg_texts).to(DEVICE)
    with torch.no_grad():
        embs = F.normalize(model.encode_text(tok).float(), dim=-1)
    return embs[:N_CONCEPTS], embs[N_CONCEPTS:]   # (80, D), (80, D)


def encode_image(img_pil, model, preprocess, pos_embs, neg_embs):
    """Returns (cls_np, concept_np_80d) in one forward pass."""
    img_t = preprocess(img_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        img_emb = F.normalize(model.encode_image(img_t).float(), dim=-1).squeeze(0)
    cls_np  = img_emb.cpu().numpy()
    sim_pos = (pos_embs @ img_emb).cpu().numpy()
    sim_neg = (neg_embs @ img_emb).cpu().numpy()
    stack   = np.stack([sim_pos, sim_neg], axis=1)
    exps    = np.exp(stack - stack.max(axis=1, keepdims=True))
    probs   = exps / exps.sum(axis=1, keepdims=True)
    return cls_np, probs[:, 0].astype(np.float32)


def encode_text_cls(text, model, tokenizer):
    """Returns (cls_np,) — used only for fusion baseline."""
    tok = tokenizer([text]).to(DEVICE)
    with torch.no_grad():
        emb = F.normalize(model.encode_text(tok).float(), dim=-1).squeeze(0)
    return emb.cpu().numpy()


# ── Stream COCO ───────────────────────────────────────────────────────────────
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
                continue          # skip captions with no detected concept (rare)
            samples.append({"image": img, "caption": caption, "kw_vec": kw})
            if len(samples) >= n_max:
                break
        except Exception:
            pass
    print(f"[data] Collected {len(samples)} samples.")
    return samples


# ── Pool encoding ─────────────────────────────────────────────────────────────
def encode_pool(samples, model, preprocess, tokenizer, pos_embs, neg_embs):
    N = len(samples)
    cls_vecs     = np.zeros((N, 512), dtype=np.float32)
    concept_vecs = np.zeros((N, N_CONCEPTS), dtype=np.float32)
    print(f"[pool] Encoding image CLS + concept vectors for {N} images...")
    for i, s in enumerate(samples):
        cls_np, cv = encode_image(s["image"], model, preprocess, pos_embs, neg_embs)
        cls_vecs[i]     = cls_np
        concept_vecs[i] = cv
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{N}")
    return cls_vecs, concept_vecs


# ── Hard contradiction oracle ─────────────────────────────────────────────────
def find_oracle_partners(cls_vecs, concept_vecs):
    """
    j_normal = argmin_k L1(concept_i, concept_k),  k≠i
    j_contra = argmax_k [ CLS_sim(i,k) × L1(concept_i, concept_k) ],  k≠i
    """
    N = len(cls_vecs)
    norms   = np.linalg.norm(cls_vecs, axis=1, keepdims=True)
    cls_n   = cls_vecs / (norms + 1e-9)
    cls_sim = cls_n @ cls_n.T
    np.fill_diagonal(cls_sim, -2.0)

    l1 = np.abs(concept_vecs[:, None, :] - concept_vecs[None, :, :]).sum(axis=2)
    # Normal: concept-nearest
    l1_n = l1.copy(); np.fill_diagonal(l1_n, np.inf)
    j_normal = l1_n.argmin(axis=1)

    # Hard negative: max CLS_sim × L1
    cls_pos = np.clip(cls_sim, 0.0, 1.0)
    l1_pos  = np.clip(l1, 0.0, None); np.fill_diagonal(l1_pos, -1.0)
    score   = cls_pos * l1_pos; np.fill_diagonal(score, -1.0)
    j_contra = score.argmax(axis=1)

    mn_l1  = np.mean([l1_n[i, j_normal[i]] for i in range(N)])
    mc_l1  = np.mean([l1_pos[i, j_contra[i]] for i in range(N)])
    mn_cls = np.mean([max(0, cls_sim[i, j_normal[i]]) for i in range(N)])
    mc_cls = np.mean([max(0, cls_sim[i, j_contra[i]]) for i in range(N)])
    print(f"\n  Oracle diagnostics (image CLIP concept, 80 COCO dims):")
    print(f"    Normal  — mean image-concept L1: {mn_l1:.4f}  CLS_sim: {mn_cls:.4f}")
    print(f"    Contra  — mean image-concept L1: {mc_l1:.4f}  CLS_sim: {mc_cls:.4f}"
          f"  (target: low CLS gap, high L1 gap)")
    return j_normal, j_contra


# ── Build pairs ───────────────────────────────────────────────────────────────
def build_pairs(samples, cls_vecs, concept_vecs,
                model, preprocess, tokenizer, pos_embs, neg_embs, rng):
    N = len(samples)
    j_normal, j_contra = find_oracle_partners(cls_vecs, concept_vecs)

    idxs = list(range(N)); rng.shuffle(idxs); sel = idxs[:200]

    normal_pairs = []
    contra_pairs = []
    print(f"\n[pairs] Building 400 pairs (image CLS + keyword concept already cached)...")
    for rank, i in enumerate(sel):
        jn = int(j_normal[i]); jc = int(j_contra[i])

        # Normal
        cls_txt_n = encode_text_cls(samples[jn]["caption"], model, tokenizer)
        normal_pairs.append({
            "label":   "normal",
            "caption": samples[jn]["caption"][:200],
            "cv_img":  concept_vecs[i],             # CLIP image concept (80-dim)
            "cv_txt":  samples[jn]["kw_vec"],        # keyword text concept (80-dim binary)
            "cls_img": cls_vecs[i],
            "cls_txt": cls_txt_n,
        })

        # Contradiction
        cls_txt_c = encode_text_cls(samples[jc]["caption"], model, tokenizer)
        contra_pairs.append({
            "label":   "contradiction",
            "caption": samples[jc]["caption"][:200],
            "cv_img":  concept_vecs[i],
            "cv_txt":  samples[jc]["kw_vec"],
            "cls_img": cls_vecs[i],
            "cls_txt": cls_txt_c,
        })

        if (rank + 1) % 50 == 0:
            print(f"  {rank+1}/{len(sel)}")

    rng.shuffle(normal_pairs); rng.shuffle(contra_pairs)
    cal   = normal_pairs[:50]  + contra_pairs[:50]
    test  = normal_pairs[50:100] + contra_pairs[50:100]
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
        "method":      name,
        "auroc":       round(auroc,4),
        "accuracy":    round(best_acc,4),
        "p_value":     float(p_val),
        "sensitivity": round(tp/(tp+fn),3) if (tp+fn)>0 else 0,
        "specificity": round(tn/(tn+fp),3) if (tn+fp)>0 else 0,
        "contingency": {"tp":tp,"fp":fp,"fn":fn,"tn":tn},
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    rng = random.Random(SEED); np.random.seed(SEED)

    print("\n" + "="*76)
    print("  Experiment F3 — COCO / OpenCLIP: 80 COCO Categories, Hard Contractions")
    print("  Hybrid routing: CLIP image concept + keyword text concept (80-dim)")
    print("="*76 + "\n")

    print("[model] Loading OpenCLIP ViT-B-32 / openai...")
    model, preprocess, tokenizer = load_model()
    pos_embs, neg_embs = build_image_templates(model, tokenizer)
    print(f"[model] Loaded. {N_CONCEPTS} COCO concept templates encoded.\n")

    samples = stream_coco(POOL_SIZE)
    if len(samples) < 400:
        raise RuntimeError(f"Need ≥400 samples, got {len(samples)}")

    cls_vecs, concept_vecs = encode_pool(
        samples, model, preprocess, tokenizer, pos_embs, neg_embs)

    cal, test = build_pairs(
        samples, cls_vecs, concept_vecs,
        model, preprocess, tokenizer, pos_embs, neg_embs, rng)

    print("\n[scoring] Computing all method scores...")
    y_test    = get_labels(test)
    s_routing = score_routing(test)
    s_fusion  = score_fusion(test)
    s_super   = score_supervised(cal, test)

    rt_n = s_routing[y_test==0]; rt_c = s_routing[y_test==1]
    fc_n = s_fusion[y_test==0];  fc_c = s_fusion[y_test==1]

    print(f"\n  Routing (CLIP img + keyword txt):")
    print(f"    normal μ={rt_n.mean():.4f}  contra μ={rt_c.mean():.4f}  "
          f"lift={rt_c.mean()/max(rt_n.mean(),1e-6):.2f}×")
    print(f"  Fusion (CLS cosine):")
    print(f"    normal μ={fc_n.mean():.4f}  contra μ={fc_c.mean():.4f}  "
          f"lift={fc_c.mean()/max(fc_n.mean(),1e-6):.2f}×")

    res_routing = evaluate(s_routing, y_test, "Routing — CLIP image × keyword text L1 (zero-shot)")
    res_fusion  = evaluate(s_fusion,  y_test, "Fusion  — OpenCLIP CLS cosine (zero-shot)")
    res_super   = evaluate(s_super,   y_test, "Supervised — concat + logistic (N=100 labels)")
    results = [res_routing, res_fusion, res_super]

    print("\n" + "="*76)
    print("  EXPERIMENT F3 RESULTS — COCO / OpenCLIP (80 categories, hard negatives)")
    print("  Normal : (image_i, caption of concept-nearest image j)")
    print("  Contra : (image_i, caption of CLS-similar & concept-different image k)")
    print("="*76)
    print(f"\n  {'Method':<50} {'AUROC':>6}  {'Acc':>6}  {'p':>12}  {'Labels?':>10}")
    print("  " + "-"*78)
    for r in results:
        flag = "No" if "zero-shot" in r["method"] else "Yes (N=100)"
        print(f"  {r['method']:<50} {r['auroc']:>6.4f}  {r['accuracy']:>6.1%}  "
              f"{r['p_value']:>12.2e}  {flag:>10}")

    delta = res_routing["auroc"] - res_fusion["auroc"]
    print(f"\n  Routing vs Fusion: AUROC {delta:+.4f}  "
          f"Lift routing={rt_c.mean()/max(rt_n.mean(),1e-6):.2f}×  "
          f"fusion={fc_c.mean()/max(fc_n.mean(),1e-6):.2f}×")

    if delta > 0.05:
        print(f"\n  ✓ ROUTING OUTPERFORMS FUSION (Δ={delta:.3f} AUROC)")
        print("    COCO captions explicitly name present objects → keyword concept")
        print("    vectors are clean and discriminative (same as C2 finding vectors).")
        print("    Hard negatives blind the CLS cosine; routing sees the concept gap.")
        print("    Domain isomorphism: structural routing advantage holds in general VL.")
    elif delta > 0:
        print(f"\n  ~ Routing marginally outperforms fusion (+{delta:.4f} AUROC).")
    else:
        print(f"\n  ~ Routing did not outperform fusion (Δ={delta:.4f}).")

    print("="*76)

    # Example
    idx = np.where(y_test==1)[0][0]
    p   = test[idx]
    print("\n  EXAMPLE contradiction pair from test set:")
    print(f"    Caption: '{p['caption'][:100]}'")
    top3_img = np.argsort(-p["cv_img"])[:3]
    top3_txt = np.argsort(-p["cv_txt"])[:3]
    print(f"    Image top-3 concepts (CLIP): {[COCO_CATS[i]+'=%.2f'%p['cv_img'][i] for i in top3_img]}")
    print(f"    Text  top-3 concepts (kw)  : {[COCO_CATS[i]+'=%.2f'%p['cv_txt'][i] for i in top3_txt]}")
    top3_diff = np.argsort(-np.abs(p["cv_img"] - p["cv_txt"]))[:3]
    print(f"    Most contradicted concepts : {[COCO_CATS[i]+'|img=%.2f,txt=%.2f'%(p['cv_img'][i],p['cv_txt'][i]) for i in top3_diff]}")

    output = {
        "experiment":    "fusion_vs_routing_coco_80cats_hard_negative",
        "timestamp":     datetime.datetime.utcnow().isoformat() + "Z",
        "model":         f"OpenCLIP {MODEL_ID[0]} / {MODEL_ID[1]}",
        "dataset":       "MSCOCO (clip-benchmark/wds_mscoco_captions)",
        "ground_truth":  "Hard contradiction oracle: argmax[CLS_sim × L1_image_concept]",
        "n_concepts":    N_CONCEPTS,
        "concept_type":  "CLIP image templates + keyword text matching (hybrid)",
        "results":       results,
        "distributions": {
            "routing_normal_mean":  round(float(rt_n.mean()), 4),
            "routing_contra_mean":  round(float(rt_c.mean()), 4),
            "routing_lift":         round(float(rt_c.mean()/max(rt_n.mean(),1e-6)), 3),
            "fusion_normal_mean":   round(float(fc_n.mean()), 4),
            "fusion_contra_mean":   round(float(fc_c.mean()), 4),
            "fusion_lift":          round(float(fc_c.mean()/max(fc_n.mean(),1e-6)), 3),
        },
        "note": (
            "Image concept: CLIP image templates for 80 COCO categories. "
            "Text concept: keyword matching (binary). "
            "Hybrid mirrors C2: TXV oracle (image-only) vs BiomedCLIP routing (both sides). "
            "Here: image-CLIP concept vs keyword concept, tested against CLIP CLS fusion."
        ),
    }
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[saved] {OUT_PATH}")


if __name__ == "__main__":
    main()
