# ============================================================
#  LTL Experiment 1: D_hard vs Random Curriculum
#  Evaluated on Winoground hard negatives
#
#  Lár Training Loop — Sajeev 2026
#  doi: 10.5281/zenodo.20581128
#
#  Runtime: ~20 min on Colab T4 (free tier)
#           ~15 min on Kaggle T4 x2
#
#  Before running:
#  1. Enable GPU: Runtime > Change runtime type > T4
#  2. Get a free HuggingFace token at huggingface.co
#  3. Accept Winoground terms at:
#     https://huggingface.co/datasets/facebook/winoground
#  4. Paste your token in HF_TOKEN below
# ============================================================

# %% [Cell 1] Install
# !pip install -q transformers datasets pillow tqdm scikit-learn

# %% [Cell 2] Imports + config
import os, json, random
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from transformers import CLIPModel, CLIPProcessor

# ── reproducibility ──
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
D_THRESHOLD = 0.25   # D_hard threshold from LTL §3.1
N_TRAIN     = 500    # pairs per condition — raise to 1000 if time allows
EPOCHS      = 100
LR          = 1e-3
DIM         = 512    # CLIP ViT-B/32 embedding dimension

HF_TOKEN = "hf_YOUR_TOKEN_HERE"   # ← paste your token

print(f"Device: {DEVICE}")


# %% [Cell 3] Load frozen CLIP ViT-B/32
print("Loading CLIP ViT-B/32 (frozen)...")
clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False
print("Done.")


# %% [Cell 4] Download COCO val2017
# On Kaggle: dataset is already at /kaggle/input/coco-2017-dataset/
# On Colab:  download below (~1 GB, ~3 min)

COCO_DIR = Path("coco")
IMG_DIR  = COCO_DIR / "val2017"
ANN_FILE = COCO_DIR / "annotations/captions_val2017.json"

if not IMG_DIR.exists():
    print("Downloading COCO val2017 (~1 GB)...")
    COCO_DIR.mkdir(exist_ok=True)
    os.system("wget -q http://images.cocodataset.org/zips/val2017.zip -O coco/val2017.zip")
    os.system("unzip -q coco/val2017.zip -d coco/")
    os.system("wget -q http://images.cocodataset.org/annotations/annotations_trainval2017.zip -O coco/ann.zip")
    os.system("unzip -q coco/ann.zip -d coco/")
    print("Done.")
else:
    print("COCO already present.")


# %% [Cell 5] Build COCO pairs + encode with CLIP
with open(ANN_FILE) as f:
    coco_data = json.load(f)

id2path = {img["id"]: IMG_DIR / img["file_name"]
           for img in coco_data["images"]}

# one caption per image, up to 5 000 pairs
pairs_raw = [(ann["image_id"], ann["caption"])
             for ann in coco_data["annotations"]]
random.shuffle(pairs_raw)

seen, pairs_unique = set(), []
for img_id, caption in pairs_raw:
    if img_id not in seen and id2path[img_id].exists():
        pairs_unique.append((img_id, caption))
        seen.add(img_id)
    if len(pairs_unique) >= 5000:
        break

print(f"Loaded {len(pairs_unique)} COCO pairs")

print("Loading images...")
coco_images = [Image.open(id2path[i]).convert("RGB") for i, _ in tqdm(pairs_unique)]
coco_texts  = [cap for _, cap in pairs_unique]


@torch.no_grad()
def encode_pairs(images, texts, batch_size=64):
    z_imgs, z_txts = [], []
    for i in tqdm(range(0, len(images), batch_size), desc="Encoding COCO"):
        imgs = images[i:i+batch_size]
        txts = texts[i:i+batch_size]
        inp  = clip_processor(text=txts, images=imgs, return_tensors="pt",
                              padding=True, truncation=True, max_length=77).to(DEVICE)
        out  = clip_model(**inp)
        z_imgs.append(out.image_embeds.cpu())
        z_txts.append(out.text_embeds.cpu())
    return torch.cat(z_imgs), torch.cat(z_txts)


Z_img, Z_txt = encode_pairs(coco_images, coco_texts)
print(f"Encoded: Z_img {Z_img.shape}, Z_txt {Z_txt.shape}")


# %% [Cell 6] DivergenceRouter — compute D scores
def compute_D(z_img, z_txt):
    """D = ||v_I - v_T||_1 / sqrt(C)   (LTL eq. 2 / DAS eq. 1)"""
    v_I = F.normalize(z_img, dim=-1)
    v_T = F.normalize(z_txt, dim=-1)
    C   = v_I.shape[-1]
    return (v_I - v_T).abs().sum(dim=-1) / (C ** 0.5)


D_scores    = compute_D(Z_img, Z_txt)
hard_mask   = D_scores >= D_THRESHOLD
hard_frac   = hard_mask.float().mean().item()
hard_idx    = hard_mask.nonzero().squeeze().tolist()
all_idx     = list(range(len(D_scores)))

print(f"\nD scores — mean: {D_scores.mean():.4f}  std: {D_scores.std():.4f}")
print(f"D_hard fraction (D ≥ {D_THRESHOLD}): {hard_frac:.1%}  ({len(hard_idx)} pairs)")

if len(hard_idx) < N_TRAIN:
    print(f"Warning: only {len(hard_idx)} hard pairs; reducing N_TRAIN.")
    N_TRAIN = len(hard_idx)

# ── select training pairs ──
hard_sel = random.sample(hard_idx, N_TRAIN)
rand_sel = random.sample(all_idx,  N_TRAIN)

Z_img_hard = Z_img[hard_sel];  Z_txt_hard = Z_txt[hard_sel]
Z_img_rand = Z_img[rand_sel];  Z_txt_rand = Z_txt[rand_sel]

print(f"\nCondition A (D_hard)  — {N_TRAIN} pairs, mean D = {D_scores[hard_sel].mean():.4f}")
print(f"Condition B (Random)  — {N_TRAIN} pairs, mean D = {D_scores[rand_sel].mean():.4f}")


# %% [Cell 7] JEPA Predictor architecture
class JEPAPredictor(nn.Module):
    """Cross-modal predictor f_theta: z_vision -> z_hat_text"""

    def __init__(self, dim=DIM, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, dim),
        )

    def forward(self, z_v):
        return self.net(z_v)

    @torch.no_grad()
    def prediction_error(self, z_v, z_t):
        """1 - cos(f(z_v), z_t).  High = hard / mismatched."""
        return 1 - F.cosine_similarity(self(z_v), z_t, dim=-1)


# %% [Cell 8] Training
def train_jepa(z_img_tr, z_txt_tr, label=""):
    predictor = JEPAPredictor().to(DEVICE)
    opt       = torch.optim.Adam(predictor.parameters(), lr=LR)
    sched     = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    z_v = z_img_tr.to(DEVICE)
    z_t = z_txt_tr.to(DEVICE)
    losses = []

    for epoch in range(1, EPOCHS + 1):
        predictor.train()
        perm   = torch.randperm(len(z_v))
        z_hat  = predictor(z_v[perm])
        loss   = (1 - F.cosine_similarity(z_hat, z_t[perm].detach(), dim=-1)).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        losses.append(loss.item())
        if epoch % 25 == 0:
            print(f"  [{label}] epoch {epoch:3d}/{EPOCHS}  loss={loss.item():.4f}")

    predictor.eval()
    return predictor, losses


print("Training Condition A — D_hard curriculum...")
pred_hard, losses_hard = train_jepa(Z_img_hard, Z_txt_hard, "D_hard")

print("\nTraining Condition B — Random curriculum...")
pred_rand, losses_rand = train_jepa(Z_img_rand, Z_txt_rand, "Random")


# %% [Cell 9] Load Winoground
# Requires: HF_TOKEN set above + terms accepted at huggingface.co
from datasets import load_dataset

print("\nLoading Winoground test split...")
winoground = load_dataset(
    "facebook/winoground",
    use_auth_token=HF_TOKEN,
    split="test",
)
print(f"Winoground: {len(winoground)} examples  (800 pairs total)")


# %% [Cell 10] Evaluate on Winoground
@torch.no_grad()
def evaluate_winoground(predictor, dataset):
    """
    For each Winoground example (img0, img1, cap0, cap1):
      Correct pairs:   (img0,cap0), (img1,cap1) → label 0
      Incorrect pairs: (img1,cap0), (img0,cap1) → label 1
    AUROC: does prediction error correctly identify mismatched pairs?
    """
    predictor.eval()
    all_errors, all_labels = [], []

    for ex in tqdm(dataset, desc="Winoground eval"):
        img0 = ex["image_0"].convert("RGB")
        img1 = ex["image_1"].convert("RGB")
        cap0 = ex["caption_0"]
        cap1 = ex["caption_1"]

        # encode all 4 (image, caption) combinations in one batch
        inp = clip_processor(
            text=[cap0, cap1, cap0, cap1],
            images=[img0, img0, img1, img1],
            return_tensors="pt", padding=True,
            truncation=True, max_length=77,
        ).to(DEVICE)
        out    = clip_model(**inp)
        z_imgs = out.image_embeds   # [4, 512]
        z_txts = out.text_embeds    # [4, 512]

        # pairs:      (img0,cap0) (img0,cap1) (img1,cap0) (img1,cap1)
        pair_labels = [    0,          1,          1,          0     ]

        for idx, lbl in enumerate(pair_labels):
            err = predictor.prediction_error(
                z_imgs[idx].unsqueeze(0),
                z_txts[idx].unsqueeze(0),
            ).item()
            all_errors.append(err)
            all_labels.append(lbl)

    auroc = roc_auc_score(all_labels, all_errors)
    return auroc, all_errors, all_labels


print("Evaluating D_hard predictor...")
auroc_hard, err_hard, lbl_hard = evaluate_winoground(pred_hard, winoground)

print("Evaluating Random predictor...")
auroc_rand, err_rand, lbl_rand = evaluate_winoground(pred_rand, winoground)


# %% [Cell 11] Results
print("\n" + "=" * 52)
print("  LTL EXPERIMENT 1 — RESULTS")
print("=" * 52)
print(f"  Encoder          : CLIP ViT-B/32 (frozen)")
print(f"  Training pairs   : {N_TRAIN} per condition")
print(f"  D threshold      : {D_THRESHOLD}  (hard fraction: {hard_frac:.1%})")
print(f"  Epochs           : {EPOCHS}")
print(f"  Eval benchmark   : Winoground ({len(winoground)} examples)")
print("-" * 52)
print(f"  Condition A — D_hard curriculum : AUROC = {auroc_hard:.4f}")
print(f"  Condition B — Random curriculum : AUROC = {auroc_rand:.4f}")
print(f"  Δ  (D_hard − Random)            :       {auroc_hard - auroc_rand:+.4f}")
print("=" * 52)

delta = auroc_hard - auroc_rand
if delta > 0.02:
    print(f"\n  ✓ D_hard curriculum beats random by {delta:.4f} AUROC.")
    print("    The routing signal is a better curriculum than uniform sampling.")
elif delta > 0:
    print(f"\n  ✓ D_hard curriculum edges random (+{delta:.4f}).")
    print("    Try N_TRAIN=1000 or EPOCHS=200 for a stronger signal.")
else:
    print(f"\n  ✗ Random matched or beat D_hard ({delta:+.4f}).")
    print("    Debug: check D_hard fraction and threshold τ.")

# save
results = {
    "auroc_hard":    auroc_hard,
    "auroc_random":  auroc_rand,
    "delta":         delta,
    "n_train":       N_TRAIN,
    "d_threshold":   D_THRESHOLD,
    "hard_fraction": hard_frac,
    "epochs":        EPOCHS,
    "encoder":       "openai/clip-vit-base-patch32",
    "eval_set":      "facebook/winoground",
    "seed":          SEED,
}
with open("ltl_exp1_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\n  Results saved → ltl_exp1_results.json")


# %% [Cell 12] Optional — also run on ARO (no auth required)
# ARO tests compositional understanding: Attribution, Relation, Order
# pip install git+https://github.com/mertyg/vision-language-models-are-bows
#
# from aro_datasets import VG_Attribution, VG_Relation
# Then evaluate similarly: prediction_error on (image, correct_caption) vs (image, foil_caption)
# AUROC: does higher error predict the foil?
