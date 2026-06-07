"""
Snath Robotics — COCO / CLIP Visual-Semantic Routing Proof.
============================================================
Proves the V1–V6 routing contract and JEPA world model on real-world data.

Domain mapping
--------------
  Stream A  : CLIP ViT-B/32 image embeddings  (CLIPImageEncoder)
              → "what the camera actually sees"
  Stream B  : CLIP ViT-B/32 caption embeddings (CLIPTextEncoder)
              → "what the semantic description claims is in the scene"
  Oracle    : matched (label=1) vs. hard-mismatched (label=0) image-caption pairs
  JEPA goal : train f_θ(z_img) → ẑ_cap with no labels — prediction error
              distinguishes mismatches the routing score might miss

This is the same contract as the physical robotics scenario:
  z_vision  (camera)        ←→ z_image  (CLIP image)
  z_proprio (body physics)  ←→ z_cap    (CLIP caption)
  D_pred (prediction error) ←→ 1 - cos(f_θ(z_img), z_cap)

The COCO dataset provides ground-truth match/mismatch labels, making it
the cleanest available benchmark for measuring routing AUROC independently
of the annotation-free physical loop.

Pipeline
--------
  Phase 0  CLIP embeddings: download COCO or use smoke-test synthetic pairs
  Phase 1  Concept projection init (vocabulary τ=100 or PCA)
  Phase 2  Oracle: build balanced matched / hard-mismatched evaluation set
  Phase 3  Baseline routing AUROC (before any training)
  Phase 4  SIGReg projection fine-tuning (λ sweep or fixed)
  Phase 5  InfoNCE + SIGReg continued training (full run only)
  Phase 6  Post-SIGReg AUROC  →  ρ = AUROC_SIGReg / AUROC_baseline
  Phase 7  JEPA predictor: train on matched pairs only (no labels)
           → AUROC of prediction error vs mismatch label
  Phase 8  D_hard mining → RoboticsDMN consolidation → LoRA adapters
  Phase 9  Post-LoRA AUROC

Usage
-----
  # Smoke test — no COCO download, N=400 synthetic pairs, ~2 min:
  python experiments/coco_proof.py --smoke-test

  # Full run — COCO train2017, ~45 min on GPU:
  python experiments/coco_proof.py --download
  python experiments/coco_proof.py

  # SIGReg λ ablation:
  python experiments/coco_proof.py --lambda-iso 0.01
  python experiments/coco_proof.py --lambda-iso 0.10
  python experiments/coco_proof.py --lambda-iso 1.00

Success criterion (pre-registered in AIA §Experiment 3)
---------------------------------------------------------
  ρ = AUROC_SIGReg / AUROC_baseline > 1.15.
  JEPA AUROC > AUROC_baseline (label-free predictor beats random).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from encoders.clip_image_encoder import CLIPImageEncoder, COCO_CLASSES
from encoders.clip_text_encoder  import CLIPTextEncoder
from divergence_router           import DivergenceRouter
from dhard                       import DHardQueue, RoboticsDHardEvent
from dmn.robotics_dmn            import RoboticsDMN
from dmn.sigreg                  import SIGRegLoss
from models.jepa_predictor       import JEPAPredictor, train_predictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE        = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR     = _HERE / "data" / "coco"
CACHE_DIR    = _HERE / "data" / "coco_clip_cache"
RESULTS_DIR  = _HERE / "experiments" / "coco_results"
ADAPTER_DIR  = _HERE / "models" / "coco_adapters"
DHARD_PATH   = _HERE / "coco_d_hard.jsonl"

# ── Routing thresholds ────────────────────────────────────────────────────────
TAU_LOW   = 0.25
TAU_HIGH  = 0.60
DELTA     = 0.25


# ==============================================================================
# Phase 0 — Download COCO
# ==============================================================================

def download_coco(data_dir: Path = DATA_DIR, split: str = "train2017") -> None:
    """
    Download MS-COCO 2017 annotations (241MB). Images are ~18GB and fetched
    on-demand during embedding precomputation. Set data_dir to a persistent
    path on Colab (Google Drive) to avoid re-downloading across sessions.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    ann_url  = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    img_url  = f"http://images.cocodataset.org/zips/{split}.zip"
    ann_path = data_dir / "annotations_trainval2017.zip"
    img_path = data_dir / f"{split}.zip"

    import urllib.request, zipfile
    if not (data_dir / "annotations").exists():
        log.info("Downloading COCO annotations (~241MB)...")
        urllib.request.urlretrieve(ann_url, ann_path)
        with zipfile.ZipFile(ann_path) as z:
            z.extractall(data_dir)
        ann_path.unlink()

    img_dir = data_dir / split
    if not img_dir.exists():
        log.info(f"Downloading COCO {split} images (~18GB)...")
        urllib.request.urlretrieve(img_url, img_path)
        with zipfile.ZipFile(img_path) as z:
            z.extractall(data_dir)
        img_path.unlink()

    log.info(f"COCO {split}: {len(list(img_dir.glob('*.jpg')))} images in {img_dir}")


# ==============================================================================
# Phase 0 — Pre-compute CLIP embeddings
# ==============================================================================

def precompute_coco_embeddings(
    data_dir:   Path = DATA_DIR,
    cache_dir:  Path = CACHE_DIR,
    split:      str  = "train2017",
    max_pairs:  Optional[int] = None,
    batch_size: int  = 64,
    device:     str  = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor, List[dict]]:
    """
    Pre-compute CLIP image + caption embeddings for COCO pairs (one caption per image).

    Returns:
        img_embs:  (N, 512) image embeddings
        cap_embs:  (N, 512) caption embeddings
        metadata:  list of {"image_id", "caption", "image_path"}

    Caches to cache_dir on first run; reloads from cache on subsequent calls.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    img_cache  = cache_dir / f"coco_{split}_img.pt"
    cap_cache  = cache_dir / f"coco_{split}_cap.pt"
    meta_cache = cache_dir / f"coco_{split}_meta.json"

    if img_cache.exists() and cap_cache.exists():
        log.info(f"Loading CLIP embeddings from cache ({img_cache})...")
        img_embs = torch.load(img_cache, map_location="cpu", weights_only=True)
        cap_embs = torch.load(cap_cache, map_location="cpu", weights_only=True)
        metadata = json.loads(meta_cache.read_text())
        if max_pairs:
            img_embs = img_embs[:max_pairs]
            cap_embs = cap_embs[:max_pairs]
            metadata = metadata[:max_pairs]
        log.info(f"Loaded {len(metadata)} pairs from cache.")
        return img_embs, cap_embs, metadata

    ann_file = data_dir / "annotations" / f"captions_{split}.json"
    assert ann_file.exists(), f"Missing {ann_file}. Run with --download first."
    with open(ann_file) as f:
        coco_ann = json.load(f)

    id_to_caption: Dict[int, str] = {}
    for ann in coco_ann["annotations"]:
        iid = ann["image_id"]
        if iid not in id_to_caption:
            id_to_caption[iid] = ann["caption"]

    id_to_file = {img["id"]: img["file_name"] for img in coco_ann["images"]}

    pairs = [
        {
            "image_id":   iid,
            "caption":    id_to_caption[iid],
            "image_path": str(data_dir / split / id_to_file[iid]),
        }
        for iid in id_to_caption
        if iid in id_to_file
    ]
    if max_pairs:
        pairs = pairs[:max_pairs]

    from encoders._clip_backbone import get_clip
    from PIL import Image
    dev = torch.device(device)
    model, preprocess, tokenizer = get_clip(dev)

    img_embs_list, cap_embs_list = [], []
    valid_pairs = []

    for i in range(0, len(pairs), batch_size):
        batch = pairs[i : i + batch_size]
        try:
            images   = [preprocess(Image.open(p["image_path"]).convert("RGB"))
                        for p in batch]
            captions = [p["caption"] for p in batch]
            img_in   = torch.stack(images).to(dev)
            cap_in   = tokenizer(captions).to(dev)
            with torch.no_grad():
                ie = F.normalize(model.encode_image(img_in), dim=-1)
                ce = F.normalize(model.encode_text(cap_in),  dim=-1)
            img_embs_list.append(ie.cpu())
            cap_embs_list.append(ce.cpu())
            valid_pairs.extend(batch)
        except Exception as exc:
            log.warning(f"Batch {i}–{i+batch_size} failed: {exc}")
            continue

        if (i // batch_size + 1) % 50 == 0:
            log.info(f"  Encoded {len(valid_pairs)}/{len(pairs)} pairs...")

    img_embs = torch.cat(img_embs_list, dim=0)
    cap_embs = torch.cat(cap_embs_list, dim=0)
    torch.save(img_embs, img_cache)
    torch.save(cap_embs, cap_cache)
    meta_cache.write_text(json.dumps(valid_pairs, indent=2))
    log.info(f"Cached {len(valid_pairs)} COCO pairs → {cache_dir}")
    return img_embs, cap_embs, valid_pairs


# ==============================================================================
# Phase 2 — Oracle: matched vs hard-mismatched pairs
# ==============================================================================

def build_oracle_pairs(
    img_embs: torch.Tensor,
    cap_embs: torch.Tensor,
    metadata: List[dict],
    n_pairs:  int = 5000,
    seed:     int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """
    Build a balanced oracle evaluation set.

    Matched    (label=1): (image_i, caption_i) — correct pairing
    Mismatched (label=0): (image_i, caption_j≠i) — hardest negative per image
                          (highest cross-image cosine similarity)

    Returns:
        oracle_img:  (2*n_pairs, 512)
        oracle_cap:  (2*n_pairs, 512)
        labels:      (2*n_pairs,)  int array
    """
    rng   = np.random.default_rng(seed)
    N     = len(metadata)
    n_use = min(n_pairs, N)

    idx     = rng.choice(N, size=n_use, replace=False)
    img_sel = img_embs[idx]
    cap_sel = cap_embs[idx]

    sim_matrix    = (img_sel @ cap_sel.T).numpy()
    np.fill_diagonal(sim_matrix, -1.0)
    mismatch_idx  = sim_matrix.argmax(axis=1)

    oracle_img = torch.cat([img_sel, img_sel], dim=0)
    oracle_cap = torch.cat([cap_sel, cap_sel[mismatch_idx]], dim=0)
    labels     = np.array([1] * n_use + [0] * n_use, dtype=int)

    perm       = rng.permutation(len(labels))
    oracle_img = oracle_img[perm]
    oracle_cap = oracle_cap[perm]
    labels     = labels[perm]

    balance = labels.mean()
    log.info(f"Oracle: {len(labels)} pairs, balance={balance:.3f} (target 0.50)")
    assert 0.45 <= balance <= 0.55, f"Oracle imbalanced (mean={balance:.3f})"
    return oracle_img, oracle_cap, labels


# ==============================================================================
# Phase 3 — Routing AUROC
# ==============================================================================

def route_and_auroc(
    enc_img:    CLIPImageEncoder,
    enc_cap:    CLIPTextEncoder,
    oracle_img: torch.Tensor,
    oracle_cap: torch.Tensor,
    labels:     np.ndarray,
    router:     DivergenceRouter,
    batch_size: int = 256,
) -> Tuple[float, float, np.ndarray]:
    """
    Route all oracle pairs. Returns (AUROC, trigger_rate, d_scores).

    High D → mismatched (label=0), so AUROC = AUC(D vs (1-labels)).
    """
    from sklearn.metrics import roc_auc_score

    d_scores = []
    n_replan = 0

    enc_img.eval()
    enc_cap.eval()

    with torch.no_grad():
        for i in range(0, len(labels), batch_size):
            img_b  = oracle_img[i : i + batch_size].to(enc_img.device)
            cap_b  = oracle_cap[i : i + batch_size].to(enc_cap.device)
            z_img  = enc_img(img_b)
            z_cap  = enc_cap(cap_b)
            for j in range(z_img.shape[0]):
                result = router.route(
                    z_vision  = z_img[j],
                    z_proprio = z_cap[j],
                )
                d_scores.append(result.divergence)
                if result.decision.name == "TRIGGER_REPLAN":
                    n_replan += 1

    d_arr        = np.array(d_scores)
    trigger_rate = n_replan / len(labels)
    auroc        = roc_auc_score(1 - labels, d_arr)
    return auroc, trigger_rate, d_arr


# ==============================================================================
# Phase 4 — SIGReg projection-only fine-tuning
# ==============================================================================

def run_sigreg_projection(
    enc_img:    CLIPImageEncoder,
    enc_cap:    CLIPTextEncoder,
    img_embs:   torch.Tensor,
    cap_embs:   torch.Tensor,
    lambda_iso: float = 0.1,
    n_epochs:   int   = 300,
) -> Tuple[dict, dict]:
    """Fine-tune both projection heads with SIGReg independently."""
    log.info(f"SIGReg projection fine-tuning (λ={lambda_iso}, {n_epochs} epochs)...")
    res_img = enc_img.finetune_projection(img_embs, lambda_iso=lambda_iso,
                                          n_epochs=n_epochs)
    log.info(f"  Image:   {res_img['isotropy_before']:.4f} → {res_img['isotropy_after']:.4f}")
    res_cap = enc_cap.finetune_projection(cap_embs, lambda_iso=lambda_iso,
                                          n_epochs=n_epochs)
    log.info(f"  Caption: {res_cap['isotropy_before']:.4f} → {res_cap['isotropy_after']:.4f}")
    return res_img, res_cap


# ==============================================================================
# Phase 5 — InfoNCE + SIGReg continued training
# ==============================================================================

def run_infonce_sigreg(
    enc_img:     CLIPImageEncoder,
    enc_cap:     CLIPTextEncoder,
    img_embs:    torch.Tensor,
    cap_embs:    torch.Tensor,
    lambda_iso:  float = 0.1,
    n_epochs:    int   = 2,
    batch_size:  int   = 256,
    lr:          float = 1e-5,
    temperature: float = 0.07,
) -> List[float]:
    """
    InfoNCE + SIGReg continued training on matched COCO pairs.
    Backbones remain frozen; only the concept projection heads are trained.
    """
    sigreg    = SIGRegLoss(lambda_iso=lambda_iso)
    optimizer = torch.optim.AdamW(
        list(enc_img.proj.parameters()) + list(enc_cap.proj.parameters()),
        lr=lr, weight_decay=0.01,
    )
    N           = len(img_embs)
    epoch_losses = []

    for epoch in range(n_epochs):
        perm       = torch.randperm(N)
        total_loss = 0.0
        n_batches  = 0

        for i in range(0, N, batch_size):
            idx   = perm[i : i + batch_size]
            x_img = img_embs[idx].to(enc_img.device)
            x_cap = cap_embs[idx].to(enc_cap.device)

            z_img   = enc_img(x_img)
            z_cap   = enc_cap(x_cap)
            z_img_n = F.normalize(z_img, dim=-1)
            z_cap_n = F.normalize(z_cap, dim=-1)
            logits  = (z_img_n @ z_cap_n.T) / temperature
            B       = logits.shape[0]
            targets = torch.arange(B, device=logits.device)
            l_nce   = (F.cross_entropy(logits, targets) +
                       F.cross_entropy(logits.T, targets)) / 2
            l_iso   = sigreg(z_img) + sigreg(z_cap)
            loss    = l_nce + l_iso

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(enc_img.proj.parameters()) + list(enc_cap.proj.parameters()),
                max_norm=1.0,
            )
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        avg_loss = total_loss / max(n_batches, 1)
        epoch_losses.append(avg_loss)
        log.info(f"  Epoch {epoch+1}/{n_epochs}: loss={avg_loss:.4f}")

    return epoch_losses


# ==============================================================================
# Phase 7 — JEPA predictor (world model, label-free)
# ==============================================================================

def run_jepa_predictor(
    enc_img:    CLIPImageEncoder,
    enc_cap:    CLIPTextEncoder,
    img_embs:   torch.Tensor,
    cap_embs:   torch.Tensor,
    oracle_img: torch.Tensor,
    oracle_cap: torch.Tensor,
    labels:     np.ndarray,
    n_epochs:   int   = 200,
    lr:         float = 1e-3,
    batch_size: int   = 128,
) -> Tuple[dict, np.ndarray]:
    """
    Train JEPA predictor z_img → ẑ_cap with zero labels.

    Runs on raw 512-dim CLIP embeddings, not on the concept projections.
    The routing uses projected concept space (DivergenceRouter); the JEPA
    uses the full CLIP space where matched/mismatched pairs are distinguishable.

    Physical:  z_vision → ẑ_proprio  |  error = surface looked safe, body says slip
    COCO:      z_img    → ẑ_cap      |  error = image shows X, caption claims Y
    """
    from sklearn.metrics import roc_auc_score

    log.info("=== JEPA predictor (label-free world model, 512-dim CLIP space) ===")
    clip_dim = img_embs.shape[1]   # 512
    device   = enc_img.device

    predictor = JEPAPredictor(embed_dim=clip_dim).to(device)

    # Train on raw matched CLIP pairs — no labels, no concept projection
    train_stats = train_predictor(
        predictor,
        img_embs.to(device).cpu(),
        cap_embs.to(device).cpu(),
        n_epochs=n_epochs, lr=lr, batch_size=batch_size,
    )

    # Evaluate on oracle (matched + hard-mismatched) in raw CLIP space
    predictor.eval()
    pred_errors = []
    with torch.no_grad():
        for i in range(0, len(labels), batch_size):
            z_img_b = oracle_img[i : i + batch_size].to(device)
            z_cap_b = oracle_cap[i : i + batch_size].to(device)
            err     = predictor.prediction_error(z_img_b, z_cap_b)
            pred_errors.append(err.cpu())
    pred_errors = torch.cat(pred_errors).numpy()

    auroc_pred = roc_auc_score(1 - labels, pred_errors)
    log.info(f"  Predictor AUROC (label-free, 512-dim): {auroc_pred:.4f}  "
             f"(mean error: {pred_errors.mean():.4f})")
    train_stats["auroc"] = auroc_pred
    return train_stats, pred_errors


# ==============================================================================
# Phase 8 — D_hard mining + RoboticsDMN consolidation
# ==============================================================================

def mine_dhard_and_consolidate(
    enc_img:       CLIPImageEncoder,
    enc_cap:       CLIPTextEncoder,
    img_embs:      torch.Tensor,
    cap_embs:      torch.Tensor,
    metadata:      List[dict],
    oracle_labels: np.ndarray,
    router:        DivergenceRouter,
    dhard_path:    Path = DHARD_PATH,
    adapter_dir:   Path = ADAPTER_DIR,
    lambda_iso:    float = 0.0,
) -> List[dict]:
    """
    Route all COCO pairs, log D_hard events, run RoboticsDMN consolidation.

    Mapping to Robotics event schema:
      z_vision      → image embedding  (visual ground truth)
      z_proprio     → caption embedding (semantic claim)
      failure_class → "compositionality" (image-caption concept mismatch)
      winner        → "vision"  if label=0 (caption wrong, image is GT)
                      "proprio" if label=1 (caption correctly described image)
      scenario_id   → COCO image_id string

    Returns list of adapter metadata dicts.
    """
    adapter_dir.mkdir(parents=True, exist_ok=True)
    if dhard_path.exists():
        dhard_path.unlink()
    queue = DHardQueue(str(dhard_path))

    enc_img.eval()
    enc_cap.eval()
    n_dhard = 0

    with torch.no_grad():
        for i, label in enumerate(oracle_labels):
            x_img  = img_embs[i].unsqueeze(0).to(enc_img.device)
            x_cap  = cap_embs[i].unsqueeze(0).to(enc_cap.device)
            z_img  = enc_img(x_img).squeeze(0)
            z_cap  = enc_cap(x_cap).squeeze(0)
            result = router.route(z_vision=z_img, z_proprio=z_cap)

            is_hard = (
                result.decision.name in ("TRIGGER_REPLAN", "STRUCTURAL_IMPASSE")
                and result.divergence >= router.delta
            )
            if not is_hard:
                continue

            # label=1 → caption correctly described the image → proprio (caption) was right
            # label=0 → caption is wrong for this image → vision (image) was the GT
            winner        = "proprio" if label == 1 else "vision"
            decision_str  = result.decision.name

            ev = RoboticsDHardEvent(
                z_vision      = z_img.cpu().tolist(),
                z_proprio     = z_cap.cpu().tolist(),
                divergence    = result.divergence,
                decision      = decision_str,
                failure_class = "compositionality",
                scenario_id   = str(metadata[i].get("image_id", i)),
                winner        = winner,
            )
            queue.push(ev)
            n_dhard += 1

    q_stats = queue.stats()
    log.info(f"D_hard: {n_dhard} events logged, {q_stats['resolved']} resolved → {dhard_path}")

    dmn   = RoboticsDMN(queue_path=str(dhard_path), adapter_dir=str(adapter_dir))
    built = dmn.consolidate(lambda_iso=lambda_iso, verbose=True)
    log.info(f"DMN: built {len(built)} adapter(s)")
    return built


# ==============================================================================
# Full experiment runner
# ==============================================================================

def _resolve_device(requested: Optional[str]) -> str:
    if requested is None or requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA requested but not available — falling back to CPU.")
        return "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        log.warning("MPS requested but not available — falling back to CPU.")
        return "cpu"
    return requested


def run_full_experiment(
    smoke_test:     bool  = False,
    lambda_iso:     float = 0.1,
    embed_dim:      int   = 80,
    use_vocab_init: bool  = True,
    device:         Optional[str] = None,
    sigreg_epochs:  int   = 300,
    infonce_epochs: int   = 2,
    split:          str   = "val2017",
    max_pairs:      Optional[int] = None,
) -> dict:
    """
    Run the complete COCO / CLIP routing and JEPA proof.

    Args:
        smoke_test:     N=400 synthetic pairs, no COCO download (~2 min).
        lambda_iso:     SIGReg weight.
        embed_dim:      Concept space dimension (80 for vocabulary init).
        use_vocab_init: COCO 80-class vocabulary projection (τ=100).
        device:         "cuda" / "mps" / "cpu" / None (auto).
        sigreg_epochs:  Epochs for SIGReg projection fine-tuning.
        infonce_epochs: Epochs for InfoNCE+SIGReg continued training.
    """
    device = _resolve_device(device)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    log.info(f"Initialising CLIP encoders (embed_dim={embed_dim}, device={device})...")
    enc_img = CLIPImageEncoder(embed_dim=embed_dim, device=device)
    enc_cap = CLIPTextEncoder(embed_dim=embed_dim,  device=device)
    router  = DivergenceRouter(
        tau_low=TAU_LOW, tau_high=TAU_HIGH, delta=DELTA,
    )

    # ── Embeddings ────────────────────────────────────────────────────────────
    ann_file = DATA_DIR / "annotations" / f"captions_{split}.json"
    if smoke_test and not ann_file.exists():
        embed_dim = 8
        enc_img   = CLIPImageEncoder(embed_dim=embed_dim, device=device)
        enc_cap   = CLIPTextEncoder(embed_dim=embed_dim,  device=device)
        N = 400
        log.info(f"Smoke test: {N} synthetic pairs (embed_dim=8, no COCO)...")
        torch.manual_seed(42)
        img_embs   = F.normalize(torch.randn(N, 512), dim=-1)
        matched    = F.normalize(img_embs[:N//2] + 0.3 * torch.randn(N//2, 512), dim=-1)
        mismatched = F.normalize(torch.randn(N//2, 512), dim=-1)
        cap_embs   = torch.cat([matched, mismatched], dim=0)
        metadata   = [{"image_id": i, "caption": f"synthetic_{i}", "image_path": ""}
                      for i in range(N)]
    else:
        _max = 200 if smoke_test else max_pairs
        img_embs, cap_embs, metadata = precompute_coco_embeddings(
            split=split, max_pairs=_max, device=device,
        )
    log.info(f"Loaded {len(metadata)} pairs.")

    # ── Concept projection init ───────────────────────────────────────────────
    if use_vocab_init and embed_dim == 80 and ann_file.exists() and not smoke_test:
        log.info("Initialising concept projection with COCO vocabulary (τ=100)...")
        enc_img.init_concept_vocabulary(freeze=False)
        enc_cap.init_concept_vocabulary(freeze=False)
    else:
        log.info(f"Initialising concept projection with PCA (embed_dim={embed_dim})...")
        ev_img = enc_img.init_pca(img_embs)
        ev_cap = enc_cap.init_pca(cap_embs)
        log.info(f"  PCA explained var: image={ev_img:.3f}  caption={ev_cap:.3f}")

    # ── Oracle ────────────────────────────────────────────────────────────────
    oracle_img, oracle_cap, oracle_labels = build_oracle_pairs(
        img_embs, cap_embs, metadata,
        n_pairs=min(2500, len(metadata) // 2),
    )

    # ── Phase 3: baseline AUROC ───────────────────────────────────────────────
    log.info("=== BASELINE ROUTING ===")
    auroc_before, trig_before, _ = route_and_auroc(
        enc_img, enc_cap, oracle_img, oracle_cap, oracle_labels, router,
    )
    log.info(f"  AUROC before:    {auroc_before:.4f}  TRIGGER_REPLAN: {trig_before:.1%}")

    # ── Phase 4–5: SIGReg ────────────────────────────────────────────────────
    log.info(f"=== SIGReg (λ={lambda_iso}) ===")
    iso_img, iso_cap = run_sigreg_projection(
        enc_img, enc_cap, img_embs, cap_embs,
        lambda_iso=lambda_iso, n_epochs=sigreg_epochs,
    )
    if not smoke_test and infonce_epochs > 0:
        log.info(f"=== InfoNCE + SIGReg ({infonce_epochs} epochs) ===")
        run_infonce_sigreg(
            enc_img, enc_cap, img_embs, cap_embs,
            lambda_iso=lambda_iso, n_epochs=infonce_epochs,
        )

    # ── Phase 6: post-SIGReg AUROC ───────────────────────────────────────────
    log.info("=== POST-SIGReg ROUTING ===")
    auroc_sigreg, trig_sigreg, _ = route_and_auroc(
        enc_img, enc_cap, oracle_img, oracle_cap, oracle_labels, router,
    )
    log.info(f"  AUROC SIGReg:    {auroc_sigreg:.4f}  TRIGGER_REPLAN: {trig_sigreg:.1%}")

    # ── Phase 7: JEPA predictor (label-free) ─────────────────────────────────
    jepa_epochs = 50 if smoke_test else 200
    jepa_stats, pred_errors = run_jepa_predictor(
        enc_img, enc_cap, img_embs, cap_embs,
        oracle_img, oracle_cap, oracle_labels,
        n_epochs=jepa_epochs,
    )
    auroc_predictor = jepa_stats["auroc"]

    # ── Phase 8: D_hard mining + DMN ─────────────────────────────────────────
    log.info("=== D_hard mining + RoboticsDMN consolidation ===")
    built = mine_dhard_and_consolidate(
        enc_img, enc_cap, img_embs, cap_embs,
        metadata, oracle_labels, router,
        lambda_iso=lambda_iso,
    )

    # ── Phase 9: apply adapters + final AUROC ────────────────────────────────
    for meta in built:
        pt_path = meta.get("pt_path", "")
        if not pt_path or not Path(pt_path).exists():
            continue
        target = meta.get("target_encoder", "")
        if target == "vision":
            enc_img.load_lora(pt_path)
            log.info(f"  LoRA → enc_img  ({meta['failure_class']}, n={meta['n_events']})")
        elif target == "proprio":
            enc_cap.load_lora(pt_path)
            log.info(f"  LoRA → enc_cap  ({meta['failure_class']}, n={meta['n_events']})")

    log.info("=== POST-LoRA ROUTING ===")
    auroc_after, trig_after, _ = route_and_auroc(
        enc_img, enc_cap, oracle_img, oracle_cap, oracle_labels, router,
    )
    log.info(f"  AUROC after LoRA: {auroc_after:.4f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    rho      = auroc_sigreg / auroc_before if auroc_before > 0 else float("nan")
    rho_lora = auroc_after  / auroc_before if auroc_before > 0 else float("nan")

    print("\n" + "=" * 62)
    print("  RESULTS — Snath Robotics COCO / CLIP ViT-B/32")
    print("=" * 62)
    print(f"  N pairs:            {len(metadata)}")
    print(f"  embed_dim:          {embed_dim}")
    print(f"  lambda_iso:         {lambda_iso}")
    print(f"  Isotropy image:     {iso_img['isotropy_before']:.4f} → {iso_img['isotropy_after']:.4f}")
    print(f"  Isotropy caption:   {iso_cap['isotropy_before']:.4f} → {iso_cap['isotropy_after']:.4f}")
    print(f"  AUROC baseline:     {auroc_before:.4f}")
    print(f"  AUROC SIGReg:       {auroc_sigreg:.4f}   ρ = {rho:.4f}")
    print(f"  AUROC after LoRA:   {auroc_after:.4f}   ρ_LoRA = {rho_lora:.4f}")
    print(f"  AUROC predictor:    {auroc_predictor:.4f}   [JEPA label-free]")
    print(f"  TRIGGER_REPLAN:     {trig_sigreg:.1%}")
    print(f"  Adapters built:     {len(built)}")
    print(f"  ρ > 1.15 (AIA §3): {'✓ PASSED' if rho > 1.15 else '✗ not yet'}")
    print(f"  JEPA > baseline:   {'✓ PROVEN' if auroc_predictor > auroc_before else '✗'}")
    print("=" * 62)

    import datetime
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    results = {
        "n_pairs":          len(metadata),
        "embed_dim":        embed_dim,
        "lambda_iso":       lambda_iso,
        "isotropy_img":     {"before": iso_img["isotropy_before"],
                             "after":  iso_img["isotropy_after"]},
        "isotropy_cap":     {"before": iso_cap["isotropy_before"],
                             "after":  iso_cap["isotropy_after"]},
        "auroc_before":     auroc_before,
        "auroc_sigreg":     auroc_sigreg,
        "auroc_after_lora": auroc_after,
        "auroc_predictor":  auroc_predictor,
        "jepa":             jepa_stats,
        "rho":              rho,
        "rho_lora":         rho_lora,
        "trigger_rate":     trig_sigreg,
        "n_adapters":       len(built),
    }
    out_path = RESULTS_DIR / f"coco_proof_{ts}.json"
    out_path.write_text(json.dumps(results, indent=2))
    log.info(f"Results saved → {out_path}")
    return results


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Snath Robotics — COCO/CLIP routing and JEPA proof"
    )
    parser.add_argument("--smoke-test",     action="store_true",
                        help="N=400 synthetic pairs, no COCO download (~2 min)")
    parser.add_argument("--lambda-iso",     type=float, default=0.1)
    parser.add_argument("--embed-dim",      type=int,   default=80)
    parser.add_argument("--no-vocab-init",  action="store_true",
                        help="PCA init instead of COCO vocabulary")
    parser.add_argument("--device",         default=None,
                        help="cuda / mps / cpu (default: auto)")
    parser.add_argument("--sigreg-epochs",  type=int,   default=300)
    parser.add_argument("--infonce-epochs", type=int,   default=2)
    parser.add_argument("--download",       action="store_true",
                        help="Download COCO split first (val2017=780MB, train2017=18GB)")
    parser.add_argument("--split",          default="val2017",
                        help="COCO split: val2017 (default, 5k images) or train2017 (118k images)")
    parser.add_argument("--max-pairs",      type=int, default=None,
                        help="Cap number of pairs (useful for quick GPU tests)")
    args = parser.parse_args()

    if args.download:
        download_coco(split=args.split)

    run_full_experiment(
        smoke_test      = args.smoke_test,
        lambda_iso      = args.lambda_iso,
        embed_dim       = args.embed_dim,
        use_vocab_init  = not args.no_vocab_init,
        device          = args.device,
        sigreg_epochs   = args.sigreg_epochs,
        infonce_epochs  = args.infonce_epochs,
        split           = args.split,
        max_pairs       = args.max_pairs,
    )
