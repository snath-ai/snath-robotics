"""
Snath Robotics — CLIPImageEncoder (visual stream for COCO routing proof).
=========================================================================
Encodes images via CLIP ViT-B/32 into a shared concept space R^embed_dim.
Stream A in the V1–V6 divergence routing contract, visual domain.

Architectural role
------------------
The image stream answers: "what does this scene actually contain?"
It is the visual ground truth against which the caption's claim is tested.

Paired with CLIPTextEncoder (Stream B = caption) in the COCO routing
experiment. The routing score D = ||softmax(z_img) - softmax(z_cap)||₁ / √G
detects compositionality failures where the caption claims content the
image doesn't support — invisible to global CLS cosine similarity.

This is the same role VisionEncoder plays in the robotics scenario:
z_vision captures what the scene looks like; the router checks whether
the semantic description (caption / proprioceptive state) agrees.

Concept projection initialisation
----------------------------------
  init_concept_vocabulary()  — CLIP text embeddings of COCO 80 class names × τ=100.
                                Requires embed_dim=80. Use for the main experiment.
  init_pca()                 — PCA directions of image embedding distribution.
                                Works for any embed_dim. Use for smoke tests.
"""
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from encoders.vision_language._clip_backbone import get_clip

# COCO 80-class vocabulary — same order as COCO detection label IDs.
COCO_CLASSES: List[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

BACKBONE_DIM = 512  # CLIP ViT-B/32 image / text embedding dimension


class CLIPImageEncoder(nn.Module):
    """
    CLIP ViT-B/32 image encoder. Stream A of the visual-language routing contract.

    Encodes images into a shared concept space R^embed_dim via:
        1. CLIP image tower  →  512-dim embedding  (backbone frozen at inference)
        2. Concept projection:  512 → embed_dim     (fine-tuned by SIGReg)

    Args:
        embed_dim:   Concept space dimension. 80 for COCO vocabulary init.
        device:      Torch device. Auto-detects CUDA > MPS > CPU.
        temperature: Scale applied to proj.weight in init_concept_vocabulary().
    """

    def __init__(
        self,
        embed_dim:   int   = 80,
        device:      Optional[str] = None,
        temperature: float = 100.0,
    ):
        super().__init__()
        self.embed_dim   = embed_dim
        self.temperature = temperature

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "cpu"   # MPS multi-instance deadlock
            else:
                device = "cpu"
        self.device = torch.device(device)

        self.proj = nn.Linear(BACKBONE_DIM, embed_dim, bias=False)
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.01)
        self.proj.to(self.device)

        self._lora_A: Optional[torch.Tensor] = None
        self._lora_B: Optional[torch.Tensor] = None

    def _clip_image_embed(self, image) -> torch.Tensor:
        """Run CLIP image tower. Returns L2-normalised (512,) tensor."""
        model, preprocess, _ = get_clip(self.device)
        with torch.no_grad():
            feats = model.encode_image(preprocess(image).unsqueeze(0).to(self.device))
        return F.normalize(feats.squeeze(0), dim=-1)

    def init_concept_vocabulary(
        self,
        class_names: List[str] = COCO_CLASSES,
        freeze: bool = False,
    ) -> None:
        """
        Initialise proj.weight with CLIP text embeddings of class_names × temperature.
        Requires embed_dim == len(class_names). For COCO use embed_dim=80.
        """
        if self.embed_dim != len(class_names):
            raise ValueError(
                f"embed_dim={self.embed_dim} must equal len(class_names)="
                f"{len(class_names)} for vocabulary init."
            )
        model, _, tokenizer = get_clip(self.device)
        with torch.no_grad():
            tokens     = tokenizer(class_names).to(self.device)
            text_feats = F.normalize(model.encode_text(tokens), dim=-1)
        self.proj.weight.data = text_feats * self.temperature
        if freeze:
            for p in self.proj.parameters():
                p.requires_grad_(False)

    def init_pca(
        self,
        clip_embeddings: torch.Tensor,
        routing_scale: float = 20.0,
    ) -> float:
        """Initialise proj.weight with PCA directions. Works for any embed_dim."""
        from sklearn.decomposition import PCA
        X = F.normalize(clip_embeddings, dim=-1).cpu().numpy()
        pca = PCA(n_components=self.embed_dim)
        pca.fit(X)
        W = torch.tensor(pca.components_, dtype=torch.float32)
        self.proj.weight.data = (W * routing_scale).to(self.device)
        return float(pca.explained_variance_ratio_.sum())

    def get_confidence(self, z: torch.Tensor) -> float:
        """Peakedness of softmax(z) — identical formula to all Snath encoders."""
        p = torch.softmax(z.flatten(), dim=0)
        n = len(p)
        return max(0.0, float((p.max() - 1.0 / n) / (1.0 - 1.0 / n)))

    def load_lora(self, pt_path: str) -> None:
        """Load a signed LoRA adapter. Applied in forward() as z + (z@A)@B."""
        payload = torch.load(pt_path, map_location="cpu", weights_only=False)
        with torch.no_grad():
            self._lora_A = payload["A"].to(self.device)
            self._lora_B = payload["B"].to(self.device)

    def finetune_projection(
        self,
        clip_embeddings: torch.Tensor,
        lambda_iso: float = 0.1,
        n_epochs:   int   = 300,
        lr:         float = 5e-4,
    ) -> dict:
        """Fine-tune concept projection with SIGReg (VICReg-style isotropy loss)."""
        from dmn.sigreg import SIGRegLoss
        sigreg    = SIGRegLoss(lambda_iso=lambda_iso)
        optimizer = torch.optim.AdamW(self.proj.parameters(), lr=lr)
        x         = clip_embeddings.to(self.device).detach()

        with torch.no_grad():
            isotropy_before = sigreg.isotropy_ratio(self.forward(x))

        loss = torch.tensor(0.0)
        for _ in range(n_epochs):
            optimizer.zero_grad()
            z     = self.forward(x)
            l_var = torch.relu(0.1 - z.std(dim=0)).mean()
            l_cov = sigreg(z)
            loss  = l_var + l_cov
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            isotropy_after = sigreg.isotropy_ratio(self.forward(x))

        return {
            "isotropy_before": isotropy_before,
            "isotropy_after":  isotropy_after,
            "final_loss":      float(loss.item()),
        }

    def forward(self, clip_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            clip_embedding: (B, 512) pre-computed L2-normalised CLIP embeddings.
        Returns:
            z: (B, embed_dim) concept projection, with LoRA if loaded.
        """
        z = self.proj(F.normalize(clip_embedding, dim=-1))
        if self._lora_A is not None:
            z = z + torch.matmul(torch.matmul(z, self._lora_A), self._lora_B)
        return z
