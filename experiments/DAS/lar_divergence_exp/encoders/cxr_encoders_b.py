"""
Option B encoders — both streams use BioBERT in the same embedding space.

Stream A: MeSH image-finding tags → natural-language description → BioBERT CLS
Stream B: Radiology report text (findings + impression)       → BioBERT CLS

Both embeddings live in BioBERT's representation space, so cosine distance
is semantically meaningful:
  - Normal case:       MeSH description ≈ report text  → low D  → COMMIT_TRAJECTORY
  - Contradiction case: MeSH says "cardiomegaly" but report negates it
                        → description far from report text → high D → TRIGGER_REPLAN

V1 (Stream Independence) holds: both streams share BioBERT *weights* (read-only),
but encode_stream_a and encode_stream_b are separate calls with no shared state.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def mesh_tags_to_text(tags: dict) -> str:
    """
    Convert a MeSH tag dict → natural-language finding description.
    Tags are root terms like 'cardiomegaly', 'pulmonary atelectasis', etc.
    """
    ROOT_TERMS = {
        "cardiomegaly", "pulmonary atelectasis", "pleural effusion",
        "opacity", "pneumothorax", "consolidation", "pulmonary edema",
    }
    # Collect root findings present in tags
    present = sorted(t for t in tags if t in ROOT_TERMS)

    if not present:
        return "Normal chest X-ray. No significant cardiopulmonary findings identified."

    finding_str = "; ".join(present)
    return (
        f"Chest X-ray findings include the following abnormalities: {finding_str}. "
        f"These findings are present based on automated image analysis."
    )


class BioBERTStreamEncoder:
    """
    Shared BioBERT encoder used for both streams in Option B.
    Confidence = sigmoid of CLS L2 norm, calibrated to BioBERT typical range (6–20).
    """

    MODEL_ID = "dmis-lab/biobert-base-cased-v1.2"

    def __init__(self, device: str = "mps", max_length: int = 256):
        self.device = device
        self.max_length = max_length
        print(f"[BioBERT-B] Loading {self.MODEL_ID}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
        self.model = AutoModel.from_pretrained(self.MODEL_ID).to(device).eval()

    @torch.no_grad()
    def encode(self, text: str) -> tuple[torch.Tensor, float]:
        """Returns (CLS latent (1,768), confidence ∈ [0,1])."""
        inputs = self.tokenizer(
            text, return_tensors="pt",
            truncation=True, max_length=self.max_length, padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.model(**inputs)
        latent = out.last_hidden_state[:, 0, :]       # CLS token (1, 768)
        norm = latent.norm(dim=-1).mean()
        conf = float(torch.sigmoid((norm - 12.0) / 6.0).item())
        return latent, min(max(conf, 0.0), 1.0)
