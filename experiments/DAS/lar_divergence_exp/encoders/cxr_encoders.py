"""
CXR Encoder wrappers for the NLM divergence experiment.

Stream A: google/vit-base-patch16-224 on chest X-ray images
Stream B: dmis-lab/biobert-base-cased-v1.2 on radiology report text

Both encoders return (latent: Tensor, confidence: float) satisfying
AbstractDivergenceRouter invariants V1-V2.
"""

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import (
    ViTModel, ViTImageProcessor,
    AutoModel, AutoTokenizer,
)


class ViTStreamEncoder:
    """
    Stream A: ViT-B/16 vision encoder.
    Confidence = max-softmax of mean-pooled last hidden state / temperature.
    """

    def __init__(self, device: str = "mps", temperature: float = 1.5):
        self.device = device
        self.temperature = temperature
        print("[ViT] Loading google/vit-base-patch16-224...")
        self.processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
        self.model = ViTModel.from_pretrained("google/vit-base-patch16-224").to(device).eval()

    @torch.no_grad()
    def encode(self, image: Image.Image) -> tuple[torch.Tensor, float]:
        """Returns (CLS latent (1,768), confidence ∈ [0,1])."""
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        out = self.model(pixel_values=pixel_values)
        latent = out.last_hidden_state[:, 0, :]           # CLS token (1, 768)
        # Confidence = sigmoid of CLS L2 norm, centred at 20 (typical ViT-B/16 range 10-40)
        norm = latent.norm(dim=-1).mean()
        conf = float(torch.sigmoid((norm - 20.0) / 10.0).item())
        return latent, min(max(conf, 0.0), 1.0)


class BioBERTStreamEncoder:
    """
    Stream B: BioBERT-base language encoder on radiology report text.
    Confidence = max-softmax of mean-pooled last hidden state / temperature.
    """

    MODEL_ID = "dmis-lab/biobert-base-cased-v1.2"

    def __init__(self, device: str = "mps", temperature: float = 1.5, max_length: int = 256):
        self.device = device
        self.temperature = temperature
        self.max_length = max_length
        print(f"[BioBERT] Loading {self.MODEL_ID}...")
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
        out    = self.model(**inputs)
        latent = out.last_hidden_state[:, 0, :]           # CLS token (1, 768)
        # Confidence = sigmoid of CLS L2 norm, centred at 12 (typical BioBERT range 6-20)
        norm = latent.norm(dim=-1).mean()
        conf = float(torch.sigmoid((norm - 12.0) / 6.0).item())
        return latent, min(max(conf, 0.0), 1.0)
