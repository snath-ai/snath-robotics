"""
Option A encoders — BiomedCLIP (microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)

Trained on 15 million biomedical image-text pairs (PMC-15M).
Image encoder and text encoder produce L2-normalised embeddings in a shared 512-dim
space specifically calibrated for biomedical image-text alignment.

Stream A: BiomedCLIP image encoder on chest X-ray PNG  → 512-dim unit vector
Stream B: BiomedCLIP text encoder on radiology report  → 512-dim unit vector

Cosine distance in this shared space IS semantically meaningful:
  Agreeing   image-text pair → D ≈ 0.2–0.5  → COMMIT_TRAJECTORY
  Contradicting pair         → D ≈ 0.7–1.2  → TRIGGER_REPLAN

Confidence: BiomedCLIP embeddings are L2-normalised (norm=1). We derive
confidence via zero-shot similarity to a "confident chest X-ray" anchor text:
  conf = sigmoid(logit_scale * cosine_sim(embedding, anchor) - shift)
This gives a calibrated [0,1] score that reflects how strongly the input
activates a recognisable medical representation.
"""

import torch
import torch.nn.functional as F
from PIL import Image
import open_clip

MODEL_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

# Anchor text used to compute Stream A image confidence (zero-shot)
ANCHOR_TEXT_A = "a chest X-ray radiograph"
# Anchor used for Stream B text confidence
ANCHOR_TEXT_B = "chest X-ray radiology report findings"


class BiomedCLIPImageEncoder:
    """
    Stream A: BiomedCLIP vision encoder on chest X-ray images.

    Confidence = sigmoid(logit_scale * cos_sim(image_emb, text_anchor_emb)).
    Measures how confidently the model recognises this as a valid chest X-ray
    representation, independent of any specific finding.
    """

    def __init__(self, model, preprocess, tokenizer, device: str = "mps"):
        self.model      = model
        self.preprocess = preprocess
        self.tokenizer  = tokenizer
        self.device     = device

        # Pre-encode the anchor text for confidence scoring
        with torch.no_grad():
            tok = self.tokenizer([ANCHOR_TEXT_A]).to(device)
            self._anchor = F.normalize(
                self.model.encode_text(tok).float(), dim=-1
            )  # (1, 512)

    @torch.no_grad()
    def encode(self, image: Image.Image) -> tuple[torch.Tensor, float]:
        """Returns (image embedding (1,512), confidence ∈ [0,1])."""
        img_t  = self.preprocess(image).unsqueeze(0).to(self.device)
        emb    = F.normalize(self.model.encode_image(img_t).float(), dim=-1)  # (1, 512)
        logit_scale = self.model.logit_scale.exp().item()
        cos_sim     = (emb * self._anchor).sum().item()
        conf        = float(torch.sigmoid(torch.tensor(logit_scale * cos_sim - 4.0)).item())
        return emb, min(max(conf, 0.0), 1.0)


class BiomedCLIPTextEncoder:
    """
    Stream B: BiomedCLIP text encoder on radiology report text.

    Confidence = sigmoid(logit_scale * cos_sim(text_emb, text_anchor_emb)).
    """

    def __init__(self, model, tokenizer, device: str = "mps"):
        self.model     = model
        self.tokenizer = tokenizer
        self.device    = device

        with torch.no_grad():
            tok = self.tokenizer([ANCHOR_TEXT_B]).to(device)
            self._anchor = F.normalize(
                self.model.encode_text(tok).float(), dim=-1
            )  # (1, 512)

    @torch.no_grad()
    def encode(self, text: str) -> tuple[torch.Tensor, float]:
        """Returns (text embedding (1,512), confidence ∈ [0,1])."""
        # BiomedCLIP tokenizer truncates to 256 tokens
        tok  = self.tokenizer([text]).to(self.device)
        emb  = F.normalize(self.model.encode_text(tok).float(), dim=-1)  # (1, 512)
        logit_scale = self.model.logit_scale.exp().item()
        cos_sim     = (emb * self._anchor).sum().item()
        conf        = float(torch.sigmoid(torch.tensor(logit_scale * cos_sim - 4.0)).item())
        return emb, min(max(conf, 0.0), 1.0)


def load_biomedclip(device: str = "mps"):
    """Load BiomedCLIP model, preprocessor, and tokenizer. Downloads ~800 MB on first run."""
    print(f"[BiomedCLIP] Loading {MODEL_ID} ...")
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(MODEL_ID)
    tokenizer = open_clip.get_tokenizer(MODEL_ID)
    model = model.to(device).eval()
    print("[BiomedCLIP] Loaded.")
    return model, preprocess_val, tokenizer
