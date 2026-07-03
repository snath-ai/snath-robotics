"""
Option C encoders — TorchXRayVision (image) + BiomedCLIP text (finding-vector)

Stream A: TorchXRayVision DenseNet-121 → 18-dim sigmoid finding probabilities
          Trained on NIH, CheXpert, MIMIC, OpenI, Kaggle (combined weights)
Stream B: BiomedCLIP zero-shot text → 18-dim finding probabilities
          Same pathology names as TXV → directly comparable finding vectors

L1 distance between the two 18-dim vectors is the divergence signal.
Normal (image ↔ own report): D ≈ 0.5–2.0
Synthetic contradiction (image ↔ different patient's report): D ≈ 3.0–6.0

Key property: both streams produce the SAME 18-dim [0,1] vector format.
The divergence is therefore semantically grounded: high D means the image
"claims" pathologies the text does not, or vice versa.
"""

import torch
import torch.nn.functional as F
import numpy as np
import torchxrayvision as xrv
import torchvision.transforms as T
from PIL import Image
import open_clip

# ── 18-finding shared vocabulary ──────────────────────────────────────────────

TXV_PATHOLOGIES = [
    "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax",
    "Edema", "Emphysema", "Fibrosis", "Effusion", "Pneumonia",
    "Pleural_Thickening", "Cardiomegaly", "Nodule", "Mass", "Hernia",
    "Lung Lesion", "Fracture", "Lung Opacity", "Enlarged Cardiomediastinum",
]

# Natural language versions for BiomedCLIP text templates
_FINDING_NAMES = [p.lower().replace("_", " ") for p in TXV_PATHOLOGIES]

PRESENT_TEMPLATES = [
    f"chest X-ray report describing {f}" for f in _FINDING_NAMES
]
ABSENT_TEMPLATES = [
    f"no {f} mentioned in the chest X-ray report" for f in _FINDING_NAMES
]


# ── Stream A: TorchXRayVision image encoder ───────────────────────────────────

class TXVImageEncoder:
    """
    Stream A: TXV DenseNet-121 image → 18-dim sigmoid finding probabilities.

    Output: p_k = P(pathology_k present | image) ∈ [0, 1]
    Confidence: mean decisiveness = mean(max(p_k, 1-p_k)), rescaled to [0,1].
    """

    def __init__(self, device: str = "mps"):
        self.device = device
        self.model = xrv.models.DenseNet(
            weights="densenet121-res224-all"
        ).to(device).eval()
        self.pathologies = list(self.model.pathologies)
        self._transform = T.Compose([
            T.Grayscale(num_output_channels=1),
            T.Resize((224, 224)),
            T.ToTensor(),
        ])
        print(f"[TXVImageEncoder] Loaded. {len(self.pathologies)} pathologies.")

    @torch.no_grad()
    def encode(self, image: Image.Image) -> tuple[np.ndarray, float]:
        """Returns (18-dim finding prob vector ∈ [0,1], confidence ∈ [0,1])."""
        img_t = self._transform(image).unsqueeze(0).to(self.device)
        img_t = img_t * 2048 - 1024          # [0,1] → [-1024, 1024]

        preds = self.model(img_t).squeeze(0)  # (18,) — sigmoid outputs
        preds = preds.clamp(0.0, 1.0)

        nan_mask = torch.isnan(preds)
        preds[nan_mask] = 0.5                 # neutral for unknown findings

        # Decisiveness ∈ [0.5, 1.0] per finding
        decisiveness = torch.stack([preds, 1.0 - preds]).max(dim=0).values
        conf = float(decisiveness.mean().item())
        conf = (conf - 0.5) * 2.0            # rescale [0.5,1] → [0,1]

        return preds.cpu().numpy(), min(max(conf, 0.0), 1.0)


# ── Stream B: BiomedCLIP text finding-vector encoder ─────────────────────────

class BiomedCLIPTextFindingEncoder:
    """
    Stream B: BiomedCLIP text → 18-dim finding probability vector.

    For each finding f_k:
        p_k = softmax([sim(report, PRESENT_TEMPLATE_k),
                       sim(report, ABSENT_TEMPLATE_k)])[0]

    Same 18-dim format as TXVImageEncoder → L1 distance is semantically grounded.
    Confidence: mean max-softmax across 18 findings, rescaled to [0,1].
    """

    def __init__(self, model, tokenizer, device: str = "mps"):
        self.model     = model
        self.tokenizer = tokenizer
        self.device    = device
        self.K         = len(TXV_PATHOLOGIES)

        # Pre-encode 36 templates (2 × 18)
        with torch.no_grad():
            tok_p = tokenizer(PRESENT_TEMPLATES).to(device)
            tok_a = tokenizer(ABSENT_TEMPLATES).to(device)
            emb_p = F.normalize(model.encode_text(tok_p).float(), dim=-1)  # (18, 512)
            emb_a = F.normalize(model.encode_text(tok_a).float(), dim=-1)  # (18, 512)

        # Interleave: [pres_0, abs_0, pres_1, abs_1, ...]
        self._templates = torch.zeros(2 * self.K, 512, device=device)
        for k in range(self.K):
            self._templates[2 * k]     = emb_p[k]
            self._templates[2 * k + 1] = emb_a[k]

        print(f"[BiomedCLIPTextFindingEncoder] {self.K} findings, {2*self.K} templates.")

    @torch.no_grad()
    def encode(self, text: str) -> tuple[np.ndarray, float]:
        """Returns (18-dim finding prob vector ∈ [0,1], confidence ∈ [0,1])."""
        tok = self.tokenizer([text]).to(self.device)
        emb = F.normalize(self.model.encode_text(tok).float(), dim=-1)  # (1, 512)

        logit_scale = self.model.logit_scale.exp()
        sims = logit_scale * (emb @ self._templates.T)   # (1, 36)

        probs     = np.zeros(self.K)
        max_probs = []
        for k in range(self.K):
            pair_logits = sims[0, 2 * k : 2 * k + 2]
            pair_probs  = pair_logits.softmax(0)
            probs[k]    = pair_probs[0].item()
            max_probs.append(pair_probs.max().item())

        conf = float(sum(max_probs) / len(max_probs))
        conf = (conf - 0.5) * 2.0

        return probs, min(max(conf, 0.0), 1.0)


def load_biomedclip(device: str = "mps"):
    MODEL_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    print(f"[BiomedCLIP] Loading {MODEL_ID} ...")
    model, _, preprocess_val = open_clip.create_model_and_transforms(MODEL_ID)
    tokenizer = open_clip.get_tokenizer(MODEL_ID)
    model = model.to(device).eval()
    print("[BiomedCLIP] Loaded.")
    return model, tokenizer
