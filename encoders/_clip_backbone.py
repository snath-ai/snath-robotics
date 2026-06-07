"""
Shared CLIP ViT-B/32 backbone singleton for CLIPImageEncoder / CLIPTextEncoder.

Uses open_clip instead of HuggingFace transformers — avoids the abseil mutex
deadlock that affects transformers on macOS (PyTorch 2.x + Apple Silicon).

Returns (model, preprocess, tokenizer):
  model:       open_clip CLIPModel — model.encode_image(tensor), model.encode_text(tokens)
  preprocess:  torchvision transform for PIL images → (3, 224, 224) tensor
  tokenizer:   open_clip.get_tokenizer("ViT-B-32") — tokenizer(list[str]) → LongTensor
"""
from __future__ import annotations

from typing import Optional, Tuple
import torch

_model      = None
_preprocess = None
_tokenizer  = None
_device: Optional[torch.device] = None


def get_clip(device: torch.device) -> Tuple:
    """
    Return (model, preprocess, tokenizer), loaded lazily and pinned to `device`.
    Subsequent calls return the cached instance regardless of device argument.
    """
    global _model, _preprocess, _tokenizer, _device
    if _model is None:
        import open_clip
        _model, _, _preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        _tokenizer = open_clip.get_tokenizer("ViT-B-32")
        _model.eval()
        _model.to(device)
        _device = device
    return _model, _preprocess, _tokenizer
