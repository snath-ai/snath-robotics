# continual_learning — experiments

Primary experiment set for the third paper of the Lár series:

> **The Lár Training Loop: Routing Flags as Gradient Signals**
> (Sajeev 2026). Zenodo, [doi:10.5281/zenodo.20581128](https://doi.org/10.5281/zenodo.20581128). Version 2, June 2026. **Published.**

This is the "Cross-Domain Validation: Annotation-Free Learning in Physical
Systems" evidence, cited directly in the paper's Code and Data Availability
section as `snath-robotics/experiments/continual_learning/`. See
[`../LTL/README.md`](../LTL/README.md) for the supplementary LTL material
(SIGReg notebook, Winoground script, CB1–CB2 bridge validation).

## Status

| File | Claim | Result |
|---|---|---|
| `prove_learning.py` | Disagreement is valid signal | AUROC $0.45\to0.94$ |
| `ablation_proof.py` | Robust to noise and training size | $\sigma{=}0.25$, $N{=}25$ |
| `prove_transfer.py` | Detection transfers (Claims 3a–3b) | drop $0.018$, $\Delta\cos{=}+0.15$ |
| `prove_policy.py` | Policy memory (Claims 4a–4c) | $6.5\times$ faster |
| `coco_proof.py` | CLIP ViT-B/32 validation | AUROC $0.9997$ |
| `curriculum_proof.py` | Threshold sensitivity | 93.8% at $D\ge0.25$ |

Raw JSON results: `coco_results/`. Every result is reproducible on a consumer
laptop (Apple Silicon MPS/CPU) with no external GPU, per the paper's Code and
Data Availability section.
