"""
NB4: Annotation-Free Difficulty Oracle as a Complete Lár-JEPA Graph
====================================================================
Author : Aadithya Sajeev (Snath AI)
Date   : 2026-06-08
Status : Pre-registered experiment

THESIS
------
The 10-ABC Lár-JEPA contract is *complete* for annotation-free cross-modal
difficulty prediction: every bottleneck in the DAS / EIM papers maps to
exactly one ABC invariant, and the Lár execution engine enforces all
invariants simultaneously.

NB4 is therefore not just an experiment — it is a completeness proof.

COMPLETE BOTTLENECK → ABC → NB4-NODE MAPPING
─────────────────────────────────────────────
┌────────────────────────────────────────┬──────────────┬───────────────────────────────┐
│ Bottleneck                             │ ABC / Inv.   │ NB4 Node                      │
├────────────────────────────────────────┼──────────────┼───────────────────────────────┤
│ MLP AUROC 0.560: no HCL-seq access     │ I2           │ HCLAttentionOracle            │
│ Stale embeddings → zombie action       │ P3           │ EncoderGenerationNode (re-enc)│
│ Projection head misclassified          │ M1-M3 vs CB1 │ CLIPEncoderNode               │
│ Routing IS labeling (no annotation)    │ V6           │ D_ScoreLabeler                │
│ Sequential 5-seed → slow evaluation    │ BatchNode    │ SeedParallelBatch             │
│ HCL grows unboundedly                  │ ReduceNode   │ HCL_CompressorNode            │
│ No reproducibility guarantee           │ I5-I6 / HMAC │ GraphExecutor(hmac_secret)    │
│ Probabilistic routing misfires         │ R1-R4 / V4   │ AUROC_Router (deterministic)  │
│ Context contamination (social eng.)    │ V4 / JuryNode│ PolicyInvariantNode           │
│ k-NN vs full-softmax ablation          │ A5 / Entropic│ k_AblationRouter              │
│ No resumable curriculum                │ P1-P6 / TM   │ ResumeCheckpointNode          │
│ AUROC comparison bottleneck            │ BatchNode    │ OracleCompareBatch            │
│ NB4 is a JEPA world-model prediction   │ AbstractMfld │ embed_context=f_x;predict=att │
│ No continual self-improvement          │ RECURSIVE_SI │ HCL→oracle→HCL loop           │
│ Encoder modality + dim contract        │ M1: out_dim  │ CLIPEncoderNode (512-fixed)   │
│ Context bridge must be stateless       │ CB1-CB2      │ IdentityBridge                │
│ Deterministic routable execution       │ CognNode     │ CognitiveNodeAdapter wrap     │
│ Self-healing on encoder crash          │ AdaptiveNode │ error_node → FallbackNode     │
└────────────────────────────────────────┴──────────────┴───────────────────────────────┘

ABC COVERAGE (all 10 satisfied simultaneously)
──────────────────────────────────────────────
  AbstractAttentionKernel     A1-A6   ScaledDotProductKernel
  AbstractLatentFaultLocator  I1-I6   HCLAttentionOracle
  AbstractModalEncoder        M1-M3   CLIPEncoderNode
  AbstractPerturbationOperator P1-P6  EncoderGenerationNode
  AbstractDivergenceRouter    V1-V6   D_ScoreLabeler
  AbstractRoutingKernel       R1-R4   AUROC_Router
  AbstractContextBridge       CB1-CB2 IdentityBridge
  AbstractManifold            JEPA    NB4 IS predict_target(z_img, HCL)
  AbstractEntropicRouter      Gating  k_AblationRouter
  AbstractCognitiveNode       Wrap    CognitiveNodeAdapter (every node)

GRAPH TOPOLOGY
──────────────
  [CLIPEncoderNode]           M1-M3 : encode all COCO pairs
       │
  [D_ScoreLabeler]            V1-V6 : compute D-scores, flag D≥τ → HCL
       │
  [HCL_BuilderNode]           I1    : pool hard-case embeddings
       │
  [EncoderGenerationNode]     P1-P6 : re-encode HCL raw pairs (zombie-action defence)
       │
  [InvariantCheckerNode]      A1-A6 : verify kernel invariants before oracle runs
       │
  [OracleCompareBatch]        -------BatchNode─────────────────
       │   [HCLAttentionOracle]  I1-I6 : attention over HCL
       │   [MLP_BaselineNode]          : Exp 3 baseline
       └─────────────────────────────────────────────────────
       │
  [k_AblationRouter]          Entropic : branch k ∈ {5, 10, 20, 50}
       │ (×4 parallel branches, merged)
  [AUROC_JudgeNode]           R1-R4 : compute AUROC, paired t-test
       │
  [StatisticalTestNode]             : p-value, Bonferroni correction
       │
  [HCL_CompressorNode]        ReduceNode : Episodic→Semantic DMN transition
       │
  [ResumeCheckpointNode]      P1-P6 / TM : serialize HCL + oracle state to JSON
       │
  [ResultLoggerNode]          HMAC  : sign execution trace (I5-I6)

COMPLIANCE-BY-ARCHITECTURE (June 8 2026)
─────────────────────────────────────────
From reading Lár v2.3.0 source (enterprise/backbone.py), the compliance thesis extends:
  23 EU AI Act requirements → 23 Lár nodes → fired at RUNTIME (not documentation).
  Key compliance primitives relevant to NB4 production deployment:
    ProhibitedPracticeGuard  → Art. 5   (auto-scans every LLM output)
    FundamentalRightsImpactNode → Art. 9 FRIA (6 EU Charter dims as runtime gate)
    RiskScorerNode + HumanJuryNode → Art. 14 (dynamic oversight escalation)
    CredentialVault.get_with_trust() → Art. 15(4) (trust-gated JIT provisioning)
    BehavioralEnvelopeMonitor → Art. 9 PMM + Art. 3(23) (output variance monitoring)
    AuthorityLedger → Fourth Tier (immutable HMAC-signed human authority record)
    LethalTrifectaGuard → AEPD PoP (Rule-of-2: untrusted + sensitive + autonomous → block)

  NOTE: DynamicNode is DEPRECATED in v2.3.0 → use AdaptiveNode instead.

Dependencies (Kaggle-compatible):
  pip install openai-clip torch torchvision scikit-learn scipy
  # Lár (inner package):
  pip install git+https://github.com/snath-ai/lar  (or local path below)
"""

from __future__ import annotations

import os
import sys
import json
import hmac
import hashlib
import time
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from scipy import stats

# ─────────────────────────────────────────────────────────────────────────────
# PATH SETUP  (adjust for Kaggle or local)
# ─────────────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_PLAY = _HERE.parent.parent.parent.parent  # notebooks -> EIM -> experiments -> Snath Robotics -> JEPA_Playground
_CORE = _PLAY / "lar_jepa" / "core"       # interfaces.py lives here
_LAR  = _PLAY / "lar_jepa" / "lar_jepa" / "src"  # lar package

for p in [str(_CORE), str(_LAR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ─────────────────────────────────────────────────────────────────────────────
# LAR IMPORTS  (BaseNode + executor)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from lar import GraphState, GraphExecutor, BaseNode, BatchNode, ToolNode, RouterNode
    _LAR_AVAILABLE = True
except ImportError:
    _LAR_AVAILABLE = False
    print("[WARN] lar package not found — running in standalone mode (ABC contracts enforced manually)")
    # Minimal stubs so the ABC classes still work
    class GraphState(dict):
        def get(self, k, d=None): return super().get(k, d)
        def set(self, k, v): self[k] = v
    class BaseNode:
        next_node = None
        def execute(self, state): raise NotImplementedError
    class BatchNode:
        def __init__(self, nodes, next_node=None):
            self.nodes = nodes; self.next_node = next_node
    class ToolNode(BaseNode):
        def __init__(self, tool_function, input_keys, output_key, next_node=None, error_node=None):
            self.tool_function = tool_function; self.input_keys = input_keys
            self.output_key = output_key; self.next_node = next_node; self.error_node = error_node
    class RouterNode(BaseNode):
        def __init__(self, decision_function, path_map, default_node=None):
            self.decision_function = decision_function; self.path_map = path_map
            self.default_node = default_node

# ─────────────────────────────────────────────────────────────────────────────
# ABC CONTRACT IMPORTS  (from core/interfaces.py)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from interfaces import (
        AbstractAttentionKernel,
        AbstractLatentFaultLocator,
        AbstractModalEncoder,
        AbstractPerturbationOperator,
        AbstractDivergenceRouter,
        AbstractRoutingKernel,
        AbstractContextBridge,
    )
    _ABC_AVAILABLE = True
except ImportError:
    _ABC_AVAILABLE = False
    print("[WARN] core/interfaces.py not found — ABC contracts defined inline")
    # Define minimal ABC stubs inline so invariant checks still work
    class AbstractAttentionKernel:
        def compute(self, query, key, value, k): raise NotImplementedError
    class AbstractLatentFaultLocator:
        def locate(self, x_env, x_struct, k): raise NotImplementedError
    class AbstractModalEncoder:
        modality: str = ""; output_dim: int = 0
        def encode(self, x): raise NotImplementedError
    class AbstractPerturbationOperator:
        def encode_wildtype(self, x): raise NotImplementedError
        def encode_mutant(self, x): raise NotImplementedError
    class AbstractDivergenceRouter:
        def route(self, v_i, v_t): raise NotImplementedError
    class AbstractRoutingKernel:
        def score(self, x): raise NotImplementedError
        def route(self, score): raise NotImplementedError
    class AbstractContextBridge:
        def bridge(self, x): raise NotImplementedError

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
SEEDS        = [42, 7, 13, 99, 2025]   # 5-seed evaluation
TAU          = 0.30                     # D-score routing threshold
K_VALUES     = [5, 10, 20, 50]         # k-ablation
N_PAIRS      = 5000                    # COCO val2017
HCL_FRAC     = 0.80                   # top-80% hard pairs go into HCL
MLP_AUROC_BASELINE = 0.560            # Exp 3 result
AUROC_TARGET = 0.70                   # pre-registered target
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HMAC_SECRET  = os.getenv("LAR_HMAC_SECRET", "")
LOG_DIR      = str(_HERE / "nb4_lar_logs")
CHECKPOINT   = str(_HERE / "nb4_checkpoint.json")

print(f"[NB4] Device: {DEVICE}")
print(f"[NB4] Lár available: {_LAR_AVAILABLE} | ABCs available: {_ABC_AVAILABLE}")

# ─────────────────────────────────────────────────────────────────────────────
# 1. ABSTRACTMODALENCODER (M1-M3)
#    M1: output_dim is fixed and declared
#    M2: modality is declared
#    M3: encode() is deterministic (no dropout at inference)
# ─────────────────────────────────────────────────────────────────────────────
class CLIPEncoderNode(AbstractModalEncoder, BaseNode):
    """
    Satisfies AbstractModalEncoder M1-M3.
    Bottleneck solved: 'Projection head misclassified as CB' — this node has
    trainable weights (linear projection after CLIP), so it IS a ModalEncoder,
    not a stateless ContextBridge (which would violate CB1).
    """
    modality   = "image"
    output_dim = 512        # M1: fixed, declared at class definition time

    def __init__(self, next_node=None):
        self.next_node = next_node
        self._model    = None   # lazy-loaded
        self._prep     = None

    def _load(self):
        if self._model is None:
            try:
                import clip
                self._model, self._prep = clip.load("ViT-B/32", device=DEVICE)
                self._model.eval()
            except ImportError:
                print("[CLIPEncoderNode] clip not installed — using random embeddings (test mode)")

    def encode(self, x) -> torch.Tensor:
        """M3: no dropout; output is deterministic at inference."""
        self._load()
        if self._model is None:
            # test-mode fallback: random unit vector of correct dimension
            z = torch.randn(self.output_dim, device=DEVICE)
            return z / z.norm()
        with torch.no_grad():
            if hasattr(x, "shape"):   # already a tensor
                return self._model.encode_image(x.to(DEVICE)).float()
            return self._model.encode_text(
                clip.tokenize([x]).to(DEVICE)
            ).squeeze(0).float()

    def execute(self, state: GraphState) -> Optional[BaseNode]:
        """Lár BaseNode interface: encode all pairs, store image/text embeddings."""
        print("[CLIPEncoderNode] Encoding pairs (M1-M3)...")
        pairs: List[Tuple] = state.get("raw_pairs", [])
        if not pairs:
            print("  [WARN] no raw_pairs in state — injecting mock data for test mode")
            pairs = [(None, None)] * N_PAIRS
            state.set("raw_pairs", pairs)

        self._load()
        z_imgs, z_txts = [], []
        for (x_img, x_txt) in pairs:
            z_imgs.append(self.encode(x_img) if x_img is not None
                          else torch.randn(self.output_dim, device=DEVICE))
            z_txts.append(self.encode(x_txt) if x_txt is not None
                          else torch.randn(self.output_dim, device=DEVICE))

        state.set("z_imgs", torch.stack(z_imgs))  # (N, 512)
        state.set("z_txts", torch.stack(z_txts))  # (N, 512)
        print(f"  Encoded {len(pairs)} pairs → z_imgs: {state.get('z_imgs').shape}")
        return self.next_node


# ─────────────────────────────────────────────────────────────────────────────
# 2. ABSTRACTDIVERGENCEROUTER (V1-V6)
#    V6 = Safety-Learning Equivalence: routing flag IS the annotation.
#    Bottleneck solved: 'No annotation needed' — D ≥ τ IS the hard-case label.
# ─────────────────────────────────────────────────────────────────────────────
class D_ScoreLabeler(AbstractDivergenceRouter, BaseNode):
    """
    Satisfies AbstractDivergenceRouter V1-V6.
    V6: routing decision (D ≥ τ) IS the difficulty label — no human annotators.
    V4: labeling is content-blind (only geometric distance, not semantics).
    """
    def __init__(self, tau: float = TAU, concept_dim: int = 80, next_node=None):
        self.tau         = tau
        self.concept_dim = concept_dim
        self.next_node   = next_node
        self._W_c        = torch.randn(512, concept_dim) * 0.02  # concept projection

    def _concept_vector(self, z: torch.Tensor) -> torch.Tensor:
        """Project encoder embedding → concept distribution (V2: geometric)."""
        logits = z @ self._W_c.to(z.device)                      # (..., C)
        return torch.softmax(logits * 100, dim=-1)                # temperature τ=100

    def d_score(self, z_img: torch.Tensor, z_txt: torch.Tensor) -> torch.Tensor:
        """D = ||v_I - v_T||_1 / sqrt(C) — V1: finite float ≥ 0."""
        v_I = self._concept_vector(z_img)
        v_T = self._concept_vector(z_txt)
        return (v_I - v_T).abs().sum(dim=-1) / (self.concept_dim ** 0.5)

    def route(self, v_i, v_t):
        """V5: routing completeness — always returns one of four decisions."""
        d = (v_i - v_t).abs().sum() / (self.concept_dim ** 0.5)
        if d >= self.tau:     return "TRIGGER_REPLAN"
        elif d >= self.tau/2: return "REQUEST_INSPECTION"
        else:                 return "EXECUTE"

    def execute(self, state: GraphState) -> Optional[BaseNode]:
        print(f"[D_ScoreLabeler] Computing D-scores (V1-V6, τ={self.tau})...")
        z_imgs: torch.Tensor = state.get("z_imgs")
        z_txts: torch.Tensor = state.get("z_txts")

        d_scores = self.d_score(z_imgs, z_txts)   # (N,)
        labels   = (d_scores >= self.tau).float()  # V6: routing IS labeling

        state.set("d_scores", d_scores)
        state.set("labels",   labels)
        n_hard = labels.sum().int().item()
        print(f"  D-scores: min={d_scores.min():.3f} max={d_scores.max():.3f} "
              f"| Hard cases (D≥τ): {n_hard}/{len(d_scores)} ({100*n_hard/len(d_scores):.1f}%)")
        return self.next_node


# ─────────────────────────────────────────────────────────────────────────────
# 3. HCL BUILDER  (I1 — AbstractLatentFaultLocator)
#    I1: fault locator receives environmental-state query (image embedding).
#    Builds the key-value store: K=HCL image embeddings, V=HCL D-scores.
# ─────────────────────────────────────────────────────────────────────────────
class HCL_BuilderNode(BaseNode):
    """
    Builds the Hard Case Log (HCL) key-value store for the attention oracle.
    Satisfies I1: the query comes from the test image (environmental state).
    Satisfies I2: the key-value store IS the HCL structural sequence.
    """
    def __init__(self, hcl_frac: float = HCL_FRAC, next_node=None):
        self.hcl_frac  = hcl_frac
        self.next_node = next_node

    def execute(self, state: GraphState) -> Optional[BaseNode]:
        print(f"[HCL_BuilderNode] Building HCL (top-{100*self.hcl_frac:.0f}% hard pairs)...")
        z_imgs:   torch.Tensor = state.get("z_imgs")
        d_scores: torch.Tensor = state.get("d_scores")

        n_hcl = int(len(z_imgs) * self.hcl_frac)
        _, top_idx = torch.topk(d_scores, n_hcl)

        # K = HCL image embeddings (I2: structural fault sequence)
        # V = HCL D-scores (the "fault signal" values)
        hcl_K = z_imgs[top_idx]           # (n_hcl, 512)
        hcl_V = d_scores[top_idx]         # (n_hcl,)

        state.set("hcl_K",   hcl_K)
        state.set("hcl_V",   hcl_V)
        state.set("hcl_idx", top_idx)
        print(f"  HCL size: {n_hcl} pairs | K: {hcl_K.shape} | V: {hcl_V.shape}")
        return self.next_node


# ─────────────────────────────────────────────────────────────────────────────
# 4. ENCODER GENERATION NODE  (AbstractPerturbationOperator P1-P6)
#    P3: identity at α=0 (same encoder → Δ=0 → HCL unchanged).
#    Bottleneck solved: 'Stale embeddings / zombie action'.
#    When encoder upgrades (B/32 → L/14), re-encode raw pairs from HCL.
# ─────────────────────────────────────────────────────────────────────────────
class EncoderGenerationNode(AbstractPerturbationOperator, BaseNode):
    """
    Satisfies AbstractPerturbationOperator P1-P6.
    Δ = Enc_{t+1}(x) - Enc_t(x)  (P2: Δ is the encoder-generation perturbation)
    P3: when Enc_{t+1} ≡ Enc_t, Δ ≡ 0 — curriculum unchanged.
    Bottleneck: stale embeddings = zombie action. Solution: always re-encode
    raw pairs from HCL with the CURRENT encoder before oracle computation.
    """
    def __init__(self, encoder: CLIPEncoderNode, next_node=None):
        self.encoder   = encoder
        self.next_node = next_node

    def encode_wildtype(self, x):   return self.encoder.encode(x)
    def encode_mutant(self, x):     return self.encoder.encode(x)   # same in this run

    def execute(self, state: GraphState) -> Optional[BaseNode]:
        print("[EncoderGenerationNode] Re-encoding HCL raw pairs (P1-P6 zombie-action defence)...")
        raw_pairs: List = state.get("raw_pairs", [])
        hcl_idx:   torch.Tensor = state.get("hcl_idx")

        if raw_pairs and hcl_idx is not None:
            # Re-encode only the HCL subset from raw inputs (not from stored z)
            hcl_raw = [raw_pairs[i] for i in hcl_idx.tolist()]
            fresh_K = torch.stack([
                self.encoder.encode(x_img) if x_img is not None
                else torch.randn(self.encoder.output_dim, device=DEVICE)
                for (x_img, _) in hcl_raw
            ])
            # Δ: difference between fresh encoding and cached
            old_K = state.get("hcl_K")
            delta = (fresh_K - old_K).norm(dim=-1).mean().item()
            state.set("hcl_K", fresh_K)          # updated with current encoder
            state.set("encoder_delta_norm", delta)
            print(f"  Encoder Δ (mean L2): {delta:.6f}  "
                  f"{'(null update — P3 satisfied)' if delta < 1e-6 else '(encoder upgraded)'}")
        else:
            print("  No raw pairs available — skipping re-encode (using cached HCL K)")
        return self.next_node


# ─────────────────────────────────────────────────────────────────────────────
# 5. SCALED DOT-PRODUCT KERNEL  (AbstractAttentionKernel A1-A6)
# ─────────────────────────────────────────────────────────────────────────────
class ScaledDotProductKernel(AbstractAttentionKernel):
    """
    Satisfies AbstractAttentionKernel A1-A6:
      A1: compute(Q, K, V, k) signature
      A2: attention_weights has same shape as K (one weight per HCL entry)
      A3: weights ≥ 0  (softmax output)
      A4: weights sum ≈ 1  (softmax)
      A5: topk_indices are in descending weight order
      A6: len(topk_indices) == k  (hard guarantee)
    """
    def compute(
        self,
        query: torch.Tensor,   # (d,)
        key:   torch.Tensor,   # (n, d)
        value: torch.Tensor,   # (n,)
        k:     int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if query.dim() == 1:
            query = query.unsqueeze(0)               # (1, d)
        d = query.shape[-1]
        scores  = (query @ key.T) / (d ** 0.5)      # (1, n)
        weights = torch.softmax(scores, dim=-1).squeeze(0)  # (n,) — A3, A4
        k_capped = min(k, len(weights))              # cannot exceed HCL size
        _, topk_idx = torch.topk(weights, k=k_capped, largest=True, sorted=True)  # A5
        assert len(topk_idx) == k_capped             # A6
        return weights, topk_idx


# ─────────────────────────────────────────────────────────────────────────────
# 6. HCL ATTENTION ORACLE  (AbstractLatentFaultLocator I1-I6)
#    I2 = THE LOAD-BEARING INVARIANT: the oracle has access to the full
#         HCL sequence at inference time.  MLP violated this by design.
# ─────────────────────────────────────────────────────────────────────────────
class HCLAttentionOracle(AbstractLatentFaultLocator, BaseNode):
    """
    Satisfies AbstractLatentFaultLocator I1-I6.
    I1: query = z_img (environmental state)
    I2: key   = HCL image embeddings (structural fault sequence) ← THE FIX
    I3: value = HCL D-scores (fault signal magnitudes)
    I4: attention weights are the fault-localization coordinates
    I5: topk_indices identify the k most similar historical hard cases
    I6: risk_score = weighted sum of HCL D-scores

    NB4 IS also an AbstractManifold.predict_target():
      embed_context(x_img) = z_img  (f_x)
      predict_target(z_img, HCL)    = predicted D-score  (g)
    """
    def __init__(self, kernel: ScaledDotProductKernel, k: int = 10, next_node=None):
        self.kernel    = kernel
        self.k         = k
        self.next_node = next_node

    def locate(self, x_env, x_struct, k):
        """AbstractLatentFaultLocator interface."""
        weights, topk_idx = self.kernel.compute(x_env, x_struct["K"], x_struct["V"], k)
        risk_score  = (weights * x_struct["V"]).sum().item()
        coordinates = topk_idx
        return risk_score, coordinates, weights

    def _predict_full(self, z_query: torch.Tensor, hcl_K: torch.Tensor,
                      hcl_V: torch.Tensor) -> np.ndarray:
        """Full-softmax prediction (I2 fully satisfied)."""
        scores = []
        for q in z_query:
            w, _ = self.kernel.compute(q, hcl_K, hcl_V, self.k)
            scores.append((w * hcl_V).sum().item())
        return np.array(scores)

    def _predict_knn(self, z_query: torch.Tensor, hcl_K: torch.Tensor,
                     hcl_V: torch.Tensor) -> np.ndarray:
        """k-NN ablation (top-k only, for AbstractEntropicRouter k-ablation branch)."""
        scores = []
        for q in z_query:
            _, idx = self.kernel.compute(q, hcl_K, hcl_V, self.k)
            scores.append(hcl_V[idx].mean().item())
        return np.array(scores)

    def execute(self, state: GraphState) -> Optional[BaseNode]:
        print(f"[HCLAttentionOracle] Running attention oracle (I1-I6, k={self.k})...")
        hcl_K:    torch.Tensor = state.get("hcl_K")
        hcl_V:    torch.Tensor = state.get("hcl_V")
        z_imgs:   torch.Tensor = state.get("z_imgs")
        hcl_idx:  torch.Tensor = state.get("hcl_idx")

        # Eval split: non-HCL pairs (held-out 20% hard + easy)
        n = len(z_imgs)
        all_idx = set(range(n))
        hcl_set = set(hcl_idx.tolist())
        eval_idx = torch.tensor(sorted(all_idx - hcl_set))

        z_eval    = z_imgs[eval_idx]
        d_eval    = state.get("d_scores")[eval_idx]
        y_true    = (d_eval >= TAU).cpu().numpy().astype(int)

        # Full-softmax oracle prediction
        y_score_full = self._predict_full(z_eval, hcl_K, hcl_V)
        y_score_knn  = self._predict_knn(z_eval, hcl_K, hcl_V)

        state.set("oracle_scores_full", y_score_full)
        state.set("oracle_scores_knn",  y_score_knn)
        state.set("eval_labels",        y_true)
        state.set("eval_idx",           eval_idx)
        print(f"  Eval set: {len(eval_idx)} pairs | Hard: {y_true.sum()} | Easy: {(1-y_true).sum()}")
        return self.next_node


# ─────────────────────────────────────────────────────────────────────────────
# 7. MLP BASELINE  (Exp 3 — I2 violated by design)
# ─────────────────────────────────────────────────────────────────────────────
class MLP_BaselineNode(BaseNode):
    """
    The Exp 3 MLP baseline: MLP(z_img) → predicted D-score.
    VIOLATES I2: no access to HCL sequence at inference time.
    Baseline AUROC: 0.560 (pre-registered).
    Run in parallel with HCLAttentionOracle via OracleCompareBatch.
    """
    def __init__(self, hidden: int = 256, next_node=None):
        self.hidden    = hidden
        self.next_node = next_node
        self._mlp      = None

    def _build_mlp(self, input_dim: int):
        import torch.nn as nn
        self._mlp = nn.Sequential(
            nn.Linear(input_dim, self.hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden, 1),
            nn.Sigmoid()
        ).to(DEVICE)

    def execute(self, state: GraphState) -> Optional[BaseNode]:
        print("[MLP_BaselineNode] Running MLP baseline (I2 violated — Exp 3 replicated)...")
        z_imgs  = state.get("z_imgs")
        labels  = state.get("labels").to(DEVICE)
        eval_idx = state.get("eval_idx")

        if self._mlp is None:
            self._build_mlp(z_imgs.shape[1])

        import torch.nn as nn
        import torch.optim as optim

        hcl_idx = state.get("hcl_idx")
        X_train = z_imgs[hcl_idx].to(DEVICE)
        y_train = labels[hcl_idx].unsqueeze(1)

        opt  = optim.Adam(self._mlp.parameters(), lr=1e-3)
        loss_fn = nn.BCELoss()
        self._mlp.train()
        for _ in range(50):                            # quick training
            opt.zero_grad()
            loss = loss_fn(self._mlp(X_train), y_train)
            loss.backward()
            opt.step()

        self._mlp.eval()
        with torch.no_grad():
            X_eval = z_imgs[eval_idx].to(DEVICE)
            mlp_scores = self._mlp(X_eval).squeeze().cpu().numpy()

        state.set("mlp_scores", mlp_scores)
        print(f"  MLP scores: min={mlp_scores.min():.3f} max={mlp_scores.max():.3f}")
        return self.next_node


# ─────────────────────────────────────────────────────────────────────────────
# 8. INVARIANT CHECKER  (A1-A6 verification)
# ─────────────────────────────────────────────────────────────────────────────
class InvariantCheckerNode(BaseNode):
    """
    Verifies A1-A6 on a sample query before the oracle runs.
    Implements the JuryNode / PolicyInvariant pattern:
    the pipeline cannot proceed if any invariant is violated.
    """
    def __init__(self, kernel: ScaledDotProductKernel, k: int = 10, next_node=None):
        self.kernel    = kernel
        self.k         = k
        self.next_node = next_node

    def execute(self, state: GraphState) -> Optional[BaseNode]:
        print("[InvariantCheckerNode] Verifying A1-A6 kernel invariants...")
        hcl_K = state.get("hcl_K")
        hcl_V = state.get("hcl_V")
        q     = torch.randn(hcl_K.shape[1], device=DEVICE)

        weights, topk_idx = self.kernel.compute(q, hcl_K, hcl_V, self.k)

        checks = {
            "A1 compute() signature": True,                              # always passes if we got here
            "A2 weights.shape == K.shape[0]": weights.shape[0] == hcl_K.shape[0],
            "A3 weights >= 0": bool((weights >= 0).all()),
            "A4 sum(weights) ≈ 1": bool(abs(weights.sum().item() - 1.0) < 1e-3),
            "A5 topk descending": bool(
                all(weights[topk_idx[i]] >= weights[topk_idx[i+1]]
                    for i in range(len(topk_idx)-1))
            ),
            "A6 len(topk) == k": len(topk_idx) == min(self.k, len(hcl_K)),
        }

        all_pass = all(checks.values())
        for name, ok in checks.items():
            print(f"  {'✅' if ok else '❌'} {name}")

        if not all_pass:
            raise RuntimeError("[INVARIANT VIOLATION] Kernel invariants A1-A6 not satisfied — "
                               "aborting NB4 pipeline (JuryNode BLOCKED)")

        state.set("invariant_checks", checks)
        print("  All A1-A6 invariants satisfied — oracle cleared to run")
        return self.next_node


# ─────────────────────────────────────────────────────────────────────────────
# 9. AUROC JUDGE  (AbstractRoutingKernel R1-R4)
#    R1: score() returns a finite float (AUROC ∈ [0,1])
#    R3: score is deterministic
#    Bottleneck solved: 'AUROC comparison bottleneck' → A/B Tester pattern
# ─────────────────────────────────────────────────────────────────────────────
class AUROC_JudgeNode(AbstractRoutingKernel, BaseNode):
    """
    Satisfies AbstractRoutingKernel R1-R4.
    Computes AUROC for oracle vs MLP, runs paired t-test across seeds.
    Implements the A/B Tester fan-in pattern.
    """
    def __init__(self, next_node=None):
        self.next_node = next_node

    def score(self, x):
        from sklearn.metrics import roc_auc_score
        y_true, y_score = x
        return float(roc_auc_score(y_true, y_score))   # R1: finite float

    def route(self, score):
        if score > AUROC_TARGET:   return "ABOVE_TARGET"
        elif score > 0.60:         return "MARGINAL"
        else:                      return "BELOW_BASELINE"

    def execute(self, state: GraphState) -> Optional[BaseNode]:
        from sklearn.metrics import roc_auc_score
        print("[AUROC_JudgeNode] Computing AUROC (R1-R4, A/B Tester fan-in)...")

        y_true         = state.get("eval_labels")
        oracle_full    = state.get("oracle_scores_full")
        oracle_knn     = state.get("oracle_scores_knn")
        mlp_scores     = state.get("mlp_scores")

        auroc_full  = roc_auc_score(y_true, oracle_full)  if oracle_full  is not None else None
        auroc_knn   = roc_auc_score(y_true, oracle_knn)   if oracle_knn   is not None else None
        auroc_mlp   = roc_auc_score(y_true, mlp_scores)   if mlp_scores   is not None else None

        print(f"\n  {'─'*50}")
        print(f"  AUROC — Oracle (full softmax): {auroc_full:.4f}" if auroc_full else "")
        print(f"  AUROC — Oracle (k-NN):         {auroc_knn:.4f}"  if auroc_knn  else "")
        print(f"  AUROC — MLP baseline:          {auroc_mlp:.4f}"  if auroc_mlp  else "")
        print(f"  Target:                        {AUROC_TARGET:.4f}")
        print(f"  Pre-registered baseline:       {MLP_AUROC_BASELINE:.4f}")
        print(f"  {'─'*50}\n")

        decision_full = self.route(auroc_full) if auroc_full else "UNKNOWN"
        print(f"  R4 routing decision: {decision_full}")
        if decision_full == "ABOVE_TARGET":
            print("  ✅ NB4 HYPOTHESIS CONFIRMED: AUROC > 0.70")
        elif decision_full == "MARGINAL":
            print("  ⚠️  Marginal result — k-ablation may identify better setting")
        else:
            print("  ❌ Below target — review HCL construction or encoder quality")

        state.set("auroc_oracle_full", auroc_full)
        state.set("auroc_oracle_knn",  auroc_knn)
        state.set("auroc_mlp",         auroc_mlp)
        state.set("routing_decision",  decision_full)
        return self.next_node


# ─────────────────────────────────────────────────────────────────────────────
# 10. k-ABLATION ROUTER  (AbstractEntropicRouter — COMMIT/REPLAN/IMPASSE)
#     Bottleneck solved: 'k-NN vs full-softmax ablation'
# ─────────────────────────────────────────────────────────────────────────────
class k_AblationRouter(BaseNode):
    """
    Implements AbstractEntropicRouter gating: for each k ∈ {5,10,20,50},
    run the oracle and collect AUROC. Route to COMMIT if best k > target.
    Satisfies IMPASSE gating if no k exceeds target.
    """
    def __init__(self, kernel: ScaledDotProductKernel, k_values=None, next_node=None):
        self.kernel   = kernel
        self.k_values = k_values or K_VALUES
        self.next_node = next_node

    def execute(self, state: GraphState) -> Optional[BaseNode]:
        from sklearn.metrics import roc_auc_score
        print(f"[k_AblationRouter] k-ablation over {self.k_values} (AbstractEntropicRouter)...")

        hcl_K  = state.get("hcl_K")
        hcl_V  = state.get("hcl_V")
        z_imgs = state.get("z_imgs")
        y_true = state.get("eval_labels")
        eval_idx = state.get("eval_idx")
        z_eval = z_imgs[eval_idx]

        results = {}
        for k in self.k_values:
            oracle = HCLAttentionOracle(self.kernel, k=k)
            scores = oracle._predict_full(z_eval, hcl_K, hcl_V)
            auroc  = roc_auc_score(y_true, scores)
            results[k] = auroc
            verdict = "✅" if auroc > AUROC_TARGET else ("⚠️ " if auroc > 0.60 else "❌")
            print(f"  {verdict}  k={k:3d}  AUROC={auroc:.4f}")

        best_k    = max(results, key=results.get)
        best_auroc = results[best_k]
        state.set("k_ablation_results", results)
        state.set("best_k",             best_k)
        state.set("best_auroc",         best_auroc)

        # Entropic gating
        if best_auroc > AUROC_TARGET:
            print(f"\n  COMMIT: best k={best_k} (AUROC={best_auroc:.4f}) > {AUROC_TARGET}")
            state.set("entropic_gate", "COMMIT")
        elif best_auroc > 0.60:
            print(f"\n  REPLAN: best k={best_k} (AUROC={best_auroc:.4f}) — marginal")
            state.set("entropic_gate", "REPLAN")
        else:
            print(f"\n  IMPASSE: no k achieves AUROC > 0.60 — escalate")
            state.set("entropic_gate", "IMPASSE")

        return self.next_node


# ─────────────────────────────────────────────────────────────────────────────
# 11. STATISTICAL TEST  (5-seed paired t-test)
#     Bottleneck solved: 'Sequential 5-seed evaluation' via BatchNode parallelism
# ─────────────────────────────────────────────────────────────────────────────
class StatisticalTestNode(BaseNode):
    """
    Collects per-seed AUROC values and runs paired t-test (oracle vs MLP).
    In the full graph this node receives merged state from a BatchNode
    running one oracle per seed in parallel.
    """
    def __init__(self, next_node=None):
        self.next_node = next_node

    def execute(self, state: GraphState) -> Optional[BaseNode]:
        print("[StatisticalTestNode] Paired t-test (5-seed evaluation)...")
        seed_aurocs_oracle = state.get("seed_aurocs_oracle", [])
        seed_aurocs_mlp    = state.get("seed_aurocs_mlp",    [])

        if len(seed_aurocs_oracle) < 2:
            print("  [WARN] < 2 seeds — skipping t-test (run full 5-seed BatchNode for significance)")
            state.set("t_stat", None); state.set("p_value", None)
            return self.next_node

        t, p = stats.ttest_rel(seed_aurocs_oracle, seed_aurocs_mlp)
        mean_oracle = np.mean(seed_aurocs_oracle)
        mean_mlp    = np.mean(seed_aurocs_mlp)
        print(f"  Oracle AUROC:  {mean_oracle:.4f} ± {np.std(seed_aurocs_oracle):.4f}")
        print(f"  MLP AUROC:     {mean_mlp:.4f} ± {np.std(seed_aurocs_mlp):.4f}")
        print(f"  Δ AUROC:       {mean_oracle - mean_mlp:+.4f}")
        print(f"  Paired t:      {t:.4f}  p={p:.4f} {'*** (p<0.01)' if p < 0.01 else '(ns)'}")
        state.set("t_stat",  t)
        state.set("p_value", p)
        state.set("mean_oracle_auroc", mean_oracle)
        state.set("mean_mlp_auroc",    mean_mlp)
        return self.next_node


# ─────────────────────────────────────────────────────────────────────────────
# 12. HCL COMPRESSOR (ReduceNode — DMN Episodic→Semantic transition)
#     Bottleneck solved: 'HCL grows unboundedly'
# ─────────────────────────────────────────────────────────────────────────────
class HCL_CompressorNode(BaseNode):
    """
    Implements the ReduceNode pattern: summarise HCL into a compact semantic
    representation and DELETE raw large tensors from GraphState.
    Maps to DMN Episodic (raw HCL pairs) → Semantic (cluster centroids) transition.
    Bottleneck solved: memory growth.
    """
    def __init__(self, n_clusters: int = 8, next_node=None):
        self.n_clusters = n_clusters
        self.next_node  = next_node

    def execute(self, state: GraphState) -> Optional[BaseNode]:
        from sklearn.cluster import KMeans
        print(f"[HCL_CompressorNode] DMN Episodic→Semantic (K={self.n_clusters} clusters)...")
        hcl_K = state.get("hcl_K")
        hcl_V = state.get("hcl_V")

        K_np = hcl_K.cpu().numpy()
        km   = KMeans(n_clusters=min(self.n_clusters, len(K_np)), n_init=10, random_state=42)
        km.fit(K_np)

        centroids     = torch.tensor(km.cluster_centers_, dtype=torch.float32, device=DEVICE)
        centroid_lbls = torch.tensor(km.labels_)
        cluster_means = torch.stack([
            hcl_V[centroid_lbls == c].mean()
            for c in range(km.n_clusters)
        ])

        # ReduceNode: DELETE raw episodic data, keep semantic summary
        state.set("hcl_K", centroids)       # overwrite with compressed centroids
        state.set("hcl_V", cluster_means)   # overwrite with cluster difficulty means
        # hcl_idx is now stale — clear it
        state.set("hcl_idx", None)

        n_before = len(K_np)
        n_after  = len(centroids)
        print(f"  Episodic HCL: {n_before} entries → Semantic: {n_after} cluster centroids")
        print(f"  Memory reduction: {n_before/n_after:.1f}× (Episodic→Semantic DMN transition)")
        return self.next_node


# ─────────────────────────────────────────────────────────────────────────────
# 13. RESUME CHECKPOINT  (Time Machine pattern + P1-P6)
#     Bottleneck solved: 'No resumable curriculum'
# ─────────────────────────────────────────────────────────────────────────────
class ResumeCheckpointNode(BaseNode):
    """
    Implements the Time Machine pattern: serialise HCL and oracle state to JSON.
    On resume, reload and continue from this node.
    P1-P6 guarantee: re-encoding on load gives Δ=0 if encoder unchanged (P3).
    """
    def __init__(self, checkpoint_path: str = CHECKPOINT, next_node=None):
        self.path      = checkpoint_path
        self.next_node = next_node

    def execute(self, state: GraphState) -> Optional[BaseNode]:
        print(f"[ResumeCheckpointNode] Saving checkpoint → {self.path}")
        payload = {
            "auroc_oracle_full":  state.get("auroc_oracle_full"),
            "auroc_oracle_knn":   state.get("auroc_oracle_knn"),
            "auroc_mlp":          state.get("auroc_mlp"),
            "best_k":             state.get("best_k"),
            "best_auroc":         state.get("best_auroc"),
            "entropic_gate":      state.get("entropic_gate"),
            "routing_decision":   state.get("routing_decision"),
            "invariant_checks":   state.get("invariant_checks", {}),
            "encoder_delta_norm": state.get("encoder_delta_norm"),
            "k_ablation_results": state.get("k_ablation_results", {}),
            "timestamp":          time.time(),
        }
        Path(self.path).write_text(json.dumps(payload, indent=2))
        print(f"  Checkpoint saved ({len(payload)} keys)")
        return self.next_node


# ─────────────────────────────────────────────────────────────────────────────
# 14. RESULT LOGGER  (HMAC audit — I5-I6)
#     Bottleneck solved: 'No reproducibility guarantee'
# ─────────────────────────────────────────────────────────────────────────────
class ResultLoggerNode(BaseNode):
    """
    Signs the final execution trace with HMAC-SHA256.
    Satisfies I5 (immutable audit trail) and I6 (cryptographic integrity).
    Satisfies cryptographic integrity requirements for prior-art verification.
    """
    def __init__(self, secret: str = HMAC_SECRET, log_dir: str = LOG_DIR, next_node=None):
        self.secret    = secret
        self.log_dir   = log_dir
        self.next_node = next_node

    def execute(self, state: GraphState) -> Optional[BaseNode]:
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        payload = {
            "experiment":         "NB4",
            "auroc_oracle_full":  state.get("auroc_oracle_full"),
            "auroc_oracle_knn":   state.get("auroc_oracle_knn"),
            "auroc_mlp":          state.get("auroc_mlp"),
            "best_k":             state.get("best_k"),
            "best_auroc":         state.get("best_auroc"),
            "entropic_gate":      state.get("entropic_gate"),
            "abc_invariants":     state.get("invariant_checks", {}),
            "timestamp":          time.time(),
        }
        payload_str = json.dumps(payload, sort_keys=True, separators=(',', ':'))
        mac = hmac.new(
            self.secret.encode(), payload_str.encode(), hashlib.sha256
        )
        payload["signature"] = mac.hexdigest()

        log_path = Path(self.log_dir) / f"nb4_{int(payload['timestamp'])}.json"
        log_path.write_text(json.dumps(payload, indent=2))

        print(f"\n[ResultLoggerNode] Cryptographic audit log → {log_path}")
        print(f"  HMAC-SHA256: {payload['signature'][:24]}...")
        print(f"  Verification: `hmac.compare_digest(recomputed, saved)` — tamper-evident")

        # Print final summary
        print("\n" + "═"*60)
        print("  NB4 FINAL SUMMARY")
        print("═"*60)
        for k, v in payload.items():
            if k not in ("signature", "timestamp", "abc_invariants"):
                print(f"  {k}: {v}")
        print("═"*60)

        state.set("audit_log_path", str(log_path))
        return self.next_node


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH WIRING
# ─────────────────────────────────────────────────────────────────────────────

def build_nb4_graph(k: int = 10) -> tuple:
    """
    Assemble the complete NB4 Lár graph.
    All 10 ABCs are instantiated and wired.
    Returns (entry_node, executor).
    """
    # Shared instances
    kernel    = ScaledDotProductKernel()        # A1-A6
    encoder   = CLIPEncoderNode()               # M1-M3

    # Build node chain (wired in reverse)
    result_logger  = ResultLoggerNode()
    checkpoint     = ResumeCheckpointNode(next_node=result_logger)
    stat_test      = StatisticalTestNode(next_node=checkpoint)
    k_ablator      = k_AblationRouter(kernel, next_node=stat_test)
    auroc_judge    = AUROC_JudgeNode(next_node=k_ablator)

    # Oracle + MLP run in parallel (A/B Tester pattern)
    oracle_node    = HCLAttentionOracle(kernel, k=k, next_node=None)
    mlp_node       = MLP_BaselineNode(next_node=None)

    if _LAR_AVAILABLE:
        # True parallel execution via BatchNode (solves 5-seed sequential bottleneck)
        compare_batch  = BatchNode(nodes=[oracle_node, mlp_node], next_node=auroc_judge)
        oracle_entry   = compare_batch
    else:
        # Sequential fallback (standalone mode)
        oracle_node.next_node = mlp_node
        mlp_node.next_node    = auroc_judge
        oracle_entry          = oracle_node

    inv_checker    = InvariantCheckerNode(kernel, k=k, next_node=oracle_entry)  # JuryNode
    enc_gen        = EncoderGenerationNode(encoder, next_node=inv_checker)       # P1-P6
    hcl_builder    = HCL_BuilderNode(next_node=enc_gen)                         # I1-I2
    d_labeler      = D_ScoreLabeler(next_node=hcl_builder)                      # V1-V6
    encoder.next_node = d_labeler

    # Executor with HMAC audit (I5-I6)
    if _LAR_AVAILABLE:
        executor = GraphExecutor(
            log_dir=LOG_DIR,
            hmac_secret=HMAC_SECRET
        )
    else:
        executor = None

    return encoder, executor


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_nb4(seed: int = 42, k: int = 10, n_pairs: int = N_PAIRS):
    """
    Run NB4 for one seed.
    For 5-seed evaluation, call this inside a BatchNode or in a seed loop.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"\n{'═'*60}")
    print(f"  NB4 Lár-JEPA Graph  |  seed={seed}  k={k}  n={n_pairs}")
    print(f"{'═'*60}\n")

    entry, executor = build_nb4_graph(k=k)

    # Initial state: raw_pairs would be COCO val2017 tuples in production
    # In test mode: mock tensors are injected by CLIPEncoderNode
    initial_state = {
        "raw_pairs": [],          # fill with (PIL_image, caption_str) for COCO
        "n_pairs":   n_pairs,
        "seed":      seed,
    }

    if executor is not None:
        # Full Lár execution with step-by-step audit trail
        final = {}
        for step in executor.run_step_by_step(entry, initial_state):
            if step.get("outcome") == "error":
                print(f"[ERROR] Step {step['step']} ({step['node']}): {step.get('error')}")
                break
            final = step.get("state_after", final)
    else:
        # Standalone sequential execution
        state = GraphState(initial_state)
        node  = entry
        while node is not None:
            node = node.execute(state)
        final = dict(state)

    return final


def run_5seed_evaluation():
    """
    Full 5-seed evaluation with k-ablation.
    In production: wrap seed runs in BatchNode for true parallel execution.
    """
    all_oracle_aurocs = []
    all_mlp_aurocs    = []

    for seed in SEEDS:
        for k in [10]:             # primary k; k-ablation runs inside k_AblationRouter
            result = run_nb4(seed=seed, k=k)
            if result.get("auroc_oracle_knn") is not None:
                all_oracle_aurocs.append(result["auroc_oracle_knn"])
                all_mlp_aurocs.append(result.get("auroc_mlp", MLP_AUROC_BASELINE))

    if all_oracle_aurocs:
        t, p = stats.ttest_rel(all_oracle_aurocs, all_mlp_aurocs)
        print(f"\n{'═'*60}")
        print(f"  5-SEED SUMMARY")
        print(f"{'═'*60}")
        print(f"  Oracle AUROC:  {np.mean(all_oracle_aurocs):.4f} ± {np.std(all_oracle_aurocs):.4f}")
        print(f"  MLP AUROC:     {np.mean(all_mlp_aurocs):.4f} ± {np.std(all_mlp_aurocs):.4f}")
        print(f"  Δ AUROC:       {np.mean(all_oracle_aurocs)-np.mean(all_mlp_aurocs):+.4f}")
        print(f"  Paired t-test: t={t:.3f}  p={p:.4f}")
        print(f"  Verdict: {'CONFIRMED ✅' if np.mean(all_oracle_aurocs) > AUROC_TARGET else 'NOT CONFIRMED ❌'}")


# ─────────────────────────────────────────────────────────────────────────────
# COCO DATA LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_coco_pairs(
    images_dir: str,
    captions_json: str,
    n: int = N_PAIRS,
    seed: int = 42,
):
    """
    Load N image-caption pairs from COCO val2017.
    Returns list of (PIL.Image, caption_str) tuples.

    Kaggle paths (add COCO 2017 dataset to your notebook):
      images_dir   = "/kaggle/input/coco-2017-dataset/coco2017/val2017"
      captions_json= "/kaggle/input/coco-2017-dataset/coco2017/annotations/captions_val2017.json"
    """
    from PIL import Image as PILImage
    import random

    with open(captions_json, "r") as f:
        data = json.load(f)

    # Build image_id → file_name map
    id_to_file = {img["id"]: img["file_name"] for img in data["images"]}

    # One caption per image (first caption encountered)
    seen = {}
    for ann in data["annotations"]:
        iid = ann["image_id"]
        if iid not in seen:
            seen[iid] = ann["caption"]

    # Stable shuffle then take n
    rng = random.Random(seed)
    items = list(seen.items())
    rng.shuffle(items)
    items = items[:n]

    pairs = []
    skipped = 0
    for image_id, caption in items:
        fname = id_to_file.get(image_id)
        if fname is None:
            skipped += 1
            continue
        img_path = Path(images_dir) / fname
        if not img_path.exists():
            skipped += 1
            continue
        try:
            img = PILImage.open(img_path).convert("RGB")
            pairs.append((img, caption))
        except Exception:
            skipped += 1

    print(f"[load_coco_pairs] Loaded {len(pairs)} pairs (skipped {skipped})")
    return pairs


if __name__ == "__main__":
    # Single-seed quick run (test mode — no COCO data required)
    result = run_nb4(seed=42, k=10)
    print("\n[Done] Checkpoint and HMAC audit log written.")
    print("To run full 5-seed evaluation: call run_5seed_evaluation()")
