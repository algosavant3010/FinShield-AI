"""
FinShield AI — DeepFake Detection System
========================================
fusion.py: Cross-Attention Multimodal Fusion Module

Mathematical Formulation
─────────────────────────────────────────────────────────────────────────────
Given feature embeddings from three branches:

    F_s ∈ ℝ^(B×d)   — Spatial  (appearance / texture)
    F_t ∈ ℝ^(B×d)   — Temporal (motion / blinking / lip-sync)
    F_f ∈ ℝ^(B×d)   — Frequency (GAN artefacts / spectral anomalies)

Step 1 — Pairwise Cross-Attention
──────────────────────────────────
For each modality pair (query q, key-value kv):

    CrossAttn(q, kv) = softmax(QKᵀ / √d_k) · V

    where  Q = W_q · q,   K = W_k · kv,   V = W_v · kv

We compute 3 cross-attended representations:
    F̃_{s←t}  = CrossAttn(F_s, F_t)   — spatial attends to temporal
    F̃_{s←f}  = CrossAttn(F_s, F_f)   — spatial attends to frequency
    F̃_{t←f}  = CrossAttn(F_t, F_f)   — temporal attends to frequency

Step 2 — Adaptive Gating
─────────────────────────
Each modality m ∈ {s, t, f} gets a learnable gate score:

    g_m = σ(W_gate · [F_m ; F̃_m])    (σ = sigmoid)

    F̂_m = g_m ⊙ F_m + (1 - g_m) ⊙ F̃_m

This gate down-weights a modality when it is unreliable
(e.g., blurry video → temporal gate closes; clean image → frequency gate opens).

Step 3 — Soft Modality Weighting
──────────────────────────────────
A small MLP predicts per-modality attention weights:

    α = softmax(MLP([F̂_s ; F̂_t ; F̂_f]))   ∈ ℝ³

    F_fused = α_s · F̂_s  +  α_t · F̂_t  +  α_f · F̂_f

Step 4 — Classifier Head
──────────────────────────
    logit = MLP(F_fused)    ∈ ℝ²     (real / fake)

Advantages over early/late fusion:
  • Cross-attention lets each modality borrow complementary evidence.
  • Adaptive gates gracefully handle missing or corrupted modalities.
  • Soft weights are interpretable: α reveals which branch drives decisions.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
# Scaled Dot-Product Cross-Attention
# ─────────────────────────────────────────────────────────────

class CrossAttention(nn.Module):
    """
    Single-head cross-attention between a query modality and a key-value modality.

    Args:
        dim   : embedding dimension d
        heads : number of attention heads
        dropout: attention dropout probability
    """

    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.heads = heads
        self.d_k = dim // heads
        self.scale = math.sqrt(self.d_k)

        self.W_q = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)
        self.W_o = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        query: torch.Tensor,       # (B, d) — query modality
        kv: torch.Tensor,          # (B, d) — key-value modality
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            out            : (B, d)  attended representation
            attn_weights   : (B, heads, 1, 1) for visualisation
        """
        B, d = query.shape

        # Expand to sequence-of-1 for multi-head attention
        Q = self.W_q(query).view(B, self.heads, 1, self.d_k)  # (B, H, 1, d_k)
        K = self.W_k(kv).view(B, self.heads, 1, self.d_k)
        V = self.W_v(kv).view(B, self.heads, 1, self.d_k)

        # Scaled dot-product: (B, H, 1, 1)
        attn = (Q @ K.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Context vector: (B, H, 1, d_k) → (B, d)
        context = (attn_weights @ V).view(B, d)
        out = self.W_o(context)

        # Residual + LayerNorm (post-norm convention)
        out = self.norm(out + query)
        return out, attn_weights


# ─────────────────────────────────────────────────────────────
# Adaptive Modality Gate
# ─────────────────────────────────────────────────────────────

class AdaptiveModalityGate(nn.Module):
    """
    Learns a per-sample gate g ∈ [0,1] that blends the original embedding
    F_m with its cross-attended counterpart F̃_m.

    F̂_m = g ⊙ F_m + (1 − g) ⊙ F̃_m
    """

    def __init__(self, dim: int):
        super().__init__()
        # Gate MLP: [F_m ; F̃_m] → scalar gate
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(
        self, original: torch.Tensor, attended: torch.Tensor
    ) -> torch.Tensor:
        g = self.gate(torch.cat([original, attended], dim=-1))   # (B, d)
        out = g * original + (1.0 - g) * attended
        return self.norm(out)


# ─────────────────────────────────────────────────────────────
# Soft Modality Weighting (Step 3)
# ─────────────────────────────────────────────────────────────

class SoftModalityWeighting(nn.Module):
    """
    Predicts α = softmax(MLP([F̂_s ; F̂_t ; F̂_f])) ∈ ℝ³
    and returns the weighted sum F_fused = Σ α_m · F̂_m.
    """

    def __init__(self, dim: int, n_modalities: int = 3):
        super().__init__()
        self.n = n_modalities
        self.weight_net = nn.Sequential(
            nn.Linear(dim * n_modalities, dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim, n_modalities),
        )

    def forward(
        self, feats: list[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        feats : list of n_modalities tensors, each (B, d)
        Returns:
            fused   : (B, d)
            alpha   : (B, n_modalities)  — interpretable weights
        """
        concat = torch.cat(feats, dim=-1)                   # (B, n*d)
        alpha = F.softmax(self.weight_net(concat), dim=-1)  # (B, n)

        fused = sum(alpha[:, i:i+1] * feats[i] for i in range(self.n))
        return fused, alpha


# ─────────────────────────────────────────────────────────────
# Full Tri-Modal Cross-Attention Fusion
# ─────────────────────────────────────────────────────────────

class TriModalFusion(nn.Module):
    """
    Implements the full 4-step fusion described above:

    1. Pairwise cross-attention (s←t, s←f, t←f)
    2. Adaptive gating per modality
    3. Soft modality weighting → F_fused
    4. Classification MLP → logit

    Args:
        dim         : shared embedding dimension (all branches must output dim)
        num_classes : 2 for binary (real/fake)
        heads       : attention heads
        dropout     : dropout in heads and MLP
    """

    def __init__(
        self,
        dim: int = 512,
        num_classes: int = 2,
        heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim

        # ── Step 1: Pairwise cross-attention ──────────────────
        # Spatial attends to temporal & frequency
        self.ca_s_from_t = CrossAttention(dim, heads, dropout)
        self.ca_s_from_f = CrossAttention(dim, heads, dropout)
        # Temporal attends to frequency
        self.ca_t_from_f = CrossAttention(dim, heads, dropout)
        # Frequency attends to spatial (catches spatial-frequency inconsistency)
        self.ca_f_from_s = CrossAttention(dim, heads, dropout)

        # Aggregate two cross-attended views per modality
        self.agg_s = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.LayerNorm(dim))
        self.agg_t = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.LayerNorm(dim))
        self.agg_f = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.LayerNorm(dim))

        # ── Step 2: Adaptive gates ─────────────────────────────
        self.gate_s = AdaptiveModalityGate(dim)
        self.gate_t = AdaptiveModalityGate(dim)
        self.gate_f = AdaptiveModalityGate(dim)

        # ── Step 3: Soft weighting ─────────────────────────────
        self.soft_weight = SoftModalityWeighting(dim, n_modalities=3)

        # ── Step 4: Classifier head ────────────────────────────
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, num_classes),
        )

    def forward(
        self,
        F_s: torch.Tensor,    # (B, d) — spatial branch output
        F_t: torch.Tensor,    # (B, d) — temporal branch output
        F_f: torch.Tensor,    # (B, d) — frequency branch output
    ) -> dict:
        """
        Returns a dict with:
            logits   : (B, 2)
            alpha    : (B, 3)  — modality weights (interpretable)
            attn_st  : spatial←temporal attention weights
            attn_sf  : spatial←frequency attention weights
            attn_tf  : temporal←frequency attention weights
        """
        # ── Step 1: Cross-attention ────────────────────────────
        s_from_t, attn_st = self.ca_s_from_t(F_s, F_t)
        s_from_f, attn_sf = self.ca_s_from_f(F_s, F_f)
        t_from_f, attn_tf = self.ca_t_from_f(F_t, F_f)
        f_from_s, _       = self.ca_f_from_s(F_f, F_s)

        # Aggregate two views for spatial and frequency
        s_attended = self.agg_s(torch.cat([s_from_t, s_from_f], dim=-1))
        t_attended = t_from_f  # temporal only attends to frequency
        f_attended = self.agg_f(torch.cat([t_from_f, f_from_s], dim=-1))

        # ── Step 2: Adaptive gating ────────────────────────────
        F̂_s = self.gate_s(F_s, s_attended)
        F̂_t = self.gate_t(F_t, t_attended)
        F̂_f = self.gate_f(F_f, f_attended)

        # ── Step 3: Soft weighting ─────────────────────────────
        F_fused, alpha = self.soft_weight([F̂_s, F̂_t, F̂_f])

        # ── Step 4: Classification ─────────────────────────────
        logits = self.classifier(F_fused)

        return {
            "logits": logits,
            "alpha": alpha,
            "fused_feat": F_fused,
            "attn_st": attn_st,
            "attn_sf": attn_sf,
            "attn_tf": attn_tf,
            "gated_spatial": F̂_s,
            "gated_temporal": F̂_t,
            "gated_freq": F̂_f,
        }


# ─────────────────────────────────────────────────────────────
# Ablation Variants (for Section 5 of the paper / report)
# ─────────────────────────────────────────────────────────────

class TwoModalFusion(nn.Module):
    """Spatial + Temporal only (removes frequency branch from fusion)."""
    def __init__(self, dim: int = 512, num_classes: int = 2, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.ca = CrossAttention(dim, heads, dropout)
        self.gate = AdaptiveModalityGate(dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, num_classes),
        )

    def forward(self, F_s, F_t, **kwargs):
        s_t, _ = self.ca(F_s, F_t)
        F̂ = self.gate(F_s, s_t)
        return {"logits": self.classifier(torch.cat([F̂, F_t], dim=-1))}


class SimpleConcatFusion(nn.Module):
    """
    Naive concatenation baseline — no cross-attention, no gating.
    Used in ablation to prove superiority of cross-attention.
    """
    def __init__(self, dim: int = 512, num_classes: int = 2, n_modalities: int = 3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(dim * n_modalities, dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(dim, num_classes),
        )

    def forward(self, *feats):
        return {"logits": self.classifier(torch.cat(feats, dim=-1))}


# ─────────────────────────────────────────────────────────────
# Quick unit test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    B, d = 4, 512
    F_s = torch.randn(B, d)
    F_t = torch.randn(B, d)
    F_f = torch.randn(B, d)

    fusion = TriModalFusion(dim=d, num_classes=2, heads=8)
    out = fusion(F_s, F_t, F_f)

    print("TriModalFusion forward pass:")
    print(f"  logits  shape : {out['logits'].shape}")
    print(f"  alpha   shape : {out['alpha'].shape}")
    print(f"  alpha example : {out['alpha'][0].detach().tolist()}")
    print(f"  attn_st shape : {out['attn_st'].shape}")
    print("fusion.py OK")
