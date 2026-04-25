"""
FinShield AI — DeepFake Detection System
========================================
model.py: TriFusion-DF Architecture

Architecture Overview
─────────────────────────────────────────────────────────────────────────────

  Video Input (B, T, 3, 224, 224)
         │
         ├──────────────────────────────────────────────────────┐
         │                                                      │
  ┌──────▼──────────────────────────────────────────────┐      │
  │   SPATIAL BRANCH (per-frame)                        │      │
  │                                                     │      │
  │   Swin-Transformer-V2 (backbone, pretrained)        │      │
  │        ↓                                            │      │
  │   Multi-Scale Feature Pyramid (C3, C4, C5)         │      │
  │        ↓                                            │      │
  │   Temporal average pooling over T frames            │      │
  │        ↓                                            │      │
  │   F_s ∈ ℝ^(B×d)                                    │      │
  └──────────────────────────────────────────────────────┘      │
                                                                │
  ┌──────────────────────────────────────────────────────┐      │
  │   TEMPORAL BRANCH                                    │      │
  │                                                     │      │
  │   CNN Frame Encoder (EfficientNet-B0 stem)          │      │
  │        ↓                                            │      │
  │   Frame-sequence: (B, T, C, H, W) → (B, T, feat)   │      │
  │        ↓                                            │      │
  │   Temporal Transformer (4 layers, causal mask)      │      │
  │        ↓                                            │      │
  │   BiLSTM (2 layers) for motion dynamics             │      │
  │        ↓                                            │      │
  │   F_t ∈ ℝ^(B×d)                                    │      │
  └──────────────────────────────────────────────────────┘      │
                                                                │
  ┌──────────────────────────────────────────────────────┐      │
  │   FREQUENCY BRANCH                                   │      │
  │   (receives DCT/FFT maps, computed in DataLoader)    │◄─────┘
  │                                                     │
  │   Patch Embedding (16×16 patches)                   │
  │        ↓                                            │
  │   4-layer ViT Encoder (lightweight)                 │
  │        ↓                                            │
  │   GAN-artefact detection head                       │
  │        ↓                                            │
  │   F_f ∈ ℝ^(B×d)                                    │
  └──────────────────────────────────────────────────────┘

         │            │            │
         F_s          F_t          F_f
         └────────────┼────────────┘
                      ↓
          TriModalFusion (cross-attention)
                      ↓
              Classifier Head
                      ↓
              logit ∈ ℝ^(B×2)

─────────────────────────────────────────────────────────────────────────────
Why This > Plain CNN-LSTM:
  • Swin-V2: hierarchical shifted-window attention → global context at O(n log n)
  • Temporal Transformer: long-range dependency modelling across frames
  • BiLSTM: fine-grained sequential motion with gradient highways
  • Frequency ViT: catches imperceptible GAN spectral signatures
  • Cross-attention fusion: each modality *queries* others for missing evidence
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from fusion import TriModalFusion, TwoModalFusion, SimpleConcatFusion

# ─────────────────────────────────────────────────────────────
# Utility: positional encoding
# ─────────────────────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d)
        return self.dropout(x + self.pe[:, :x.size(1)])


# ─────────────────────────────────────────────────────────────
# 1. SPATIAL BRANCH — Swin-V2 inspired hierarchical ViT
# ─────────────────────────────────────────────────────────────

class WindowAttention(nn.Module):
    """
    Shifted-window multi-head self-attention (simplified SwinV2 block).
    In production, swap for timm's SwinTransformerV2 pretrained weights.
    """
    def __init__(self, dim: int, window_size: int = 7, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.d_k = dim // num_heads
        self.scale = self.d_k ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

        # Relative position bias table (w²*2-1)² → num_heads
        self.relative_pos_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads)
        )
        nn.init.trunc_normal_(self.relative_pos_bias_table, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, d)
        B, N, d = x.shape
        h = self.num_heads
        dk = d // h

        qkv = self.qkv(x).chunk(3, dim=-1)
        Q, K, V = [t.view(B, N, h, dk).transpose(1, 2) for t in qkv]

        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)

        out = (attn @ V).transpose(1, 2).reshape(B, N, d)
        return self.norm(self.proj(out) + x)


class SwinBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.attn = WindowAttention(dim, num_heads=num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(self.norm1(x)) + x
        return self.mlp(self.norm2(x)) + x


class SpatialBranch(nn.Module):
    """
    Hierarchical spatial feature extractor.

    Architecture:
      1. Patch embedding: 4×4 patches → C₀ = 96 channels
      2. 4 Swin stages with patch-merging downsampling
      3. Multi-scale feature aggregation (C3, C4, C5 concatenated)
      4. Projection to shared embedding dim d

    For real training, replace with:
        import timm; timm.create_model('swinv2_base_window8_256', pretrained=True)
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dim: int = 96,
        out_dim: int = 512,
        depths: List[int] = [2, 2, 6, 2],
        num_heads: List[int] = [3, 6, 12, 24],
        dropout: float = 0.1,
    ):
        super().__init__()
        n_patches = (img_size // patch_size) ** 2

        # Patch embedding
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size),
            Rearrange("b c h w -> b (h w) c"),
            nn.LayerNorm(embed_dim),
        )

        # 4 Swin stages
        dims = [embed_dim * (2 ** i) for i in range(4)]
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for i, (depth, heads) in enumerate(zip(depths, num_heads)):
            stage = nn.Sequential(*[SwinBlock(dims[i], heads, dropout=dropout) for _ in range(depth)])
            self.stages.append(stage)
            if i < 3:
                self.downsamples.append(nn.Sequential(
                    Rearrange("b (h w) c -> b c h w", h=img_size // (patch_size * 2 ** i)),
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                    Rearrange("b c h w -> b (h w) c"),
                    nn.LayerNorm(dims[i + 1]),
                ))

        # Multi-scale aggregation (stages 2, 3, 4 → concat → project)
        ms_dim = dims[1] + dims[2] + dims[3]
        self.ms_pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(
            nn.Linear(ms_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, 224, 224) — single frame
        Returns: (B, out_dim)
        """
        x = self.patch_embed(x)    # (B, N, C0)

        multi_scale_feats = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i > 0:  # collect stages 1, 2, 3 (0-indexed)
                feat = self.ms_pool(x.transpose(1, 2)).squeeze(-1)  # (B, C_i)
                multi_scale_feats.append(feat)
            if i < len(self.downsamples):
                x = self.downsamples[i](x)

        combined = torch.cat(multi_scale_feats, dim=-1)   # (B, sum_dims)
        return self.proj(combined)                         # (B, out_dim)


class SpatialBranchWrapper(nn.Module):
    """Applies SpatialBranch per-frame and pools over time."""

    def __init__(self, out_dim: int = 512, **kwargs):
        super().__init__()
        self.backbone = SpatialBranch(out_dim=out_dim, **kwargs)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: (B, T, 3, 224, 224) → (B, d)"""
        B, T, C, H, W = frames.shape
        # Merge B and T for parallel processing
        frames_flat = frames.view(B * T, C, H, W)
        feats = self.backbone(frames_flat)              # (B*T, d)
        feats = feats.view(B, T, -1)                   # (B, T, d)
        return feats.mean(dim=1)                        # (B, d) — temporal avg


# ─────────────────────────────────────────────────────────────
# 2. TEMPORAL BRANCH — CNN stem + Temporal Transformer + BiLSTM
# ─────────────────────────────────────────────────────────────

class LightCNNFrameEncoder(nn.Module):
    """Lightweight CNN that extracts per-frame embeddings (≈ EfficientNet-B0 stem)."""

    def __init__(self, in_channels: int = 3, out_channels: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),          nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),         nn.GELU(),
            nn.Conv2d(128, out_channels, 3, stride=2, padding=1), nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B*T, 3, H, W) → (B*T, out_channels)"""
        return self.pool(self.conv(x)).flatten(1)


class TemporalTransformerEncoder(nn.Module):
    """
    4-layer causal temporal transformer.
    Attends to eye blinking (~0.15–0.4s), lip sync (phoneme-level),
    and head-pose jitter across the clip.
    """

    def __init__(self, d_model: int = 256, nhead: int = 8, num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.pos_enc = SinusoidalPositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True   # pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        """x: (B, T, d_model) → (B, T, d_model)"""
        x = self.pos_enc(x)
        T = x.size(1)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool() if causal else None
        return self.transformer(x, mask=mask)


class TemporalBranch(nn.Module):
    """
    Full temporal branch: CNN frame encoder → Temporal Transformer → BiLSTM.

    The BiLSTM after the Transformer serves as a 'motion smoothing' layer:
    it accumulates long-range temporal context with learnable forget gates
    that are particularly sensitive to frame-to-frame inconsistencies.
    """

    def __init__(
        self,
        in_channels: int = 3,
        cnn_dim: int = 256,
        transformer_dim: int = 256,
        lstm_hidden: int = 256,
        out_dim: int = 512,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cnn = LightCNNFrameEncoder(in_channels, cnn_dim)
        self.input_proj = nn.Linear(cnn_dim, transformer_dim)
        self.transformer = TemporalTransformerEncoder(transformer_dim, num_layers=num_layers, dropout=dropout)
        self.bilstm = nn.LSTM(
            input_size=transformer_dim,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        # BiLSTM output = 2*lstm_hidden
        self.out_proj = nn.Sequential(
            nn.LayerNorm(lstm_hidden * 2),
            nn.Linear(lstm_hidden * 2, out_dim),
            nn.GELU(),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: (B, T, 3, H, W) → (B, out_dim)"""
        B, T, C, H, W = frames.shape
        flat = frames.view(B * T, C, H, W)
        cnn_feats = self.cnn(flat).view(B, T, -1)      # (B, T, cnn_dim)
        x = self.input_proj(cnn_feats)                  # (B, T, transformer_dim)
        x = self.transformer(x)                         # (B, T, transformer_dim)
        lstm_out, _ = self.bilstm(x)                    # (B, T, 2*lstm_hidden)
        # Aggregate: mean + max pooling over time → richer temporal summary
        feat = lstm_out.mean(dim=1) + lstm_out.max(dim=1).values
        return self.out_proj(feat)                       # (B, out_dim)


# ─────────────────────────────────────────────────────────────
# 3. FREQUENCY BRANCH — Patch-ViT on DCT / FFT maps
# ─────────────────────────────────────────────────────────────

class FrequencyPatchEmbedding(nn.Module):
    """Divides frequency map into patches and projects to embedding dim."""
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 256):
        super().__init__()
        n_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.n_patches = n_patches
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches + 1, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) → (B, n_patches+1, embed_dim)"""
        B = x.size(0)
        x = self.patch_embed(x)                          # (B, embed_dim, h, w)
        x = x.flatten(2).transpose(1, 2)                 # (B, n_patches, embed_dim)
        cls = repeat(self.cls_token, "1 1 d -> b 1 d", b=B)
        x = torch.cat([cls, x], dim=1)                   # (B, n_patches+1, d)
        return x + self.pos_embed


class FrequencyViTBlock(nn.Module):
    """Standard ViT encoder block (pre-norm) for frequency domain tokens."""
    def __init__(self, dim: int, heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim), nn.Dropout(dropout),
        )
        self.drop_path = nn.Dropout(dropout * 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = self.norm1(x)
        attn_out, _ = self.attn(n, n, n)
        x = x + self.drop_path(attn_out)
        return x + self.drop_path(self.mlp(self.norm2(x)))


class FrequencyBranch(nn.Module):
    """
    Lightweight ViT applied to DCT/FFT frequency maps.

    Key capability: detects GAN checkerboard artefacts, spectral inconsistencies,
    and inter-frame frequency drift — all invisible to spatial-only models.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 256,
        depth: int = 4,
        num_heads: int = 8,
        out_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_embed = FrequencyPatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.encoder = nn.Sequential(
            *[FrequencyViTBlock(embed_dim, num_heads, dropout=dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
        )

    def forward(self, freq_maps: torch.Tensor) -> torch.Tensor:
        """
        freq_maps: (B, T, 3, 224, 224) — DCT or FFT maps per frame
        Returns:   (B, out_dim)
        """
        B, T, C, H, W = freq_maps.shape
        flat = freq_maps.view(B * T, C, H, W)
        tokens = self.patch_embed(flat)          # (B*T, N+1, d)
        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)
        cls = tokens[:, 0]                       # CLS token: (B*T, d)
        cls = cls.view(B, T, -1).mean(dim=1)    # temporal avg: (B, d)
        return self.proj(cls)                    # (B, out_dim)


# ─────────────────────────────────────────────────────────────
# 4. TRIFUSION-DF — Full Model
# ─────────────────────────────────────────────────────────────

class TriFusionDF(nn.Module):
    """
    TriFusion Deepfake Detector — main model integrating all three branches.

    Args:
        embed_dim      : shared embedding dimension across all branches
        num_classes    : 2 (real / fake)
        num_frames     : temporal clip length
        spatial_depths : Swin stage depths
        fusion_heads   : attention heads in cross-attention fusion
        dropout        : global dropout rate
        fusion_mode    : "full" | "no_freq" | "no_temporal" | "concat" (ablation)
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_classes: int = 2,
        num_frames: int = 16,
        spatial_depths: List[int] = [2, 2, 6, 2],
        fusion_heads: int = 8,
        dropout: float = 0.1,
        fusion_mode: str = "full",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.fusion_mode = fusion_mode

        # ── Branch instantiation ──────────────────────────────
        self.spatial_branch = SpatialBranchWrapper(out_dim=embed_dim, depths=spatial_depths, dropout=dropout)
        self.temporal_branch = TemporalBranch(out_dim=embed_dim, dropout=dropout)
        self.frequency_branch = FrequencyBranch(out_dim=embed_dim, dropout=dropout)

        # ── Fusion ─────────────────────────────────────────────
        if fusion_mode == "full":
            self.fusion = TriModalFusion(dim=embed_dim, num_classes=num_classes, heads=fusion_heads, dropout=dropout)
        elif fusion_mode == "no_freq":
            self.fusion = TwoModalFusion(dim=embed_dim, num_classes=num_classes, heads=fusion_heads, dropout=dropout)
        elif fusion_mode == "no_temporal":
            self.fusion = TwoModalFusion(dim=embed_dim, num_classes=num_classes, heads=fusion_heads, dropout=dropout)
        elif fusion_mode == "concat":
            n = 3 if fusion_mode == "full" else 2
            self.fusion = SimpleConcatFusion(dim=embed_dim, num_classes=num_classes, n_modalities=n)
        else:
            raise ValueError(f"Unknown fusion_mode: {fusion_mode}")

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict:
        """
        batch keys:
            spatial : (B, T, 3, 224, 224)  — normalised RGB
            freq    : (B, T, 3, 224, 224)  — DCT/FFT maps

        Returns dict with 'logits', 'alpha', attention maps, etc.
        """
        spatial = batch["spatial"]
        freq = batch["freq"]

        # ── Extract per-branch embeddings ─────────────────────
        F_s = self.spatial_branch(spatial)      # (B, d)
        F_t = self.temporal_branch(spatial)     # (B, d) — uses same RGB frames
        F_f = self.frequency_branch(freq)       # (B, d)

        # ── Fuse ─────────────────────────────────────────────
        if self.fusion_mode == "no_freq":
            out = self.fusion(F_s, F_t)
        elif self.fusion_mode == "no_temporal":
            out = self.fusion(F_s, F_f)
        else:
            out = self.fusion(F_s, F_t, F_f)

        out["branch_feats"] = {"spatial": F_s, "temporal": F_t, "frequency": F_f}
        return out

    def predict_proba(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Returns fake probability for each sample in batch. Shape: (B,)"""
        with torch.no_grad():
            out = self.forward(batch)
        return F.softmax(out["logits"], dim=-1)[:, 1]  # P(fake)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────

def build_model(config: dict) -> TriFusionDF:
    """Build model from a config dict (loaded from configs/model_config.yaml)."""
    return TriFusionDF(
        embed_dim=config.get("embed_dim", 512),
        num_classes=config.get("num_classes", 2),
        num_frames=config.get("num_frames", 16),
        spatial_depths=config.get("spatial_depths", [2, 2, 6, 2]),
        fusion_heads=config.get("fusion_heads", 8),
        dropout=config.get("dropout", 0.1),
        fusion_mode=config.get("fusion_mode", "full"),
    )


# ─────────────────────────────────────────────────────────────
# ONNX / TorchScript export
# ─────────────────────────────────────────────────────────────

def export_torchscript(model: TriFusionDF, output_path: str, num_frames: int = 16):
    """Export model to TorchScript for deployment."""
    model.eval()
    dummy_batch = {
        "spatial": torch.randn(1, num_frames, 3, 224, 224),
        "freq":    torch.randn(1, num_frames, 3, 224, 224),
    }
    try:
        traced = torch.jit.trace(model, (dummy_batch,), strict=False)
        traced.save(output_path)
        print(f"[Export] TorchScript saved: {output_path}")
    except Exception as e:
        print(f"[Export] TorchScript failed (use ONNX instead): {e}")


def export_onnx(model: TriFusionDF, output_path: str, num_frames: int = 16):
    """Export model to ONNX for cross-platform deployment."""
    model.eval()
    dummy_spatial = torch.randn(1, num_frames, 3, 224, 224)
    dummy_freq    = torch.randn(1, num_frames, 3, 224, 224)
    # For ONNX, we flatten the forward to take positional args
    class ONNXWrapper(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, spatial, freq):
            return self.m({"spatial": spatial, "freq": freq})["logits"]

    wrapper = ONNXWrapper(model)
    torch.onnx.export(
        wrapper,
        (dummy_spatial, dummy_freq),
        output_path,
        input_names=["spatial", "freq"],
        output_names=["logits"],
        dynamic_axes={"spatial": {0: "batch"}, "freq": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        verbose=False,
    )
    print(f"[Export] ONNX saved: {output_path}")


# ─────────────────────────────────────────────────────────────
# Architecture diagram (text)
# ─────────────────────────────────────────────────────────────

ARCHITECTURE_DIAGRAM = """
╔══════════════════════════════════════════════════════════════════════════╗
║                       TriFusion-DF Architecture                        ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║  Input: Video Clip (B, T=16, 3, 224, 224)                              ║
║         │                                                              ║
║    ┌────┴────────────────────────────────────────────────────┐         ║
║    │                  FACE DETECTION                         │         ║
║    │  RetinaFace → Crop (224×224) + Align + DCT/FFT maps     │         ║
║    └────┬────────────────────────┬───────────────────────────┘         ║
║         │ RGB Frames             │ Frequency Maps                      ║
║    ┌────▼────────┐  ┌────────────▼──────────────────────┐             ║
║    │  SPATIAL    │  │          FREQUENCY                 │             ║
║    │  BRANCH     │  │          BRANCH                    │             ║
║    │             │  │                                    │             ║
║    │ Swin-V2     │  │  Patch Embed (16×16)               │             ║
║    │ Stage 1-4   │  │  4-layer ViT Encoder               │             ║
║    │ (shifted    │  │  CLS token pooling                 │             ║
║    │  window     │  │  → detects GAN checkerboard,       │             ║
║    │  attention) │  │    spectral artefacts              │             ║
║    │             │  │                                    │             ║
║    │ Multi-Scale │  │  F_f ∈ ℝ^(B×512)                  │             ║
║    │ Feature     │  └────────────────────────────────────┘             ║
║    │ Pyramid     │                   │                                 ║
║    │             │        ┌──────────┘                                 ║
║    │ F_s ∈ ℝ^(B×512)     │                                            ║
║    └────┬────────┘        │                                            ║
║         │                 │                                            ║
║    ┌────▼────────┐        │                                            ║
║    │  TEMPORAL   │        │                                            ║
║    │  BRANCH     │        │                                            ║
║    │             │        │                                            ║
║    │ CNN Stem    │        │                                            ║
║    │ (per frame) │        │                                            ║
║    │ ↓           │        │                                            ║
║    │ Temporal    │        │                                            ║
║    │ Transformer │        │                                            ║
║    │ (4L causal) │        │                                            ║
║    │ ↓           │        │                                            ║
║    │ BiLSTM      │        │                                            ║
║    │ (2L, bidirectional)  │                                            ║
║    │             │        │                                            ║
║    │ F_t ∈ ℝ^(B×512)     │                                            ║
║    └────┬────────┘        │                                            ║
║         │                 │                                            ║
║    ┌────▼─────────────────▼──────────────────────────────┐            ║
║    │         TriModalFusion (Cross-Attention)             │            ║
║    │                                                      │            ║
║    │  Step 1: Pairwise Cross-Attention                    │            ║
║    │    F̃_{s←t} = CrossAttn(F_s, F_t)                    │            ║
║    │    F̃_{s←f} = CrossAttn(F_s, F_f)                    │            ║
║    │    F̃_{t←f} = CrossAttn(F_t, F_f)                    │            ║
║    │                                                      │            ║
║    │  Step 2: Adaptive Gating                             │            ║
║    │    g_m = σ(MLP([F_m ; F̃_m]))                        │            ║
║    │    F̂_m = g_m ⊙ F_m + (1-g_m) ⊙ F̃_m                │            ║
║    │                                                      │            ║
║    │  Step 3: Soft Modality Weighting                     │            ║
║    │    α = softmax(MLP([F̂_s; F̂_t; F̂_f]))               │            ║
║    │    F_fused = α_s·F̂_s + α_t·F̂_t + α_f·F̂_f           │            ║
║    │                                                      │            ║
║    │  Step 4: Classifier                                  │            ║
║    │    logit = MLP(F_fused) ∈ ℝ²                        │            ║
║    └──────────────────────────────────────────────────────┘            ║
║                              │                                         ║
║                   P(real) | P(fake)                                    ║
╚══════════════════════════════════════════════════════════════════════════╝
"""


if __name__ == "__main__":
    print(ARCHITECTURE_DIAGRAM)

    # Smoke test — small tensors
    model = TriFusionDF(embed_dim=256, num_frames=4, spatial_depths=[1, 1, 2, 1], fusion_heads=4)
    print(f"Parameters: {model.count_parameters():,}")

    dummy_batch = {
        "spatial": torch.randn(2, 4, 3, 224, 224),
        "freq":    torch.randn(2, 4, 3, 224, 224),
    }
    out = model(dummy_batch)
    print(f"logits shape : {out['logits'].shape}")
    print(f"alpha        : {out['alpha'][0].detach().tolist()}")
    print("model.py OK")
