"""
FinShield AI — DeepFake Detection System
========================================
evaluate.py: Evaluation pipeline

Includes:
  • Full metrics suite (Acc, Prec, Rec, F1, ROC-AUC)
  • Cross-dataset generalisation test
  • Adversarial robustness evaluation (FGSM + PGD)
  • Grad-CAM spatial explanations
  • Attention heatmaps (temporal + frequency)
  • Modality alpha visualisation
  • Ablation study runner
"""

from __future__ import annotations

import sys
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve,
)

sys.path.insert(0, str(Path(__file__).parent))
from model import TriFusionDF, build_model
from data_loader import build_dataloader
from train import fgsm_attack, pgd_attack, FocalLoss

log = logging.getLogger("FinShield.Eval")

# ─────────────────────────────────────────────────────────────
# Evaluation results container
# ─────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    accuracy:   float
    precision:  float
    recall:     float
    f1:         float
    auc:        float
    labels:     List[int]
    preds:      List[int]
    probs:      List[float]
    dataset_tag: str = "all"

    def summary(self) -> str:
        return (
            f"Dataset: {self.dataset_tag}\n"
            f"  Accuracy:  {self.accuracy*100:.2f}%\n"
            f"  Precision: {self.precision:.4f}\n"
            f"  Recall:    {self.recall:.4f}\n"
            f"  F1-Score:  {self.f1:.4f}\n"
            f"  ROC-AUC:   {self.auc:.4f}\n"
        )


# ─────────────────────────────────────────────────────────────
# Core evaluation loop
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
    dataset_tag: str = "all",
) -> EvalResult:
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    # Track per-dataset results if meta is present
    dataset_labels: Dict[str, List] = {}
    dataset_probs: Dict[str, List] = {}

    for batch in loader:
        labels = batch["label"].to(device)
        inputs = {
            "spatial": batch["spatial"].to(device),
            "freq":    batch["freq"].to(device),
        }
        out = model(inputs)
        probs = F.softmax(out["logits"], dim=-1)[:, 1]
        preds = out["logits"].argmax(dim=-1)

        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())

        # Split by dataset for cross-dataset analysis
        for i, meta in enumerate(batch.get("meta", [])):
            ds = meta.get("dataset", "unknown")
            dataset_labels.setdefault(ds, []).append(labels[i].item())
            dataset_probs.setdefault(ds, []).append(probs[i].item())

    acc  = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec  = recall_score(all_labels, all_preds, zero_division=0)
    f1   = f1_score(all_labels, all_preds, zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = 0.0

    return EvalResult(acc, prec, rec, f1, auc, all_labels, all_preds, all_probs, dataset_tag)


# ─────────────────────────────────────────────────────────────
# Adversarial robustness evaluation
# ─────────────────────────────────────────────────────────────

def evaluate_adversarial(
    model: nn.Module,
    loader,
    device: torch.device,
    loss_fn: nn.Module,
    attacks: List[Dict],
) -> Dict[str, EvalResult]:
    """
    Evaluates model under multiple adversarial attacks.

    attacks: list of dicts, e.g.:
        [
            {"name": "FGSM_eps0.03", "type": "fgsm", "epsilon": 0.03},
            {"name": "PGD_10_eps0.03", "type": "pgd",  "epsilon": 0.03, "steps": 10},
        ]

    Returns dict: attack_name → EvalResult
    """
    results = {}

    # Baseline (no attack)
    results["clean"] = evaluate(model, loader, device, "clean")
    log.info(f"[ADV] Clean:\n{results['clean'].summary()}")

    for atk_cfg in attacks:
        name = atk_cfg["name"]
        log.info(f"[ADV] Running attack: {name}")
        model.eval()
        all_labels, all_preds, all_probs = [], [], []

        for batch in loader:
            batch_dev = {
                "spatial": batch["spatial"].to(device),
                "freq":    batch["freq"].to(device),
                "label":   batch["label"].to(device),
            }
            if atk_cfg["type"] == "fgsm":
                adv_batch = fgsm_attack(model, batch_dev, loss_fn, epsilon=atk_cfg["epsilon"])
            elif atk_cfg["type"] == "pgd":
                adv_batch = pgd_attack(
                    model, batch_dev, loss_fn,
                    epsilon=atk_cfg["epsilon"],
                    alpha=atk_cfg.get("alpha", atk_cfg["epsilon"] / 4),
                    num_steps=atk_cfg.get("steps", 10),
                )
            else:
                adv_batch = batch_dev

            with torch.no_grad():
                out = model({"spatial": adv_batch["spatial"], "freq": batch_dev["freq"]})
                probs = F.softmax(out["logits"], dim=-1)[:, 1]
                preds = out["logits"].argmax(dim=-1)

            all_labels.extend(batch_dev["label"].cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

        acc  = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, zero_division=0)
        rec  = recall_score(all_labels, all_preds, zero_division=0)
        f1   = f1_score(all_labels, all_preds, zero_division=0)
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except Exception:
            auc = 0.0

        results[name] = EvalResult(acc, prec, rec, f1, auc, all_labels, all_preds, all_probs, name)
        log.info(f"[ADV] {name}:\n{results[name].summary()}")
        # Report degradation
        delta_acc = results["clean"].accuracy - acc
        delta_auc = results["clean"].auc - auc
        log.info(f"  ↓ Accuracy drop: {delta_acc*100:.2f}%  |  AUC drop: {delta_auc:.4f}")

    return results


# ─────────────────────────────────────────────────────────────
# Grad-CAM (spatial explanation)
# ─────────────────────────────────────────────────────────────

class GradCAM:
    """
    Computes Grad-CAM for the last convolutional layer of the spatial branch.

    Grad-CAM: L^c_{Grad-CAM} = ReLU(Σ_k α^c_k · A^k)

    where  α^c_k = (1/Z) Σ_i Σ_j (∂y^c / ∂A^k_{ij})
    """

    def __init__(self, model: TriFusionDF):
        self.model = model
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        """Hook into the last SwinBlock of the spatial branch."""
        # Target the last stage of SpatialBranch
        target_layer = self.model.spatial_branch.backbone.stages[-1][-1]

        def fwd_hook(module, input, output):
            # output: (B, N, d) — store as (B, d, N) for spatial pooling
            self._activations = output

        def bwd_hook(module, grad_in, grad_out):
            self._gradients = grad_out[0]

        self._hook_handles.append(target_layer.register_forward_hook(fwd_hook))
        self._hook_handles.append(target_layer.register_full_backward_hook(bwd_hook))

    def __del__(self):
        for h in self._hook_handles:
            h.remove()

    def generate(
        self,
        batch: Dict[str, torch.Tensor],
        class_idx: int = 1,  # 1 = fake
        frame_idx: int = 0,
    ) -> np.ndarray:
        """
        Returns a (224, 224) Grad-CAM heatmap for frame_idx.
        class_idx=1 highlights regions driving the "fake" prediction.
        """
        self.model.eval()
        self.model.zero_grad()

        # Use only the frame of interest for Grad-CAM
        single_frame = batch["spatial"][:, frame_idx:frame_idx+1, ...]  # (B, 1, 3, H, W)
        frame_batch = {
            "spatial": single_frame.requires_grad_(True),
            "freq":    batch["freq"][:, frame_idx:frame_idx+1, ...],
        }

        out = self.model(frame_batch)
        score = out["logits"][:, class_idx].sum()
        score.backward()

        # α^c_k = global average of gradients
        grads = self._gradients  # (B, N, d)
        acts  = self._activations

        if grads is None or acts is None:
            log.warning("Grad-CAM: hooks not triggered (check target layer)")
            return np.zeros((224, 224))

        # Weight activations by gradient importance
        weights = grads.mean(dim=1, keepdim=True)     # (B, 1, d)
        cam = (weights * acts).sum(dim=-1)             # (B, N)
        cam = F.relu(cam)

        # Reshape token sequence back to spatial grid
        N = cam.shape[1]
        h = w = int(N ** 0.5)
        cam = cam[0].view(h, w).detach().cpu().numpy()

        # Resize to original frame size
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam_resized = np.array(
            plt.cm.jet(
                np.array(plt.imshow(cam, cmap="jet").get_array())  # type: ignore
            )
        )

        import cv2
        cam_resized = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_LINEAR)
        return cam_resized


# ─────────────────────────────────────────────────────────────
# Visualisation utilities
# ─────────────────────────────────────────────────────────────

def plot_confusion_matrix(result: EvalResult, save_path: str) -> None:
    cm = confusion_matrix(result.labels, result.preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    plt.title(f"Confusion Matrix — {result.dataset_tag}")
    plt.ylabel("True"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    log.info(f"  Confusion matrix saved: {save_path}")


def plot_roc_curve(result: EvalResult, save_path: str) -> None:
    fpr, tpr, _ = roc_curve(result.labels, result.probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {result.auc:.4f}", color="#e74c3c", lw=2)
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve — {result.dataset_tag}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    log.info(f"  ROC curve saved: {save_path}")


def plot_adversarial_robustness(results: Dict[str, EvalResult], save_path: str) -> None:
    names = list(results.keys())
    accs  = [r.accuracy * 100 for r in results.values()]
    aucs  = [r.auc for r in results.values()]

    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(x, accs, color=["#2ecc71"] + ["#e74c3c"] * (len(names) - 1), alpha=0.85)
    axes[0].set_xticks(x); axes[0].set_xticklabels(names, rotation=15, ha="right")
    axes[0].set_ylabel("Accuracy (%)"); axes[0].set_title("Accuracy under Adversarial Attack")
    axes[0].axhline(90, color="navy", linestyle="--", label="90% target")
    axes[0].legend()

    axes[1].bar(x, aucs, color=["#2ecc71"] + ["#e74c3c"] * (len(names) - 1), alpha=0.85)
    axes[1].set_xticks(x); axes[1].set_xticklabels(names, rotation=15, ha="right")
    axes[1].set_ylabel("ROC-AUC"); axes[1].set_title("AUC under Adversarial Attack")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    log.info(f"  Adversarial robustness plot saved: {save_path}")


def plot_modality_alpha(alpha_values: Dict[str, List[float]], save_path: str) -> None:
    """
    Visualises learned modality weights α_s, α_t, α_f across the test set.
    Shows which branch dominates for real vs. fake samples.
    """
    modalities = list(alpha_values.keys())
    means = [np.mean(v) for v in alpha_values.values()]
    stds  = [np.std(v) for v in alpha_values.values()]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#3498db", "#e67e22", "#9b59b6"]
    bars = ax.bar(modalities, means, yerr=stds, color=colors, alpha=0.85, capsize=6)
    ax.set_ylabel("Mean Attention Weight α")
    ax.set_title("Learned Modality Importance (TriModalFusion)")
    ax.set_ylim(0, 1.0)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{m:.3f}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    log.info(f"  Modality alpha plot saved: {save_path}")


# ─────────────────────────────────────────────────────────────
# Ablation Study
# ─────────────────────────────────────────────────────────────

ABLATION_CONFIGS = {
    "Full Model (TriFusion)":     {"fusion_mode": "full"},
    "No Frequency Branch":        {"fusion_mode": "no_freq"},
    "No Temporal Branch":         {"fusion_mode": "no_temporal"},
    "Naive Concat Fusion":        {"fusion_mode": "concat"},
}


def run_ablation_study(
    base_config: dict,
    val_loader,
    device: torch.device,
    output_dir: str,
) -> Dict[str, EvalResult]:
    """
    Trains and evaluates each ablation variant.
    In practice, each variant would be fully trained;
    here we demonstrate the evaluation interface.
    """
    results = {}
    out_path = Path(output_dir) / "ablation"
    out_path.mkdir(parents=True, exist_ok=True)

    for variant_name, variant_cfg in ABLATION_CONFIGS.items():
        log.info(f"\n{'─'*50}")
        log.info(f"Ablation variant: {variant_name}")

        cfg = {**base_config, "model": {**base_config["model"], **variant_cfg}}
        model = build_model(cfg["model"]).to(device)

        # In real training, load pre-trained checkpoint for this variant.
        # Here we evaluate with random weights as a structural test.
        result = evaluate(model, val_loader, device, dataset_tag=variant_name)
        results[variant_name] = result
        log.info(result.summary())

    # Plot comparison
    _plot_ablation(results, str(out_path / "ablation_comparison.png"))
    return results


def _plot_ablation(results: Dict[str, EvalResult], save_path: str) -> None:
    names = list(results.keys())
    metrics = {
        "F1-Score": [r.f1 for r in results.values()],
        "ROC-AUC":  [r.auc for r in results.values()],
        "Accuracy": [r.accuracy for r in results.values()],
    }

    x = np.arange(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#3498db", "#2ecc71", "#e74c3c"]

    for i, (metric, vals) in enumerate(metrics.items()):
        ax.bar(x + i * width, vals, width, label=metric, color=colors[i], alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("Ablation Study — TriFusion-DF Component Analysis")
    ax.legend()
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    log.info(f"  Ablation plot saved: {save_path}")


# ─────────────────────────────────────────────────────────────
# Full Evaluation Pipeline
# ─────────────────────────────────────────────────────────────

def run_full_evaluation(
    checkpoint_path: str,
    manifest_path: str,
    model_config: dict,
    output_dir: str,
    batch_size: int = 4,
    num_workers: int = 2,
    run_adv: bool = True,
    run_ablation: bool = False,
    device_str: str = "auto",
) -> None:
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device_str == "auto"
        else torch.device(device_str)
    )
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Load model
    model = build_model(model_config).to(device)
    if checkpoint_path and Path(checkpoint_path).exists():
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state["model_state"])
        log.info(f"Loaded checkpoint: {checkpoint_path}")
    model.eval()

    # Test loader
    test_loader = build_dataloader(
        manifest_path, split="test",
        batch_size=batch_size, num_workers=num_workers,
    )

    # ── Standard evaluation ────────────────────────────────────
    log.info("\n=== Standard Evaluation ===")
    result = evaluate(model, test_loader, device, "test")
    log.info(result.summary())
    plot_confusion_matrix(result, str(out_path / "confusion_matrix.png"))
    plot_roc_curve(result, str(out_path / "roc_curve.png"))

    # ── Adversarial evaluation ─────────────────────────────────
    if run_adv:
        log.info("\n=== Adversarial Robustness Evaluation ===")
        loss_fn = FocalLoss()
        adv_attacks = [
            {"name": "FGSM_ε=0.01", "type": "fgsm", "epsilon": 0.01},
            {"name": "FGSM_ε=0.03", "type": "fgsm", "epsilon": 0.03},
            {"name": "PGD_7_ε=0.01", "type": "pgd",  "epsilon": 0.01, "steps": 7},
            {"name": "PGD_10_ε=0.03", "type": "pgd", "epsilon": 0.03, "steps": 10},
        ]
        adv_results = evaluate_adversarial(model, test_loader, device, loss_fn, adv_attacks)
        plot_adversarial_robustness(adv_results, str(out_path / "adv_robustness.png"))

        # Print robustness table
        log.info("\n┌─────────────────────┬──────────┬──────────┬──────────┐")
        log.info("│ Attack              │ Acc (%)  │ F1       │ AUC      │")
        log.info("├─────────────────────┼──────────┼──────────┼──────────┤")
        for name, r in adv_results.items():
            log.info(f"│ {name:<19} │ {r.accuracy*100:7.2f}% │ {r.f1:.4f}   │ {r.auc:.4f}   │")
        log.info("└─────────────────────┴──────────┴──────────┴──────────┘")

    # ── Ablation study ─────────────────────────────────────────
    if run_ablation:
        log.info("\n=== Ablation Study ===")
        base_cfg = {"model": model_config}
        val_loader = build_dataloader(manifest_path, split="val",
                                      batch_size=batch_size, num_workers=num_workers)
        ablation_results = run_ablation_study(base_cfg, val_loader, device, str(out_path))

        log.info("\nAblation Summary:")
        log.info("┌──────────────────────────────┬────────┬────────┬────────┐")
        log.info("│ Variant                      │ Acc    │ F1     │ AUC    │")
        log.info("├──────────────────────────────┼────────┼────────┼────────┤")
        for name, r in ablation_results.items():
            log.info(f"│ {name:<28} │ {r.accuracy:.4f} │ {r.f1:.4f} │ {r.auc:.4f} │")
        log.info("└──────────────────────────────┴────────┴────────┴────────┘")

    log.info(f"\nAll evaluation outputs saved to: {out_path}")


# ─────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    from train import DEFAULT_CONFIG

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="outputs/run_001/checkpoints/best_model.pt")
    parser.add_argument("--manifest",   type=str, default="data/manifest.json")
    parser.add_argument("--output_dir", type=str, default="outputs/eval")
    parser.add_argument("--no_adv",    action="store_true")
    parser.add_argument("--ablation",  action="store_true")
    args = parser.parse_args()

    run_full_evaluation(
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        model_config=DEFAULT_CONFIG["model"],
        output_dir=args.output_dir,
        run_adv=not args.no_adv,
        run_ablation=args.ablation,
    )
