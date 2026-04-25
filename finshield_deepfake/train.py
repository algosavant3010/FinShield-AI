"""
FinShield AI — DeepFake Detection System
========================================
train.py: Production training loop

Features:
  • Mixed-precision (FP16) via torch.cuda.amp
  • Cosine LR schedule with linear warmup
  • Adversarial training via FGSM + PGD (interleaved with clean batches)
  • Gradient clipping (norm ≤ 1.0)
  • Checkpoint save/resume
  • TensorBoard logging
  • Early stopping with patience
  • Class-weighted focal loss for imbalance
"""

from __future__ import annotations

import os
import sys
import time
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

# Local imports
sys.path.insert(0, str(Path(__file__).parent))
from model import TriFusionDF, build_model
from data_loader import build_dataloader, DeepfakeVideoDataset

# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("FinShield.Train")


# ─────────────────────────────────────────────────────────────
# Loss — Focal Loss (handles class imbalance better than CE)
# ─────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal loss: FL(p_t) = -α_t (1 − p_t)^γ log(p_t)

    γ > 0 reduces the relative loss for well-classified examples,
    putting more focus on hard, misclassified samples.
    Especially useful when real:fake ratio is skewed.
    """

    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        p_t = torch.exp(-ce)
        focal = (1 - p_t) ** self.gamma * ce
        return focal.mean()


# ─────────────────────────────────────────────────────────────
# Adversarial Attack Utilities (FGSM + PGD)
# ─────────────────────────────────────────────────────────────

def fgsm_attack(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    loss_fn: nn.Module,
    epsilon: float = 0.03,
) -> Dict[str, torch.Tensor]:
    """
    Fast Gradient Sign Method (Goodfellow et al., 2014)

    δ = ε · sign(∇_x L(f(x), y))

    Perturbs the spatial frames only (most impactful attack surface).
    """
    spatial = batch["spatial"].clone().requires_grad_(True)
    perturbed_batch = {**batch, "spatial": spatial}

    model.zero_grad()
    out = model(perturbed_batch)
    loss = loss_fn(out["logits"], batch["label"])
    loss.backward()

    with torch.no_grad():
        adv_spatial = spatial + epsilon * spatial.grad.sign()
        adv_spatial = torch.clamp(adv_spatial, -3.0, 3.0)  # stay in normalised range

    return {**batch, "spatial": adv_spatial.detach()}


def pgd_attack(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    loss_fn: nn.Module,
    epsilon: float = 0.03,
    alpha: float = 0.007,
    num_steps: int = 10,
) -> Dict[str, torch.Tensor]:
    """
    Projected Gradient Descent (Madry et al., 2017)

    x_{t+1} = Π_{x+S}(x_t + α · sign(∇_x L(f(x_t), y)))

    Stronger than FGSM; used for adversarial training robustness.
    """
    original = batch["spatial"].clone()
    adv = original + torch.empty_like(original).uniform_(-epsilon, epsilon)
    adv = torch.clamp(adv, -3.0, 3.0)

    for step in range(num_steps):
        adv = adv.clone().requires_grad_(True)
        perturbed_batch = {**batch, "spatial": adv}
        model.zero_grad()
        out = model(perturbed_batch)
        loss = loss_fn(out["logits"], batch["label"])
        loss.backward()

        with torch.no_grad():
            adv = adv + alpha * adv.grad.sign()
            delta = torch.clamp(adv - original, -epsilon, epsilon)
            adv = torch.clamp(original + delta, -3.0, 3.0)

    return {**batch, "spatial": adv.detach()}


# ─────────────────────────────────────────────────────────────
# LR Scheduler (cosine + linear warmup)
# ─────────────────────────────────────────────────────────────

class WarmupCosineScheduler(optim.lr_scheduler._LRScheduler):
    """
    Linear warmup for `warmup_steps` steps, then cosine annealing to `min_lr`.
    Outperforms step-decay for transformer training.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        if step < self.warmup_steps:
            factor = step / max(1, self.warmup_steps)
        else:
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            factor = 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())
            factor = max(factor, self.min_lr / max(b for b in self.base_lrs))
        return [base_lr * factor for base_lr in self.base_lrs]


# ─────────────────────────────────────────────────────────────
# Checkpoint management
# ─────────────────────────────────────────────────────────────

def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    epoch: int,
    metrics: dict,
    path: str,
    is_best: bool = False,
) -> None:
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "sched_state": scheduler.state_dict(),
        "metrics": metrics,
    }
    torch.save(state, path)
    if is_best:
        best_path = str(Path(path).parent / "best_model.pt")
        torch.save(state, best_path)
        log.info(f"  ✓ New best model saved: {best_path}")


def load_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    path: str,
) -> int:
    """Returns the epoch to resume from."""
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optim_state"])
    scheduler.load_state_dict(state["sched_state"])
    log.info(f"Resumed from epoch {state['epoch']}: {state['metrics']}")
    return state["epoch"] + 1


# ─────────────────────────────────────────────────────────────
# Single epoch training
# ─────────────────────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader,
    optimizer: optim.Optimizer,
    scheduler,
    loss_fn: nn.Module,
    scaler: GradScaler,
    device: torch.device,
    adv_config: dict,
    epoch: int,
    writer: Optional[SummaryWriter] = None,
    log_interval: int = 10,
) -> dict:
    model.train()
    total_loss = 0.0
    total_adv_loss = 0.0
    correct = 0
    total = 0
    t0 = time.time()

    for batch_idx, batch in enumerate(loader):
        # Move to device
        batch = {
            "spatial": batch["spatial"].to(device, non_blocking=True),
            "freq":    batch["freq"].to(device, non_blocking=True),
            "label":   batch["label"].to(device, non_blocking=True),
        }

        # ── Clean forward pass ─────────────────────────────────
        with autocast(enabled=scaler.is_enabled()):
            out = model(batch)
            loss_clean = loss_fn(out["logits"], batch["label"])

        # ── Adversarial training (every `adv_every` steps) ────
        adv_loss = torch.tensor(0.0, device=device)
        if adv_config["enabled"] and batch_idx % adv_config["adv_every"] == 0:
            model.eval()  # disable dropout during attack generation
            if adv_config["attack"] == "pgd":
                adv_batch = pgd_attack(
                    model, batch, loss_fn,
                    epsilon=adv_config["epsilon"],
                    alpha=adv_config["alpha"],
                    num_steps=adv_config["pgd_steps"],
                )
            else:
                adv_batch = fgsm_attack(model, batch, loss_fn, epsilon=adv_config["epsilon"])
            model.train()

            with autocast(enabled=scaler.is_enabled()):
                adv_out = model(adv_batch)
                adv_loss = loss_fn(adv_out["logits"], adv_batch["label"])

        # ── Combined loss ──────────────────────────────────────
        lam = adv_config.get("lambda", 0.5)
        total_batch_loss = loss_clean + lam * adv_loss

        # ── Backward ───────────────────────────────────────────
        optimizer.zero_grad()
        scaler.scale(total_batch_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # ── Metrics ────────────────────────────────────────────
        preds = out["logits"].argmax(dim=1)
        correct += (preds == batch["label"]).sum().item()
        total += batch["label"].size(0)
        total_loss += loss_clean.item()
        total_adv_loss += adv_loss.item()

        global_step = epoch * len(loader) + batch_idx
        if writer and batch_idx % log_interval == 0:
            writer.add_scalar("train/loss_clean", loss_clean.item(), global_step)
            writer.add_scalar("train/loss_adv", adv_loss.item(), global_step)
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
            if "alpha" in out:
                alpha = out["alpha"].mean(dim=0).detach().cpu()
                writer.add_scalars("train/modality_alpha", {
                    "spatial": alpha[0].item(),
                    "temporal": alpha[1].item(),
                    "freq": alpha[2].item(),
                }, global_step)

        if batch_idx % log_interval == 0:
            elapsed = time.time() - t0
            log.info(
                f"  Epoch {epoch} [{batch_idx}/{len(loader)}] "
                f"loss={loss_clean.item():.4f} adv={adv_loss.item():.4f} "
                f"acc={correct/max(total,1)*100:.1f}% "
                f"lr={scheduler.get_last_lr()[0]:.2e} "
                f"t={elapsed:.1f}s"
            )

    return {
        "loss": total_loss / len(loader),
        "adv_loss": total_adv_loss / len(loader),
        "acc": correct / max(total, 1),
    }


# ─────────────────────────────────────────────────────────────
# Validation pass
# ─────────────────────────────────────────────────────────────

def validate_epoch(
    model: nn.Module,
    loader,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    writer: Optional[SummaryWriter] = None,
) -> dict:
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in loader:
            batch = {
                "spatial": batch["spatial"].to(device),
                "freq":    batch["freq"].to(device),
                "label":   batch["label"].to(device),
            }
            with autocast():
                out = model(batch)
            loss = loss_fn(out["logits"], batch["label"])
            total_loss += loss.item()

            probs = F.softmax(out["logits"], dim=-1)[:, 1]
            preds = out["logits"].argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch["label"].cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    # Compute metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    acc  = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec  = recall_score(all_labels, all_preds, zero_division=0)
    f1   = f1_score(all_labels, all_preds, zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = 0.0

    metrics = {
        "loss": total_loss / max(len(loader), 1),
        "acc": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc,
    }

    if writer:
        for k, v in metrics.items():
            writer.add_scalar(f"val/{k}", v, epoch)

    log.info(
        f"  [VAL] Epoch {epoch} | "
        f"loss={metrics['loss']:.4f} acc={acc*100:.1f}% "
        f"prec={prec:.3f} rec={rec:.3f} f1={f1:.3f} AUC={auc:.4f}"
    )
    return metrics


# ─────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────

def train(config: dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    if device.type == "cuda":
        log.info(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ── Paths ──────────────────────────────────────────────────
    out_dir = Path(config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    # ── Data ───────────────────────────────────────────────────
    log.info("Building dataloaders...")
    train_loader = build_dataloader(
        config["manifest"], split="train",
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        num_frames=config["num_frames"],
        sampling=config.get("sampling", "motion"),
    )
    val_loader = build_dataloader(
        config["manifest"], split="val",
        batch_size=config["batch_size"] * 2,
        num_workers=config["num_workers"],
        num_frames=config["num_frames"],
        sampling="uniform",
    )

    # ── Model ──────────────────────────────────────────────────
    log.info("Building model...")
    model = build_model(config["model"]).to(device)
    log.info(f"  Parameters: {model.count_parameters():,}")

    # Class-weighted focal loss
    dataset = train_loader.dataset
    class_weights = dataset.class_weights().to(device)
    loss_fn = FocalLoss(gamma=config.get("focal_gamma", 2.0), weight=class_weights)

    # ── Optimiser ──────────────────────────────────────────────
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config.get("weight_decay", 1e-4),
        betas=(0.9, 0.999),
    )

    total_steps = len(train_loader) * config["epochs"]
    warmup_steps = len(train_loader) * config.get("warmup_epochs", 2)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)
    scaler = GradScaler(enabled=config.get("fp16", True))

    # ── Resume ─────────────────────────────────────────────────
    start_epoch = 0
    resume_path = config.get("resume_checkpoint")
    if resume_path and Path(resume_path).exists():
        start_epoch = load_checkpoint(model, optimizer, scheduler, resume_path)

    # ── TensorBoard ────────────────────────────────────────────
    writer = SummaryWriter(log_dir=str(out_dir / "tb_logs"))

    # ── Training ───────────────────────────────────────────────
    best_f1 = 0.0
    patience_counter = 0
    patience = config.get("patience", 10)

    adv_config = {
        "enabled": config.get("adversarial_training", True),
        "attack": config.get("adv_attack", "pgd"),
        "epsilon": config.get("adv_epsilon", 0.03),
        "alpha": config.get("adv_alpha", 0.007),
        "pgd_steps": config.get("pgd_steps", 7),
        "adv_every": config.get("adv_every", 4),  # apply every 4 batches
        "lambda": config.get("adv_lambda", 0.5),
    }

    log.info(f"Starting training for {config['epochs']} epochs")
    log.info(f"  Adversarial training: {adv_config['enabled']} ({adv_config['attack'].upper()})")
    log.info(f"  Mixed precision: {config.get('fp16', True)}")

    for epoch in range(start_epoch, config["epochs"]):
        log.info(f"\n{'='*60}")
        log.info(f"EPOCH {epoch+1}/{config['epochs']}")

        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler,
            loss_fn, scaler, device, adv_config, epoch, writer,
        )
        val_metrics = validate_epoch(model, val_loader, loss_fn, device, epoch, writer)

        # Save checkpoint
        ckpt_path = str(ckpt_dir / f"epoch_{epoch+1:03d}.pt")
        is_best = val_metrics["f1"] > best_f1
        if is_best:
            best_f1 = val_metrics["f1"]
            patience_counter = 0
        else:
            patience_counter += 1

        save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, ckpt_path, is_best)

        log.info(
            f"  Train loss: {train_metrics['loss']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f} | Best F1: {best_f1:.4f} | "
            f"Patience: {patience_counter}/{patience}"
        )

        if patience_counter >= patience:
            log.info("Early stopping triggered.")
            break

    writer.close()
    log.info(f"\nTraining complete. Best val F1: {best_f1:.4f}")
    log.info(f"Checkpoints saved to: {ckpt_dir}")


# ─────────────────────────────────────────────────────────────
# Default config
# ─────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "manifest": "data/manifest.json",
    "output_dir": "outputs/run_001",
    "batch_size": 4,
    "num_workers": 4,
    "num_frames": 16,
    "epochs": 50,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "warmup_epochs": 2,
    "patience": 10,
    "focal_gamma": 2.0,
    "fp16": True,
    "sampling": "motion",
    "adversarial_training": True,
    "adv_attack": "pgd",
    "adv_epsilon": 0.03,
    "adv_alpha": 0.007,
    "pgd_steps": 7,
    "adv_every": 4,
    "adv_lambda": 0.5,
    "resume_checkpoint": None,
    "model": {
        "embed_dim": 512,
        "num_classes": 2,
        "num_frames": 16,
        "spatial_depths": [2, 2, 6, 2],
        "fusion_heads": 8,
        "dropout": 0.1,
        "fusion_mode": "full",
    },
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, default=DEFAULT_CONFIG["manifest"])
    parser.add_argument("--output_dir", type=str, default=DEFAULT_CONFIG["output_dir"])
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--no_adv", action="store_true")
    parser.add_argument("--fusion_mode", type=str, default="full",
                        choices=["full", "no_freq", "no_temporal", "concat"])
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    config.update({
        "manifest": args.manifest,
        "output_dir": args.output_dir,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "adversarial_training": not args.no_adv,
        "resume_checkpoint": args.resume,
    })
    config["model"]["fusion_mode"] = args.fusion_mode

    train(config)
