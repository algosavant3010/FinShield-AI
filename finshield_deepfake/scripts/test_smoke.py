"""
scripts/test_smoke.py
Quick smoke test — verifies the entire pipeline works WITHOUT needing
datasets, GPU, or a trained checkpoint.
Runs on CPU in ~30-60 seconds.

Usage:
    python scripts/test_smoke.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import numpy as np


def test_data_loader():
    print("\n[1/4] Testing data_loader.py ...")
    from data_loader import (
        FaceDetector, uniform_sample, motion_keyframe_sample,
        compute_dct_map, compute_fft_map, build_val_augmentation
    )
    # Face detector
    det = FaceDetector()
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    det.detect(blank)  # No crash on blank frame
    print("      FaceDetector: OK")

    # Sampling strategies
    frames = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(32)]
    uni = uniform_sample(frames, 16)
    mot = motion_keyframe_sample(frames, 16)
    assert len(uni) == 16 and len(mot) == 16
    print("      Frame sampling (uniform + motion): OK")

    # Frequency maps
    face = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    dct = compute_dct_map(face)
    fft = compute_fft_map(face)
    assert dct.shape == (3, 224, 224)
    assert fft.shape == (3, 224, 224)
    print("      DCT / FFT maps: OK")

    # Augmentation
    aug = build_val_augmentation()
    rgb = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    out = aug(image=rgb)
    assert out["image"].shape == (3, 224, 224)
    print("      Augmentation pipeline: OK")
    print("  [PASS] data_loader.py")


def test_fusion():
    print("\n[2/4] Testing fusion.py ...")
    from fusion import TriModalFusion, TwoModalFusion, SimpleConcatFusion

    B, d = 2, 256
    F_s = torch.randn(B, d)
    F_t = torch.randn(B, d)
    F_f = torch.randn(B, d)

    # Full fusion
    tri = TriModalFusion(dim=d, num_classes=2, heads=4)
    out = tri(F_s, F_t, F_f)
    assert out["logits"].shape == (B, 2)
    assert out["alpha"].shape  == (B, 3)
    assert abs(out["alpha"][0].sum().item() - 1.0) < 1e-5, "Alpha must sum to 1"
    print("      TriModalFusion: OK")

    two = TwoModalFusion(dim=d, num_classes=2, heads=4)
    out2 = two(F_s, F_t)
    assert out2["logits"].shape == (B, 2)
    print("      TwoModalFusion: OK")

    cat = SimpleConcatFusion(dim=d, num_classes=2)
    out3 = cat(F_s, F_t, F_f)
    assert out3["logits"].shape == (B, 2)
    print("      SimpleConcatFusion: OK")
    print("  [PASS] fusion.py")


def test_model():
    print("\n[3/4] Testing model.py ...")
    from model import TriFusionDF

    # Small config for CPU test
    model = TriFusionDF(
        embed_dim=128,
        num_frames=2,
        spatial_depths=[1, 1, 1, 1],
        fusion_heads=4,
        dropout=0.0,
    )
    n_params = model.count_parameters()
    print(f"      Parameters: {n_params:,}")

    dummy = {
        "spatial": torch.randn(1, 2, 3, 224, 224),
        "freq":    torch.randn(1, 2, 3, 224, 224),
    }
    out = model(dummy)
    probs = F.softmax(out["logits"], dim=-1)[0]
    assert out["logits"].shape == (1, 2)
    assert "alpha" in out

    print(f"      Logits: {out['logits'][0].tolist()}")
    print(f"      P(real)={probs[0]:.3f}  P(fake)={probs[1]:.3f}")
    print(f"      Modality α: spatial={out['alpha'][0,0]:.3f}  temporal={out['alpha'][0,1]:.3f}  freq={out['alpha'][0,2]:.3f}")

    # Test ablation modes
    for mode in ["no_freq", "no_temporal", "concat"]:
        m = TriFusionDF(embed_dim=128, num_frames=2, spatial_depths=[1,1,1,1], fusion_heads=4, fusion_mode=mode)
        o = m(dummy)
        assert o["logits"].shape == (1, 2), f"Mode {mode} failed"
        print(f"      Ablation mode '{mode}': OK")

    print("  [PASS] model.py")


def test_train_utils():
    print("\n[4/4] Testing train.py utilities ...")
    from train import FocalLoss, fgsm_attack, pgd_attack
    from model import TriFusionDF

    model = TriFusionDF(embed_dim=128, num_frames=2, spatial_depths=[1,1,1,1], fusion_heads=4)
    loss_fn = FocalLoss(gamma=2.0)
    dummy = {
        "spatial": torch.randn(1, 2, 3, 224, 224),
        "freq":    torch.randn(1, 2, 3, 224, 224),
        "label":   torch.tensor([1]),
    }

    # Focal loss
    out = model(dummy)
    loss = loss_fn(out["logits"], dummy["label"])
    assert loss.item() > 0
    print(f"      Focal loss: {loss.item():.4f}  OK")

    # FGSM
    adv = fgsm_attack(model, dummy, loss_fn, epsilon=0.03)
    assert adv["spatial"].shape == dummy["spatial"].shape
    diff = (adv["spatial"] - dummy["spatial"]).abs().max().item()
    assert diff <= 0.031
    print(f"      FGSM attack: max perturbation={diff:.4f}  OK")

    # PGD
    adv2 = pgd_attack(model, dummy, loss_fn, epsilon=0.03, alpha=0.007, num_steps=3)
    assert adv2["spatial"].shape == dummy["spatial"].shape
    print(f"      PGD attack (3 steps): OK")

    print("  [PASS] train.py")


if __name__ == "__main__":
    print("=" * 55)
    print("  FinShield AI — Smoke Test (CPU, no datasets needed)")
    print("=" * 55)

    try:
        test_data_loader()
        test_fusion()
        test_model()
        test_train_utils()
        print("\n" + "=" * 55)
        print("  ALL TESTS PASSED ✓")
        print("  Your environment is ready for training.")
        print("=" * 55)
    except Exception as e:
        print(f"\n  FAILED: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)
