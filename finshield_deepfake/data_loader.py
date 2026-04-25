"""
FinShield AI — DeepFake Detection System
========================================
data_loader.py: Video ingestion, face detection, frame sampling,
                augmentation pipeline, and PyTorch DataLoader factory.

Architecture decisions
----------------------
* RetinaFace (lightweight ONNX variant) for sub-40ms face detection per frame.
* Two complementary sampling strategies:
    - Uniform sampling   : baseline coverage
    - Motion-keyframe    : captures discontinuities adversaries exploit
* Frequency maps (DCT / FFT magnitude) computed here so they can be
  cached to disk, avoiding redundant GPU work during training.
* Augmentation mimics real-world degradations: WhatsApp-style compression,
  additive noise, motion blur, frame-dropping, and colour jitter.
"""

from __future__ import annotations

import os
import cv2
import math
import random
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Callable

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
FACE_SIZE = 224          # Spatial / frequency branch input size
FREQ_SIZE = 224          # Frequency branch map size
NUM_FRAMES = 16          # Frames sampled per clip
DCT_CHANNELS = 3         # Chrominance + luminance DCT maps stacked
CLIP_DURATION = 4.0      # Max seconds per clip (for motion sampling)
FAKE_LABEL = 1
REAL_LABEL = 0


# ─────────────────────────────────────────────────────────────
# Face Detector (lightweight wrapper — swap for ONNX RetinaFace)
# ─────────────────────────────────────────────────────────────
class FaceDetector:
    """
    Wraps OpenCV Haar cascade as a CPU-only fallback.
    In production, replace `detect` with a RetinaFace / MTCNN ONNX call.
    The interface is kept identical so swapping is a one-liner.
    """

    def __init__(self, min_confidence: float = 0.85):
        self.min_confidence = min_confidence
        # Haar cascade as fast fallback; production uses RetinaFace ONNX
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(cascade_path)

    def detect(self, frame_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Returns (x, y, w, h) of the largest detected face, or None.
        Alignment is done by the caller via `align_face`.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        if len(faces) == 0:
            return None
        # Pick the largest face by area
        largest = max(faces, key=lambda b: b[2] * b[3])
        return tuple(largest)

    def crop_and_align(
        self, frame_bgr: np.ndarray, margin: float = 0.25
    ) -> Optional[np.ndarray]:
        """
        Detects face, adds a margin, resizes to FACE_SIZE×FACE_SIZE.
        Returns None if no face found.
        """
        bbox = self.detect(frame_bgr)
        if bbox is None:
            return None
        x, y, w, h = bbox
        H, W = frame_bgr.shape[:2]
        mx, my = int(w * margin), int(h * margin)
        x1 = max(0, x - mx)
        y1 = max(0, y - my)
        x2 = min(W, x + w + mx)
        y2 = min(H, y + h + my)
        face = frame_bgr[y1:y2, x1:x2]
        face = cv2.resize(face, (FACE_SIZE, FACE_SIZE))
        return face


# ─────────────────────────────────────────────────────────────
# Frame Sampling Strategies
# ─────────────────────────────────────────────────────────────

def uniform_sample(frames: List[np.ndarray], n: int) -> List[np.ndarray]:
    """Evenly-spaced temporal coverage — baseline."""
    if len(frames) <= n:
        # Pad with last frame
        pad = [frames[-1]] * (n - len(frames))
        return frames + pad
    indices = np.linspace(0, len(frames) - 1, n, dtype=int)
    return [frames[i] for i in indices]


def motion_keyframe_sample(frames: List[np.ndarray], n: int) -> List[np.ndarray]:
    """
    Selects frames with highest inter-frame optical-flow magnitude.
    Rationale: deepfakes often show abrupt artefact spikes at keyframes;
    this sampler biases toward those regions.
    """
    if len(frames) <= n:
        pad = [frames[-1]] * (n - len(frames))
        return frames + pad

    scores = [0.0]  # first frame has no predecessor
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    for i in range(1, len(frames)):
        curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(curr_gray, prev_gray)
        scores.append(float(diff.mean()))
        prev_gray = curr_gray

    # Blend motion score with uniform prior to ensure coverage
    uniform_prior = np.ones(len(frames))
    scores = np.array(scores) + 0.3 * uniform_prior
    scores /= scores.sum()

    chosen = np.random.choice(len(frames), size=n, replace=False, p=scores)
    chosen.sort()
    return [frames[i] for i in chosen]


# ─────────────────────────────────────────────────────────────
# Frequency Map Extraction
# ─────────────────────────────────────────────────────────────

def compute_dct_map(face_bgr: np.ndarray) -> np.ndarray:
    """
    Computes 2D DCT on each YCbCr channel, log-normalises, stacks to (3, H, W).
    DCT highlights periodic GAN artefacts invisible in the spatial domain.
    """
    ycbcr = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    channels = []
    for c in range(3):
        ch = ycbcr[:, :, c]
        ch_dct = cv2.dct(ch)
        # Log compression + normalise to [0, 1]
        ch_dct = np.log(np.abs(ch_dct) + 1e-8)
        ch_dct = (ch_dct - ch_dct.min()) / (ch_dct.max() - ch_dct.min() + 1e-8)
        channels.append(ch_dct)
    dct_map = np.stack(channels, axis=0)  # (3, H, W)
    return dct_map.astype(np.float32)


def compute_fft_map(face_bgr: np.ndarray) -> np.ndarray:
    """
    FFT magnitude spectrum per channel — complementary to DCT.
    Captures global frequency statistics (GAN checkerboard patterns, etc.)
    """
    ycbcr = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    channels = []
    for c in range(3):
        ch = ycbcr[:, :, c]
        f = np.fft.fft2(ch)
        fshift = np.fft.fftshift(f)
        magnitude = np.log(np.abs(fshift) + 1e-8)
        magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
        channels.append(magnitude)
    fft_map = np.stack(channels, axis=0)  # (3, H, W)
    return fft_map.astype(np.float32)


# ─────────────────────────────────────────────────────────────
# Augmentation Pipelines (Albumentations)
# ─────────────────────────────────────────────────────────────

def build_train_augmentation() -> A.Compose:
    """
    Augmentation rationale:
    - ImageCompression: mimics WhatsApp/social-media degradation (quality=40-90)
    - GaussNoise / ISONoise: real-world camera noise
    - MotionBlur / GaussianBlur: hand-shake, out-of-focus artefacts
    - HorizontalFlip: symmetry invariance
    - RandomBrightnessContrast / HueSaturationValue: lighting variation
    - CoarseDropout: simulates partial occlusion / packet loss artefacts
    - ShiftScaleRotate: pose variation
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5),
        A.OneOf([
            A.MotionBlur(blur_limit=5),
            A.GaussianBlur(blur_limit=3),
            A.MedianBlur(blur_limit=3),
        ], p=0.3),
        A.ImageCompression(quality_lower=40, quality_upper=90, p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0)),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5)),
        ], p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
        A.CoarseDropout(max_holes=4, max_height=20, max_width=20, p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def build_val_augmentation() -> A.Compose:
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


# ─────────────────────────────────────────────────────────────
# Core Dataset
# ─────────────────────────────────────────────────────────────

class DeepfakeVideoDataset(Dataset):
    """
    Loads video clips from a JSON manifest.

    Manifest format (one entry per video):
    {
        "path": "/data/ff++/fake/video_001.mp4",
        "label": 1,            # 1=fake, 0=real
        "dataset": "FF++",     # provenance tag for cross-dataset eval
        "split": "train"
    }

    Each __getitem__ returns:
        spatial_frames   : Tensor (T, 3, 224, 224)  — normalised RGB crops
        freq_maps        : Tensor (T, 3, 224, 224)  — DCT/FFT maps
        label            : int
        meta             : dict  (path, dataset, split)
    """

    def __init__(
        self,
        manifest_path: str,
        split: str = "train",
        num_frames: int = NUM_FRAMES,
        sampling: str = "motion",          # "uniform" | "motion"
        freq_type: str = "dct",            # "dct" | "fft" | "both"
        augmentation: Optional[A.Compose] = None,
        face_detector: Optional[FaceDetector] = None,
        cache_dir: Optional[str] = None,   # Pre-extracted face cache
    ):
        super().__init__()
        self.split = split
        self.num_frames = num_frames
        self.sampling_fn = (
            motion_keyframe_sample if sampling == "motion" else uniform_sample
        )
        self.freq_type = freq_type
        self.augmentation = augmentation or (
            build_train_augmentation() if split == "train" else build_val_augmentation()
        )
        self.face_detector = face_detector or FaceDetector()
        self.cache_dir = Path(cache_dir) if cache_dir else None

        with open(manifest_path) as f:
            all_entries = json.load(f)

        self.entries = [e for e in all_entries if e.get("split", split) == split]
        print(f"[DataLoader] {split}: {len(self.entries)} videos loaded")
        print(f"  Fake: {sum(1 for e in self.entries if e['label'] == FAKE_LABEL)}")
        print(f"  Real: {sum(1 for e in self.entries if e['label'] == REAL_LABEL)}")

    # ── Internal helpers ────────────────────────────────────────

    def _load_frames(self, video_path: str) -> List[np.ndarray]:
        """Decodes all frames from a video file (BGR)."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        if not frames:
            raise RuntimeError(f"Could not decode any frame from: {video_path}")
        return frames

    def _extract_face_crops(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Detects & aligns faces; falls back to centre-crop if detection fails."""
        crops = []
        H, W = frames[0].shape[:2]
        fallback = lambda f: cv2.resize(
            f[H // 8: 7 * H // 8, W // 8: 7 * W // 8], (FACE_SIZE, FACE_SIZE)
        )
        for frame in frames:
            crop = self.face_detector.crop_and_align(frame)
            crops.append(crop if crop is not None else fallback(frame))
        return crops

    def _compute_freq(self, face_bgr: np.ndarray) -> np.ndarray:
        """Returns frequency map (3, H, W) according to self.freq_type."""
        face_resized = cv2.resize(face_bgr, (FREQ_SIZE, FREQ_SIZE))
        if self.freq_type == "fft":
            return compute_fft_map(face_resized)
        elif self.freq_type == "both":
            dct = compute_dct_map(face_resized)
            fft = compute_fft_map(face_resized)
            return (dct + fft) / 2.0
        else:  # default: dct
            return compute_dct_map(face_resized)

    # ── Dataset interface ────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict:
        entry = self.entries[idx]
        video_path = entry["path"]
        label = int(entry["label"])

        # Load & sample frames
        try:
            all_frames = self._load_frames(video_path)
        except RuntimeError:
            # Return zero tensors on bad video — handled downstream
            return self._dummy_sample(label, entry)

        sampled = self.sampling_fn(all_frames, self.num_frames)
        face_crops = self._extract_face_crops(sampled)

        spatial_frames, freq_maps = [], []
        for crop_bgr in face_crops:
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            # Albumentations augmentation
            aug_out = self.augmentation(image=crop_rgb)
            spatial_frames.append(aug_out["image"])   # (3, H, W) tensor

            # Frequency map — no pixel augmentation, only normalise
            freq = self._compute_freq(crop_bgr)       # (3, H, W) numpy
            freq_maps.append(torch.from_numpy(freq))

        spatial = torch.stack(spatial_frames, dim=0)  # (T, 3, H, W)
        freq = torch.stack(freq_maps, dim=0)          # (T, 3, H, W)

        return {
            "spatial": spatial,
            "freq": freq,
            "label": torch.tensor(label, dtype=torch.long),
            "meta": {
                "path": video_path,
                "dataset": entry.get("dataset", "unknown"),
                "split": self.split,
            },
        }

    def _dummy_sample(self, label: int, entry: dict) -> Dict:
        spatial = torch.zeros(self.num_frames, 3, FACE_SIZE, FACE_SIZE)
        freq = torch.zeros(self.num_frames, 3, FREQ_SIZE, FREQ_SIZE)
        return {
            "spatial": spatial,
            "freq": freq,
            "label": torch.tensor(label, dtype=torch.long),
            "meta": entry,
        }

    # ── Class weights for imbalance ──────────────────────────────
    def class_weights(self) -> torch.Tensor:
        labels = [e["label"] for e in self.entries]
        n_real = labels.count(REAL_LABEL)
        n_fake = labels.count(FAKE_LABEL)
        total = len(labels)
        w_real = total / (2.0 * max(n_real, 1))
        w_fake = total / (2.0 * max(n_fake, 1))
        return torch.tensor([w_real, w_fake], dtype=torch.float)

    def sample_weights(self) -> List[float]:
        cw = self.class_weights()
        return [float(cw[e["label"]]) for e in self.entries]


# ─────────────────────────────────────────────────────────────
# DataLoader Factory
# ─────────────────────────────────────────────────────────────

def build_dataloader(
    manifest_path: str,
    split: str,
    batch_size: int = 4,
    num_workers: int = 4,
    num_frames: int = NUM_FRAMES,
    sampling: str = "motion",
    freq_type: str = "dct",
    cache_dir: Optional[str] = None,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Factory that returns a DataLoader with:
    - WeightedRandomSampler (train) or SequentialSampler (val/test)
    - Pinned memory for fast GPU transfer
    """
    dataset = DeepfakeVideoDataset(
        manifest_path=manifest_path,
        split=split,
        num_frames=num_frames,
        sampling=sampling,
        freq_type=freq_type,
        cache_dir=cache_dir,
    )

    if split == "train":
        weights = dataset.sample_weights()
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
        )
        shuffle = False
    else:
        sampler = None
        shuffle = False

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=(split == "train"),
        collate_fn=default_collate_fn,
    )
    return loader


def default_collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate: stacks tensors, keeps meta as list of dicts."""
    return {
        "spatial": torch.stack([b["spatial"] for b in batch]),   # (B, T, 3, H, W)
        "freq": torch.stack([b["freq"] for b in batch]),         # (B, T, 3, H, W)
        "label": torch.stack([b["label"] for b in batch]),       # (B,)
        "meta": [b["meta"] for b in batch],
    }


# ─────────────────────────────────────────────────────────────
# Manifest Generator (utility for dataset preparation)
# ─────────────────────────────────────────────────────────────

def generate_manifest(
    dataset_roots: Dict[str, Dict[str, str]],
    output_path: str,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> None:
    """
    Walks dataset_roots and writes a JSON manifest.

    dataset_roots format:
    {
        "FF++": {"real": "/data/ff++/real", "fake": "/data/ff++/fake"},
        "Celeb-DF": {"real": "/data/celebdf/real", "fake": "/data/celebdf/fake"},
        "DFDC": {"real": "/data/dfdc/real", "fake": "/data/dfdc/fake"},
    }
    """
    rng = random.Random(seed)
    entries = []
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    for dataset_name, paths in dataset_roots.items():
        for label_str, folder in paths.items():
            label = FAKE_LABEL if label_str == "fake" else REAL_LABEL
            root = Path(folder)
            if not root.exists():
                print(f"  WARNING: {root} not found, skipping.")
                continue
            videos = [p for p in root.rglob("*") if p.suffix.lower() in video_exts]
            rng.shuffle(videos)

            n = len(videos)
            n_test = max(1, int(n * test_ratio))
            n_val = max(1, int(n * val_ratio))
            n_train = n - n_test - n_val

            for i, v in enumerate(videos):
                if i < n_train:
                    split = "train"
                elif i < n_train + n_val:
                    split = "val"
                else:
                    split = "test"
                entries.append({
                    "path": str(v),
                    "label": label,
                    "dataset": dataset_name,
                    "split": split,
                })

    with open(output_path, "w") as f:
        json.dump(entries, f, indent=2)

    print(f"[Manifest] {len(entries)} entries written to {output_path}")


# ─────────────────────────────────────────────────────────────
# Quick sanity-check (run directly)
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile, json

    # Create a tiny dummy manifest pointing at a single webcam capture
    dummy = [
        {"path": "/dev/video0", "label": 0, "dataset": "webcam", "split": "train"}
    ]
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(dummy, f)
        manifest = f.name

    print("FaceDetector smoke test:")
    det = FaceDetector()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = det.detect(frame)
    print(f"  Detection on blank frame: {result}")

    print("DCT map smoke test:")
    face = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    dct = compute_dct_map(face)
    print(f"  DCT map shape: {dct.shape}, range: [{dct.min():.3f}, {dct.max():.3f}]")
    print("data_loader.py OK")
