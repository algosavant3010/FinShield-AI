"""
FinShield AI — DeepFake Detection System
========================================
inference_api.py: FastAPI inference server

Endpoints:
  POST /predict       — single video inference
  POST /predict_batch — batch video inference
  GET  /health        — service health check
  GET  /model_info    — architecture & version info

Run with:
  uvicorn inference_api:app --host 0.0.0.0 --port 8000 --workers 1

Docker:
  docker build -t finshield-api .
  docker run --gpus all -p 8000:8000 finshield-api
"""

from __future__ import annotations

import io
import os
import time
import logging
import tempfile
import base64
from pathlib import Path
from typing import Optional, List, Dict, Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).parent))
from model import TriFusionDF, build_model
from data_loader import (
    FaceDetector, uniform_sample, motion_keyframe_sample,
    compute_dct_map, build_val_augmentation, FACE_SIZE, NUM_FRAMES
)

# ─────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("FinShield.API")

MODEL_CONFIG = {
    "embed_dim": 512,
    "num_classes": 2,
    "num_frames": NUM_FRAMES,
    "spatial_depths": [2, 2, 6, 2],
    "fusion_heads": 8,
    "dropout": 0.0,          # No dropout at inference
    "fusion_mode": "full",
}

CHECKPOINT_PATH = os.getenv("FINSHIELD_CHECKPOINT", "outputs/run_001/checkpoints/best_model.pt")
CONFIDENCE_THRESHOLD = float(os.getenv("FINSHIELD_THRESHOLD", "0.5"))
MAX_VIDEO_SIZE_MB = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────
# Model singleton (loaded once at startup)
# ─────────────────────────────────────────────────────────────

class ModelManager:
    _model: Optional[TriFusionDF] = None
    _face_detector: Optional[FaceDetector] = None
    _augmentation = None
    _version: str = "1.0.0"

    @classmethod
    def load(cls, checkpoint_path: str) -> None:
        log.info(f"Loading model on device: {DEVICE}")
        cls._model = build_model(MODEL_CONFIG).to(DEVICE)
        if Path(checkpoint_path).exists():
            state = torch.load(checkpoint_path, map_location=DEVICE)
            cls._model.load_state_dict(state["model_state"])
            log.info(f"Checkpoint loaded: {checkpoint_path}")
        else:
            log.warning(f"Checkpoint not found: {checkpoint_path}. Using random weights.")
        cls._model.eval()
        cls._face_detector = FaceDetector()
        cls._augmentation = build_val_augmentation()
        log.info(f"Model ready. Parameters: {cls._model.count_parameters():,}")

    @classmethod
    def model(cls) -> TriFusionDF:
        if cls._model is None:
            raise RuntimeError("Model not loaded. Call ModelManager.load() first.")
        return cls._model

    @classmethod
    def preprocess_video(
        cls,
        video_bytes: bytes,
        num_frames: int = NUM_FRAMES,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Full preprocessing pipeline:
          video bytes → frames → face detection → augmentation → freq maps → tensors
        Returns None if video cannot be processed.
        """
        # Write to temp file (OpenCV requires file path)
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        try:
            cap = cv2.VideoCapture(tmp_path)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()
        finally:
            os.unlink(tmp_path)

        if not frames:
            return None

        # Frame sampling
        sampled = motion_keyframe_sample(frames, num_frames)

        # Face detection + crop
        spatial_tensors = []
        freq_tensors = []
        aug = cls._augmentation
        det = cls._face_detector

        for frame_bgr in sampled:
            face = det.crop_and_align(frame_bgr)
            if face is None:
                face = cv2.resize(frame_bgr, (FACE_SIZE, FACE_SIZE))

            # Spatial augmentation
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            aug_out = aug(image=face_rgb)
            spatial_tensors.append(aug_out["image"])   # (3, H, W)

            # Frequency map
            freq = compute_dct_map(face)
            freq_tensors.append(torch.from_numpy(freq))

        spatial = torch.stack(spatial_tensors, dim=0).unsqueeze(0)  # (1, T, 3, H, W)
        freq    = torch.stack(freq_tensors, dim=0).unsqueeze(0)     # (1, T, 3, H, W)
        return {"spatial": spatial.to(DEVICE), "freq": freq.to(DEVICE)}


# ─────────────────────────────────────────────────────────────
# Pydantic request / response schemas
# ─────────────────────────────────────────────────────────────

class PredictResponse(BaseModel):
    verdict:            str          = Field(..., description="'GENUINE' or 'FAKE'")
    fake_probability:   float        = Field(..., ge=0.0, le=1.0)
    confidence:         float        = Field(..., ge=0.0, le=1.0)
    modality_weights:   Dict[str, float]
    inference_time_ms:  float
    model_version:      str

class BatchPredictResponse(BaseModel):
    results: List[PredictResponse]
    total_time_ms: float

class HealthResponse(BaseModel):
    status: str
    device: str
    model_loaded: bool

class ModelInfoResponse(BaseModel):
    name:       str
    version:    str
    parameters: int
    embed_dim:  int
    num_frames: int
    fusion_mode: str
    device:     str


# ─────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="FinShield AI — DeepFake Detection API",
    description=(
        "Multimodal deepfake detection API using TriFusion-DF architecture. "
        "Detects deepfakes via spatial, temporal, and frequency analysis with "
        "cross-attention fusion."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    ModelManager.load(CHECKPOINT_PATH)


# ─────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        device=str(DEVICE),
        model_loaded=ModelManager._model is not None,
    )


@app.get("/model_info", response_model=ModelInfoResponse)
async def model_info():
    m = ModelManager.model()
    return ModelInfoResponse(
        name="TriFusion-DF",
        version=ModelManager._version,
        parameters=m.count_parameters(),
        embed_dim=MODEL_CONFIG["embed_dim"],
        num_frames=MODEL_CONFIG["num_frames"],
        fusion_mode=MODEL_CONFIG["fusion_mode"],
        device=str(DEVICE),
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(..., description="Video file (mp4, avi, mov)")):
    """
    Predicts whether the uploaded facial video is genuine or deepfaked.

    Returns:
      - verdict: 'GENUINE' | 'FAKE'
      - fake_probability: model's confidence that video is fake [0, 1]
      - modality_weights: per-branch attention weights (interpretable)
      - inference_time_ms: end-to-end latency
    """
    # Validate file size
    content = await file.read()
    size_mb = len(content) / 1e6
    if size_mb > MAX_VIDEO_SIZE_MB:
        raise HTTPException(413, f"File too large ({size_mb:.1f}MB). Max: {MAX_VIDEO_SIZE_MB}MB")

    if not file.filename or not any(
        file.filename.lower().endswith(ext) for ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    ):
        raise HTTPException(400, "Unsupported file type. Use .mp4, .avi, .mov, .mkv, or .webm")

    t_start = time.perf_counter()

    # Preprocess
    batch = ModelManager.preprocess_video(content)
    if batch is None:
        raise HTTPException(422, "Could not decode video or detect faces.")

    # Inference
    model = ModelManager.model()
    with torch.no_grad():
        out = model(batch)

    probs = F.softmax(out["logits"], dim=-1)[0]
    fake_prob = probs[1].item()
    verdict = "FAKE" if fake_prob >= CONFIDENCE_THRESHOLD else "GENUINE"
    confidence = max(fake_prob, 1 - fake_prob)

    # Modality weights
    alpha = out.get("alpha")
    if alpha is not None:
        alpha_vals = alpha[0].cpu().tolist()
        modality_weights = {
            "spatial": round(alpha_vals[0], 4),
            "temporal": round(alpha_vals[1], 4),
            "frequency": round(alpha_vals[2], 4),
        }
    else:
        modality_weights = {"spatial": 0.0, "temporal": 0.0, "frequency": 0.0}

    elapsed_ms = (time.perf_counter() - t_start) * 1000

    log.info(
        f"Prediction: {verdict} | fake_prob={fake_prob:.4f} | "
        f"alpha={modality_weights} | t={elapsed_ms:.1f}ms"
    )

    return PredictResponse(
        verdict=verdict,
        fake_probability=round(fake_prob, 6),
        confidence=round(confidence, 6),
        modality_weights=modality_weights,
        inference_time_ms=round(elapsed_ms, 2),
        model_version=ModelManager._version,
    )


@app.post("/predict_batch", response_model=BatchPredictResponse)
async def predict_batch(files: List[UploadFile] = File(...)):
    """Batch prediction for up to 10 videos simultaneously."""
    if len(files) > 10:
        raise HTTPException(400, "Max 10 files per batch request.")

    t_total = time.perf_counter()
    results = []

    for f in files:
        content = await f.read()
        batch = ModelManager.preprocess_video(content)
        if batch is None:
            results.append(PredictResponse(
                verdict="ERROR",
                fake_probability=0.0,
                confidence=0.0,
                modality_weights={},
                inference_time_ms=0.0,
                model_version=ModelManager._version,
            ))
            continue

        t_start = time.perf_counter()
        with torch.no_grad():
            out = ModelManager.model()(batch)
        probs = F.softmax(out["logits"], dim=-1)[0]
        fake_prob = probs[1].item()

        results.append(PredictResponse(
            verdict="FAKE" if fake_prob >= CONFIDENCE_THRESHOLD else "GENUINE",
            fake_probability=round(fake_prob, 6),
            confidence=round(max(fake_prob, 1 - fake_prob), 6),
            modality_weights={},
            inference_time_ms=round((time.perf_counter() - t_start) * 1000, 2),
            model_version=ModelManager._version,
        ))

    return BatchPredictResponse(
        results=results,
        total_time_ms=round((time.perf_counter() - t_total) * 1000, 2),
    )


# ─────────────────────────────────────────────────────────────
# ONNX export helper
# ─────────────────────────────────────────────────────────────

def export_for_deployment(checkpoint_path: str, export_dir: str = "exports") -> None:
    """
    Exports the model to both ONNX and TorchScript formats.
    Run once after training; use exported model in production for 2-5× speedup.
    """
    from model import export_onnx, export_torchscript
    out = Path(export_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = build_model(MODEL_CONFIG)
    if Path(checkpoint_path).exists():
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state["model_state"])
    model.eval()

    export_onnx(model, str(out / "finshield_df.onnx"), NUM_FRAMES)
    export_torchscript(model, str(out / "finshield_df.pt"), NUM_FRAMES)
    log.info(f"Exported to {export_dir}/")


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "inference_api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
        workers=1,   # GPU models: single worker avoids memory duplication
    )
