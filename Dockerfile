FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY finshield_deepfake/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY finshield_deepfake/ ./finshield_deepfake/

# Expose API port
EXPOSE 8000

# Default: start the inference API
ENV FINSHIELD_CHECKPOINT="/app/checkpoints/best_model.pt"
CMD ["uvicorn", "finshield_deepfake.inference_api:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
