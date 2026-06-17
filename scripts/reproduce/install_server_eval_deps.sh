#!/usr/bin/env bash
set -euo pipefail

PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

if command -v conda >/dev/null 2>&1; then
  conda create -n reekfsort "python=${PYTHON_VERSION}" -y
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate reekfsort
else
  echo "conda not found. Please create/activate a Python ${PYTHON_VERSION} environment first." >&2
fi

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision --index-url "${TORCH_INDEX_URL}"
python -m pip install -e .
python -m pip install \
  opencv-python ultralytics loguru thop ninja tensorboard pycocotools \
  onnx onnxruntime onnx-simplifier ftfy yacs filterpy lapx gdown huggingface_hub
python -m pip install yolox==0.3.0 --no-deps

python - <<'PY'
import cv2
import torch
import ultralytics
from boxmot.trackers.bbox.reekfsort import ReEkfSort

print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("cv2", cv2.__version__)
print("ultralytics", ultralytics.__version__)
print("ReEkfSort import ok:", ReEkfSort.__name__)
PY
