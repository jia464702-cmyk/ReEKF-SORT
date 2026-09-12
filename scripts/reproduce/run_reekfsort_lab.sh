#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

ACTION="${1:-status}"
if [[ $# -gt 0 ]]; then
  shift
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="$PYTHON_BIN"
elif command -v python3.11 >/dev/null 2>&1; then
  PYTHON_BIN="python3.11"
else
  PYTHON_BIN="python3"
fi
VENV_DIR="${REEKF_VENV_DIR:-$ROOT/.venv-reekf}"
TORCH_VERSION="${REEKF_TORCH_VERSION:-2.5.1}"
TORCHVISION_VERSION="${REEKF_TORCHVISION_VERSION:-0.20.1}"
TORCH_INDEX_URL="${REEKF_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 2
  fi
}

setup_environment() {
  require_command "$PYTHON_BIN"
  require_command nvidia-smi
  require_command gcc

  "$PYTHON_BIN" -c 'import sys; assert (3, 10) <= sys.version_info[:2] < (3, 13), "Python 3.10-3.12 is required"'

  echo "NVIDIA driver and GPU:"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  fi
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  if [[ ! -f "$VENV_DIR/.reekf_setup_complete" || "${REEKF_REINSTALL:-0}" == "1" ]]; then
    # yolox 0.3.0 has legacy metadata rejected by pip 24.1+, so keep the
    # environment on the last compatible pip while installing it.
    python -m pip install "pip==24.0" "setuptools>=69,<76" wheel
    python -m pip install \
      "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" \
      --index-url "$TORCH_INDEX_URL"
    python -m pip install -r requirements/reekfsort_eval.txt
    python -m pip install -e . --no-deps
    python -m pip install "yolox==0.3.0" --no-deps "pytest>=8,<10"
    touch "$VENV_DIR/.reekf_setup_complete"
  else
    echo "Reusing prepared environment: $VENV_DIR (set REEKF_REINSTALL=1 to reinstall)"
  fi
  python -m pip check || true
  python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable. Check the NVIDIA driver and PyTorch wheel.")
print("GPU:", torch.cuda.get_device_name(0))
assert torch.cuda.get_device_capability(0)[0] >= 8, "Expected an Ampere-class GPU such as RTX 3090"
PY
}

show_server_status() {
  require_command nvidia-smi
  echo "Server load:"
  uptime || true
  echo
  if command -v free >/dev/null 2>&1; then
    echo "Host memory:"
    free -h
    echo
  fi
  echo "Current NVIDIA GPU usage and processes:"
  nvidia-smi
  echo
  if command -v squeue >/dev/null 2>&1; then
    echo "Slurm queue:"
    squeue || true
    echo
  fi
  echo "No experiment was started. Recheck this status immediately before every manual run."
  if [[ -f "$ROOT/runs/reekfsort_pipeline/state.json" ]]; then
    echo "Pipeline state: $ROOT/runs/reekfsort_pipeline/state.json"
  fi
}

if [[ "$ACTION" == "status" ]]; then
  show_server_status
  exit 0
fi

if [[ "$ACTION" == "commands" ]]; then
  echo "Manual experiment commands are documented in:"
  echo "  docs/trackers/reekfsort_server_setup.md"
  echo "Start by checking the shared GPU:"
  echo "  bash scripts/reproduce/run_reekfsort_lab.sh status"
  exit 0
fi

if [[ "$ACTION" == "all" || "$ACTION" == "experiments" ]]; then
  echo "The automatic full-batch action is disabled for shared-server safety." >&2
  echo "Check GPU status, then run one dataset or one experiment group at a time." >&2
  echo "See docs/trackers/reekfsort_server_setup.md for exact commands." >&2
  exit 2
fi

setup_environment
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"
export PYTHONUNBUFFERED=1

if [[ "$ACTION" == "setup" ]]; then
  python scripts/reproduce/reekfsort_lab.py doctor "$@"
else
  python scripts/reproduce/reekfsort_lab.py "$ACTION" "$@"
fi

