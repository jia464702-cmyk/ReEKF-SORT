#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DIST_DIR="${ROOT_DIR}/dist"
PACKAGE_ROOT="reekfsort_open_source_release"
ARCHIVE="${DIST_DIR}/${PACKAGE_ROOT}.tar.gz"
FILELIST="${DIST_DIR}/reekfsort_open_source_filelist.txt"
CHECKSUMS="${DIST_DIR}/reekfsort_open_source_checksums.sha256"
STAGING_DIR="$(mktemp -d /tmp/reekfsort_package.XXXXXX)"
SCAN_DIR="$(mktemp -d /tmp/reekfsort_package_scan.XXXXXX)"

cleanup() {
  rm -rf "${STAGING_DIR}" "${SCAN_DIR}"
}
trap cleanup EXIT

mkdir -p "${DIST_DIR}"
rm -f "${ARCHIVE}" "${FILELIST}" "${CHECKSUMS}"

cd "${ROOT_DIR}"

STAGE_ROOT="${STAGING_DIR}/${PACKAGE_ROOT}"
mkdir -p "${STAGE_ROOT}"

tar \
  --exclude="./dist" \
  --exclude="./models" \
  --exclude="./downloads" \
  --exclude="./runs" \
  --exclude="./outputs" \
  --exclude="./release_audit_report*.md" \
  --exclude="./boxmot/engine/eval/trackeval/data" \
  --exclude="./assets/MOT17-mini" \
  --exclude="./assets/mmot-mini" \
  --exclude="./assets/reid-mini" \
  --exclude="./.git" \
  --exclude="./.venv" \
  --exclude="./.pytest_cache" \
  --exclude="./.ruff_cache" \
  --exclude="./.mypy_cache" \
  --exclude="./.ipynb_checkpoints" \
  --exclude="*/.ipynb_checkpoints" \
  --exclude="./.DS_Store" \
  --exclude="*/.DS_Store" \
  --exclude="*/__pycache__" \
  --exclude="*.pyc" \
  --exclude="*.npy" \
  --exclude="*.parquet" \
  --exclude="*.pt" \
  --exclude="*.pth" \
  --exclude="*.onnx" \
  --exclude="*.engine" \
  --exclude="*.tar" \
  --exclude="*.tar.gz" \
  --exclude="*.tgz" \
  --exclude="*.zip" \
  --exclude="*.7z" \
  --exclude="*.rar" \
  --exclude="*.mp4" \
  --exclude="*.avi" \
  --exclude="*.tmp" \
  --exclude="*.part" \
  --exclude="*.bak" \
  --exclude="*.swp" \
  --exclude="*~" \
  -cf - \
  . | tar -xf - -C "${STAGE_ROOT}"

python - "${STAGE_ROOT}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for notebook in root.rglob("*.ipynb"):
    data = json.loads(notebook.read_text(encoding="utf-8"))
    changed = False
    for cell in data.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("outputs") != []:
            cell["outputs"] = []
            changed = True
        if cell.get("execution_count") is not None:
            cell["execution_count"] = None
            changed = True
    if changed:
        notebook.write_text(
            json.dumps(data, ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )
PY

tar \
  --sort=name \
  --mtime="UTC 2026-06-17" \
  --owner=0 \
  --group=0 \
  --numeric-owner \
  -C "${STAGING_DIR}" \
  -czf "${ARCHIVE}" \
  "${PACKAGE_ROOT}"

tar -tzf "${ARCHIVE}" | LC_ALL=C sort > "${FILELIST}"

tar -xzf "${ARCHIVE}" -C "${SCAN_DIR}"

USER_ROOT="/""Users/"
HOME_ROOT="/""home/"
DATA_USERS_ROOT="/""data/users/"
DATA3_ROOT="/""data3/"
LEAK_PATTERN="${USER_ROOT}[^[:space:]]+|${HOME_ROOT}[^[:space:]]+|${DATA_USERS_ROOT}[^[:space:]]+|${DATA3_ROOT}[^[:space:]]+"
if grep -RInE "${LEAK_PATTERN}" "${SCAN_DIR}/${PACKAGE_ROOT}"; then
  echo "ERROR: absolute personal/server path leak found in release archive." >&2
  exit 1
fi

if find "${SCAN_DIR}/${PACKAGE_ROOT}" -type f \( \
    -name "*.pyc" -o \
    -name "*.npy" -o \
    -name "*.parquet" -o \
    -name "*.pt" -o \
    -name "*.pth" -o \
    -name "*.onnx" -o \
    -name "*.engine" -o \
    -name "*.tar" -o \
    -name "*.tar.gz" -o \
    -name "*.tgz" -o \
    -name "*.zip" -o \
    -name "*.7z" -o \
    -name "*.rar" -o \
    -name "*.mp4" -o \
    -name "*.avi" -o \
    -name "*.tmp" -o \
    -name "*.part" -o \
    -name "*.bak" -o \
    -name "*.swp" \
  \) | grep -q .; then
  echo "ERROR: forbidden generated artifact found in release archive." >&2
  find "${SCAN_DIR}/${PACKAGE_ROOT}" -type f \( \
    -name "*.pyc" -o \
    -name "*.npy" -o \
    -name "*.parquet" -o \
    -name "*.pt" -o \
    -name "*.pth" -o \
    -name "*.onnx" -o \
    -name "*.engine" -o \
    -name "*.tar" -o \
    -name "*.tar.gz" -o \
    -name "*.tgz" -o \
    -name "*.zip" -o \
    -name "*.7z" -o \
    -name "*.rar" -o \
    -name "*.mp4" -o \
    -name "*.avi" -o \
    -name "*.tmp" -o \
    -name "*.part" -o \
    -name "*.bak" -o \
    -name "*.swp" \
  \)
  exit 1
fi

if find "${SCAN_DIR}/${PACKAGE_ROOT}" -type f -size +50M | grep -q .; then
  echo "ERROR: unexpectedly large file found in release archive." >&2
  find "${SCAN_DIR}/${PACKAGE_ROOT}" -type f -size +50M -printf "%p %s bytes\n"
  exit 1
fi

(
  cd "${DIST_DIR}"
  sha256sum "$(basename "${ARCHIVE}")" "$(basename "${FILELIST}")" > "$(basename "${CHECKSUMS}")"
)

echo "Wrote ${ARCHIVE}"
echo "Wrote ${FILELIST}"
echo "Wrote ${CHECKSUMS}"
