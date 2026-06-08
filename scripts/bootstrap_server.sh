#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
KERNEL_NAME="${KERNEL_NAME:-comp4040-repr}"

cd "$PROJECT_DIR"

choose_python() {
  for candidate in python3.12 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      if "$candidate" - <<'PY'; then
import sys
version = sys.version_info
raise SystemExit(0 if (3, 10) <= version[:2] <= (3, 12) else 1)
PY
        echo "$candidate"
        return 0
      fi
    fi
  done
  return 1
}

PYTHON_BIN="$(choose_python || true)"
UV_BIN=""

if [[ -z "${PYTHON_BIN:-}" ]]; then
  echo "No system Python 3.11/3.12 found; installing Python 3.12 with uv in user-space."
  UV_BIN="$(command -v uv || true)"
  if [[ -z "$UV_BIN" ]]; then
    if ! command -v curl >/dev/null 2>&1; then
      echo "curl is required to install uv when Python 3.11/3.12 is missing." >&2
      exit 1
    fi
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    UV_BIN="$(command -v uv || true)"
  fi
  if [[ -z "$UV_BIN" ]]; then
    echo "uv installation did not produce a uv executable on PATH." >&2
    exit 1
  fi
  "$UV_BIN" python install 3.12
  "$UV_BIN" venv --python 3.12 "$VENV_DIR"
else
  echo "Using Python: $("$PYTHON_BIN" --version)"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi found; installing requirements with default torch wheels."
  python -m pip install -r requirements.txt
else
  echo "No nvidia-smi found; installing CPU torch first."
  python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2.0"
  python -m pip install -r requirements.txt
fi

python -m pip install ipykernel
python -m ipykernel install --user --name "$KERNEL_NAME" --display-name "Python (COMP4040 server)"

python - <<'PY'
import importlib

modules = [
    "numpy",
    "pandas",
    "pyarrow",
    "sklearn",
    "gensim",
    "sentence_transformers",
    "torch",
    "matplotlib",
    "tqdm",
    "requests",
    "seaborn",
    "umap",
    "faiss",
]

for module_name in modules:
    module = importlib.import_module(module_name)
    print(f"{module_name}: {getattr(module, '__version__', 'ok')}")

import torch
print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
PY

echo "Server bootstrap complete: $PROJECT_DIR"
