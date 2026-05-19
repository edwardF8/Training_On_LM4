#!/usr/bin/env bash
# Bootstrap script for the Capo (bioS) training pipeline.
#
# What it does:
#   1. Verifies Python 3.10+ is on PATH (needed for newer typing syntax).
#   2. Creates a `.venv` virtual environment if one doesn't already exist.
#   3. Activates the venv and installs everything in requirements.txt.
#   4. Runs `wandb login` so the Trainer can stream metrics to your account.
#   5. Prints a short sanity report (torch/transformers versions, device).
#
# Usage (from the project root):
#   bash setup.sh
#
# Idempotent — re-running just skips already-done steps. To start over,
# `rm -rf .venv` then re-run.

set -euo pipefail

# --- Config ------------------------------------------------------------------
PYTHON="${PYTHON:-python3}"   # override with `PYTHON=python3.11 bash setup.sh`
VENV_DIR=".venv"
REQ_FILE="requirements.txt"


# --- 1. Python version check -------------------------------------------------
echo "==> Checking Python version..."
if ! "$PYTHON" -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"; then
    echo "ERROR: need Python 3.10+ (found $("$PYTHON" --version 2>&1))." >&2
    echo "       Install a newer Python and re-run, e.g. \`PYTHON=python3.11 bash setup.sh\`." >&2
    exit 1
fi
echo "    OK: $("$PYTHON" --version)"


# --- 2. Create venv ----------------------------------------------------------
if [ ! -d "$VENV_DIR" ]; then
    echo "==> Creating virtual environment at $VENV_DIR ..."
    "$PYTHON" -m venv "$VENV_DIR"
else
    echo "==> Reusing existing venv at $VENV_DIR"
fi


# --- 3. Activate + install ---------------------------------------------------
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

echo "==> Upgrading pip ..."
python -m pip install --upgrade pip --quiet

echo "==> Installing requirements from $REQ_FILE ..."
pip install -r "$REQ_FILE"


# --- 4. wandb login ----------------------------------------------------------
# `wandb login` is interactive but idempotent: if you're already logged in
# it prints "Currently logged in as: ..." and exits 0. Otherwise it prompts
# for an API key (grab one from https://wandb.ai/authorize).
echo "==> Configuring Weights & Biases ..."
wandb login || {
    echo "WARN: wandb login failed. You can still train (the Trainer will print to stderr),"  >&2
    echo "      but online logging will be disabled. Re-run \`wandb login\` later to fix."     >&2
}


# --- 5. Sanity check ---------------------------------------------------------
echo "==> Environment summary:"
python - <<'PY'
import torch, transformers
print(f"    torch        = {torch.__version__}")
print(f"    transformers = {transformers.__version__}")
print(f"    CUDA         = {torch.cuda.is_available()}"
      + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
print(f"    MPS          = {torch.backends.mps.is_available()}")
PY


# --- 6. Final instructions ---------------------------------------------------
cat <<EOF

Setup complete. To start training:

    source $VENV_DIR/bin/activate
    python main.py

Outputs:
  cache/{NAME}/  — packed token files, people.json, data_config.json
  runs/{NAME}/   — HF Trainer checkpoints

Edit CONFIG.NAME at the top of main.py to namespace a fresh experiment.
EOF
