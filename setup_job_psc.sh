#!/bin/bash
#SBATCH --job-name=trainingBios
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --time=48:00:00
#SBATCH --account=cis240072p
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=egfriedm@andrew.cmu.edu

# Notes on the conservative defaults above:
#   - GPU-shared lets you book 1 GPU instead of a whole 8-GPU node.
#   - gpu:1 picks one A100 on the shared partition; switch to gpu:v100-32:1
#     if you ever want a cheaper V100 node.
#   - 5 CPUs covers HF Trainer's 2 dataloader workers + the main process.
#   - No explicit --mem: GPU-shared auto-allocates ~125G per A100 which is
#     plenty for these small models.

set -e   # bail if `conda activate` (or any earlier command) fails

# --- Environment ------------------------------------------------------------
mkdir -p logs

# Clean module state so the job is reproducible regardless of login shell.
module purge
module load cuda
module load anaconda3
# module load anaconda3                # uncomment if your env needs it

# Initialize conda for this non-interactive shell. `module load anaconda3`
# puts conda on $PATH but does NOT define the `conda` shell function, so
# `conda activate` fails with "Run 'conda init' before 'conda deactivate'".
# Try several known-good init paths; first one that works wins.
echo "--- conda init debug ---"
echo "which conda: $(which conda 2>/dev/null || echo none)"
echo "CONDA_EXE:   ${CONDA_EXE:-unset}"

# Disable -e for the init block so a failing fallback doesn't kill the job.
set +e
_conda_initialized=0

# 1. The canonical conda-recommended pattern.
eval "$(conda shell.bash hook 2>/dev/null)" \
    && _conda_initialized=1 && echo "conda init: shell.bash hook OK"

# 2. Source conda.sh from `conda info --base`.
if [ "$_conda_initialized" = 0 ]; then
    _base="$(conda info --base 2>/dev/null)"
    if [ -n "$_base" ] && [ -f "$_base/etc/profile.d/conda.sh" ]; then
        source "$_base/etc/profile.d/conda.sh" \
            && _conda_initialized=1 \
            && echo "conda init: sourced $_base/etc/profile.d/conda.sh"
    fi
fi

# 3. Derive base from $(which conda) and source from there.
if [ "$_conda_initialized" = 0 ]; then
    _conda_bin="$(which conda 2>/dev/null)"
    if [ -n "$_conda_bin" ]; then
        _base="$(dirname "$(dirname "$_conda_bin")")"
        if [ -f "$_base/etc/profile.d/conda.sh" ]; then
            source "$_base/etc/profile.d/conda.sh" \
                && _conda_initialized=1 \
                && echo "conda init: sourced $_base/etc/profile.d/conda.sh (via which)"
        fi
    fi
fi

set -e
if [ "$_conda_initialized" = 0 ]; then
    echo "ERROR: could not initialize conda — none of the init paths worked." >&2
    exit 1
fi
echo "--- conda init done ---"

# Activate the project conda env.
conda activate lm4
echo "Active env: ${CONDA_DEFAULT_ENV:-none}  ($(which python))"

# Keep HuggingFace + wandb caches on project storage (data_storage -> /ocean)
# so they never count against the 25G /jet home quota. Previously these went
# to $LOCAL node-scratch when set (fast, but per-job ephemeral — re-downloads
# every job) and silently fell back to $HOME otherwise, which is what kept
# filling the quota. The repo's cache/ runs/ wandb/ logs/ dirs are symlinks
# into data_storage too, so ALL heavy writes land on /ocean.
export HF_HOME="$HOME/data_storage/hf_cache"
export WANDB_CACHE_DIR="$HOME/data_storage/wandb_cache"
mkdir -p "$HF_HOME" "$WANDB_CACHE_DIR"

# wandb auth: prefer `wandb login` once on the login node (writes ~/.netrc).
# If you can't do that, uncomment and set:
# export WANDB_API_KEY=<your_api_key>

# Make Python output unbuffered so the SLURM log streams live.
export PYTHONUNBUFFERED=1

# --- Run --------------------------------------------------------------------
cd "$SLURM_SUBMIT_DIR"

echo "=== Job $SLURM_JOB_ID on $(hostname) ==="
echo "Started: $(date)"
nvidia-smi || true
echo

python -u $1

echo
echo "Finished: $(date)"
