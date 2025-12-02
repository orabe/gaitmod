#!/bin/bash
#
# SLURM batch script for training on a subject subset.
#
# Usage:
#   sbatch --export=SUBJECTS="PW_EM59,PW_SN61" scripts/lstm_outer_subject_batch.sbatch
#   sbatch scripts/lstm_outer_subject_batch.sbatch

#SBATCH --job-name=PW_EM59
#SBATCH --partition=gpu-2d
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96gb
#SBATCH --output=logs/slurm/PW_EM59_%j.out
#SBATCH --error=logs/slurm/PW_EM59_%j.err

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"

mkdir -p "$PROJECT_ROOT/logs/slurm"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

# --- Environment setup ------------------------------------------------------
# Customize this block to activate your preferred Python environment.
# shellcheck disable=SC2015
if [[ -f "$HOME/.bashrc" ]]; then
    source "$HOME/.bashrc"
fi
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate gaitmod-cluster
else
    echo "[SLURM] conda not found on PATH; please load/activate your environment before submitting." >&2
    exit 1
fi
# ----------------------------------------------------------------------------

SUBJECTS="${SUBJECTS:-}"
CONFIG_PATH="${1:-}"

if [[ -z "$CONFIG_PATH" ]]; then
    echo "[SLURM] Hyperparameter config path argument is required" >&2
    exit 2
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "[SLURM] Hyperparameter config not found: $CONFIG_PATH" >&2
    exit 2
fi
shift || true

CMD=(python examples/LFP/classification_experiments/train_lstm_hctsa.py)
CMD+=(--hyperparams-config "$CONFIG_PATH")
if [[ -n "${RUN_ID:-}" ]]; then
    CMD+=(--run-id "$RUN_ID")
fi
if [[ -n "$SUBJECTS" ]]; then
    CMD+=(--outer-subjects "$SUBJECTS")
fi

echo "[SLURM] Running: ${CMD[*]}"
"${CMD[@]}"
