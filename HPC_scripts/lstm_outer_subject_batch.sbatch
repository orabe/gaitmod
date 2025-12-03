#!/bin/bash
#
# SLURM batch script for training on a subject subset.
#
# Usage:
#   sbatch HPC_scripts/lstm_outer_subject_batch.sbatch <CONFIG_PATH> <SUBJECTS> [RUN_ID]
#   sbatch --job-name=PW_EM59 HPC_scripts/lstm_outer_subject_batch.sbatch config.json PW_EM59 20250101_120000

#SBATCH --job-name=lstm_subject
#SBATCH --partition=gpu-2h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96gb
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

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

CONFIG_PATH="${1:-}"
SUBJECTS="${2:-}"
RUN_ID="${3:-}"

if [[ -z "$CONFIG_PATH" || -z "$SUBJECTS" ]]; then
    echo "[SLURM] Usage: $0 <config_path> <subjects> [run_id]" >&2
    exit 2
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "[SLURM] Hyperparameter config not found: $CONFIG_PATH" >&2
    exit 2
fi
if [[ -z "$RUN_ID" ]]; then
    RUN_ID="$(date +'%Y%m%d_%H%M%S')"
fi

CMD=(python examples/LFP/classification_experiments/train_lstm_hctsa.py)
CMD+=(--hyperparams-config "$CONFIG_PATH")
CMD+=(--run-id "$RUN_ID")
CMD+=(--outer-subjects "$SUBJECTS")

echo "[SLURM] Running: ${CMD[*]}"
"${CMD[@]}"
