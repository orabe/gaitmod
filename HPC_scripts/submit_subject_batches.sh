#!/bin/bash
#
# Helper script to submit multiple subject batches in sequence.

set -euo pipefail

CONFIG_NAME="hparams_test.json"
BATCH_SCRIPT="HPC_scripts/outer_subject_batch.sbatch"
HYPERPARAMS_CONFIG="gaitmod/configs/hparams_configs/$CONFIG_NAME"

if [[ ! -f "$HYPERPARAMS_CONFIG" ]]; then
    echo "Hyperparameter config not found: $HYPERPARAMS_CONFIG" >&2
    exit 1
fi

# Define batches. Update this array to match your subject roster.
declare -a SUBJECT_BATCHES=(
    "PW_EM59",
    "PW_FH57",
    "PW_HK59",
    "PW_HZ58",
    "PW_SN61",
    "PW_SN66",
    "PW_US68",
)

RUN_ID="${RUN_ID:-$(date +'%Y%m%d_%H%M%S')}"
echo "Using RUN_ID=$RUN_ID"

if [[ ! -f "$BATCH_SCRIPT" ]]; then
    echo "Batch script not found: $BATCH_SCRIPT" >&2
    exit 1
fi

for subject in "${SUBJECT_BATCHES[@]}"; do
    echo "Submitting job for subject: $subject (config: $HYPERPARAMS_CONFIG)"
    sbatch --job-name="$subject" "$BATCH_SCRIPT" "$HYPERPARAMS_CONFIG" "$subject" "$RUN_ID"
done
