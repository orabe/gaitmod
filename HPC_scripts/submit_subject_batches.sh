#!/bin/bash
#
# Helper script to submit multiple subject batches in sequence.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BATCH_SCRIPT="$SCRIPT_DIR/lstm_outer_subject_batch.sbatch"

# Define batches. Update this array to match your subject roster.
declare -a SUBJECT_BATCHES=(
    "PW_EM59",
    # "PW_FH57",
    # "PW_HK59",
    # "PW_HZ58",
    # "PW_SN61",
    # "PW_SN66",
    # "PW_US68"
)

RUN_ID="${RUN_ID:-$(date +'%Y%m%d_%H%M%S')}"
echo "Using RUN_ID=$RUN_ID"

if [[ ! -f "$BATCH_SCRIPT" ]]; then
    echo "Batch script not found: $BATCH_SCRIPT" >&2
    exit 1
fi

for subjects in "${SUBJECT_BATCHES[@]}"; do
    echo "Submitting batch for subjects: $subjects"
    sbatch --export=SUBJECTS="$subjects",RUN_ID="$RUN_ID" "$BATCH_SCRIPT"
done
