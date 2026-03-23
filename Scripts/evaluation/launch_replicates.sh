#!/bin/bash
# launch_replicates.sh — Submit N replicate pipeline runs via SLURM.
#
# Usage:
#   bash Scripts/evaluation/launch_replicates.sh --replicates 5
#   bash Scripts/evaluation/launch_replicates.sh --replicates 5 --context context_wp1.md
#   bash Scripts/evaluation/launch_replicates.sh --replicates 10 --dry-run
#
# Each replicate uses the same context.md configuration (same run_label),
# so the evaluation framework will group them automatically.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
BATCH_SCRIPT="${PROJECT_DIR}/Batch_Script.sh"

# Defaults
REPLICATES=5
CONTEXT_PATH=""
DRY_RUN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --replicates|-n)
            REPLICATES="$2"
            shift 2
            ;;
        --context|-c)
            CONTEXT_PATH="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--replicates N] [--context PATH] [--dry-run]"
            echo ""
            echo "Options:"
            echo "  --replicates, -n  Number of replicate runs (default: 5)"
            echo "  --context, -c     Path to context.md to use (default: project context.md)"
            echo "  --dry-run         Print sbatch commands without executing"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

# Validate
if [[ ! -f "$BATCH_SCRIPT" ]]; then
    echo "ERROR: Batch script not found: $BATCH_SCRIPT" >&2
    exit 1
fi

if [[ -n "$CONTEXT_PATH" && ! -f "$CONTEXT_PATH" ]]; then
    echo "ERROR: Context file not found: $CONTEXT_PATH" >&2
    exit 1
fi

# If a custom context.md was provided, temporarily swap it in
ORIGINAL_CONTEXT=""
if [[ -n "$CONTEXT_PATH" ]]; then
    ORIGINAL_CONTEXT="${PROJECT_DIR}/context.md"
    if [[ -f "$ORIGINAL_CONTEXT" ]]; then
        echo "Backing up original context.md -> context.md.bak"
        cp "$ORIGINAL_CONTEXT" "${ORIGINAL_CONTEXT}.bak"
    fi
    echo "Using context file: $CONTEXT_PATH"
    cp "$CONTEXT_PATH" "${PROJECT_DIR}/context.md"
fi

# Submit replicates
echo "Submitting ${REPLICATES} replicate runs..."
echo ""

JOB_IDS=()
for i in $(seq 1 "$REPLICATES"); do
    if [[ "$DRY_RUN" == true ]]; then
        echo "  [DRY RUN] Replicate $i/$REPLICATES: sbatch $BATCH_SCRIPT"
    else
        JOB_ID=$(sbatch --parsable "$BATCH_SCRIPT")
        JOB_IDS+=("$JOB_ID")
        echo "  Replicate $i/$REPLICATES: submitted (Job ID: $JOB_ID)"
    fi
done

# Restore original context.md if we swapped it
if [[ -n "$CONTEXT_PATH" && -f "${PROJECT_DIR}/context.md.bak" ]]; then
    echo ""
    echo "Restoring original context.md from backup"
    mv "${PROJECT_DIR}/context.md.bak" "${PROJECT_DIR}/context.md"
fi

echo ""
if [[ "$DRY_RUN" == true ]]; then
    echo "Dry run complete. No jobs submitted."
else
    echo "All ${REPLICATES} replicates submitted."
    echo "Job IDs: ${JOB_IDS[*]}"
    echo ""
    echo "Monitor with:  squeue -u \$USER"
    echo "After completion, evaluate with:"
    echo "  python -m Scripts.evaluation full --glob 'Outputs/slurm_*' --baseline <label>"
fi
