#!/usr/bin/env bash
# Monitor preprocessing job 3264115 on VACC.
# When it completes: verify manifest, submit new training run, cancel old job 3264101.

set -euo pipefail

VACC_SCRIPTS=~/.claude/plugins/marketplaces/claude-plugins-official/plugins/vacc/skills/vacc/scripts
PREPROCESS_JOB=3264115
OLD_TRAIN_JOB=3264101
POLL_INTERVAL=300  # seconds between polls

source_vacc() { bash "$VACC_SCRIPTS/vacc_cmd.sh" "$@"; }

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " BIOSENSE ML — Preprocessing Monitor"
echo " Job: $PREPROCESS_JOB   Old train job: $OLD_TRAIN_JOB"
echo " Poll interval: ${POLL_INTERVAL}s"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Ensure VACC session is alive
bash "$VACC_SCRIPTS/vacc_session.sh"

while true; do
    TIMESTAMP=$(date '+%H:%M:%S')

    # Get job state
    STATE=$(source_vacc "sacct -j $PREPROCESS_JOB --noheader --format=State --parsable2 | head -1 | tr -d ' '" 2>/dev/null | tail -1 || echo "UNKNOWN")

    # Get latest log line
    LOG_LINE=$(source_vacc "tail -1 ~/projects/biosense_ml/logs/preprocess_${PREPROCESS_JOB}.out 2>/dev/null || echo '(no log yet)'" 2>/dev/null | tail -1)

    echo "[$TIMESTAMP] State: $STATE | $LOG_LINE"

    case "$STATE" in
        COMPLETED)
            echo ""
            echo "✓ Preprocessing COMPLETED"
            echo ""

            # Check manifest
            MANIFEST=$(source_vacc "python3 -c \"
import json, pathlib
m = json.load(open('/users/k/t/ktbielaw/projects/biosense_ml/data/processed/manifest.json'))
print(f'Samples: {m[\\\"num_samples\\\"]}, Shards: {len(m[\\\"shard_paths\\\"])}')
\"" 2>/dev/null | tail -1)
            echo "Manifest: $MANIFEST"
            echo ""

            # Submit new training run
            echo "Submitting new training job..."
            NEW_JOB=$(source_vacc "cd ~/projects/biosense_ml && sbatch slurm/submit_autoencoder.sh 2>&1" | grep -oE '[0-9]{7}' | head -1)
            echo "New training job: $NEW_JOB"

            # Wait a moment then check it started
            sleep 10
            STATUS=$(source_vacc "squeue -j $NEW_JOB --noheader --format='%T %R' 2>/dev/null || echo 'not in queue'" | tail -1)
            echo "New job status: $STATUS"

            # Cancel old training job
            echo ""
            echo "Cancelling old training job $OLD_TRAIN_JOB (trained on uncropped data)..."
            source_vacc "scancel $OLD_TRAIN_JOB 2>/dev/null; echo done" | tail -1

            echo ""
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo " Done. New training job: $NEW_JOB"
            echo " Monitor with: vacc_monitor.sh $NEW_JOB"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            exit 0
            ;;
        FAILED|TIMEOUT|CANCELLED|NODE_FAIL)
            echo ""
            echo "✗ Preprocessing $STATE — fetching logs..."
            source_vacc "tail -80 ~/projects/biosense_ml/logs/preprocess_${PREPROCESS_JOB}.out 2>/dev/null || echo '(no log)'"
            echo ""
            source_vacc "tail -40 ~/projects/biosense_ml/logs/preprocess_${PREPROCESS_JOB}.err 2>/dev/null || echo '(no stderr log)'"
            exit 1
            ;;
        *)
            sleep "$POLL_INTERVAL"
            ;;
    esac
done
