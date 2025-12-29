#!/bin/bash
# Training script for Linux server
# Usage: ./scripts/train.sh
# 
# This script runs training in the background and logs all output to train.log
# Training will continue even if SSH connection is lost (nohup)

set -e  # Exit on error

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to project root directory
cd "$PROJECT_ROOT"

# Log file path (in scripts/ directory, same as script)
LOG_FILE="$SCRIPT_DIR/train.log"

# Configuration files
TRAIN_CONFIG="$PROJECT_ROOT/configs/train.yaml"
DATA_CONFIG="$PROJECT_ROOT/configs/data.yaml"

# Python script
TRAIN_SCRIPT="$PROJECT_ROOT/src/detect/train.py"

# Check if config files exist
if [ ! -f "$TRAIN_CONFIG" ]; then
    echo "Error: Training config not found: $TRAIN_CONFIG" >&2
    exit 1
fi

if [ ! -f "$DATA_CONFIG" ]; then
    echo "Error: Data config not found: $DATA_CONFIG" >&2
    exit 1
fi

if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "Error: Training script not found: $TRAIN_SCRIPT" >&2
    exit 1
fi

# Print startup info
echo "==========================================" | tee -a "$LOG_FILE"
echo "Starting training at $(date)" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo "Project root: $PROJECT_ROOT" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "Train config: $TRAIN_CONFIG" | tee -a "$LOG_FILE"
echo "Data config: $DATA_CONFIG" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Server data paths
# valfull 包含原始 val 数据的 GT labels
SLICED_ROOT="/root/autodl-tmp/MTSD_download/mtsd-resized"
GT_LABELS_DIR="$SLICED_ROOT/valfull/labels"

# Run training with nohup (disconnect-safe)
# -u: unbuffered Python output (real-time logging)
# 2>&1: redirect stderr to stdout
# >>: append to log file (no terminal output)
# nohup: continue running after SSH disconnect
nohup python -u "$TRAIN_SCRIPT" \
    --config "$TRAIN_CONFIG" \
    --data "$DATA_CONFIG" \
    --gt-labels-dir "$GT_LABELS_DIR" \
    --full-val-interval 10 \
    >> "$LOG_FILE" 2>&1 &

# Get process ID
TRAIN_PID=$!

# Save PID to file for later reference
echo "$TRAIN_PID" > "$SCRIPT_DIR/train.pid"

echo "Training started with PID: $TRAIN_PID" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "To monitor progress: tail -f $LOG_FILE" | tee -a "$LOG_FILE"
echo "To stop training: kill $TRAIN_PID" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Wait a moment to check if process started successfully
sleep 2

if ps -p $TRAIN_PID > /dev/null; then
    echo "Training process is running successfully." | tee -a "$LOG_FILE"
    echo "You can safely disconnect from SSH. Training will continue." | tee -a "$LOG_FILE"
    exit 0
else
    echo "Error: Training process failed to start. Check $LOG_FILE for details." >&2
    exit 1
fi

