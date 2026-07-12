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

# Configuration files (override: ./scripts/train.sh [train_cfg] [data_cfg])
TRAIN_CONFIG="${1:-$PROJECT_ROOT/configs/train_core146.yaml}"
DATA_CONFIG="${2:-$PROJECT_ROOT/configs/data_core146.yaml}"

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

# Full-image validation disabled for mtsd_core146:
# 数据集里没有 val/index 和 valfull GT labels，且旧 valfull GT 是 401 类 id，
# 与 146 类模型不匹配。恢复 full-val 需先用 401→146 映射重新生成 valfull labels。
# Run training with nohup (disconnect-safe)
# -u: unbuffered Python output (real-time logging)
# 2>&1: redirect stderr to stdout
# >>: append to log file (no terminal output)
# nohup: continue running after SSH disconnect
nohup python -u "$TRAIN_SCRIPT" \
    --config "$TRAIN_CONFIG" \
    --data "$DATA_CONFIG" \
    --no-full-val \
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

