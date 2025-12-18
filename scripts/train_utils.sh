#!/bin/bash
# Utility functions for training management
# Usage: source scripts/train_utils.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/train.log"
PID_FILE="$SCRIPT_DIR/train.pid"

# View training log (tail with follow)
train_log() {
    if [ -f "$LOG_FILE" ]; then
        tail -f "$LOG_FILE"
    else
        echo "Log file not found: $LOG_FILE"
        exit 1
    fi
}

# View last N lines of log
train_log_last() {
    local lines=${1:-50}
    if [ -f "$LOG_FILE" ]; then
        tail -n "$lines" "$LOG_FILE"
    else
        echo "Log file not found: $LOG_FILE"
        exit 1
    fi
}

# Check if training is running
train_status() {
    if [ ! -f "$PID_FILE" ]; then
        echo "No training PID file found. Training may not be running."
        return 1
    fi
    
    local pid=$(cat "$PID_FILE")
    
    if ps -p "$pid" > /dev/null 2>&1; then
        echo "Training is running (PID: $pid)"
        echo "Log file: $LOG_FILE"
        return 0
    else
        echo "Training process not found (PID: $pid may have exited)"
        return 1
    fi
}

# Stop training
train_stop() {
    if [ ! -f "$PID_FILE" ]; then
        echo "No training PID file found."
        exit 1
    fi
    
    local pid=$(cat "$PID_FILE")
    
    if ps -p "$pid" > /dev/null 2>&1; then
        echo "Stopping training (PID: $pid)..."
        kill "$pid"
        
        # Wait for process to stop
        local count=0
        while ps -p "$pid" > /dev/null 2>&1 && [ $count -lt 10 ]; do
            sleep 1
            count=$((count + 1))
        done
        
        if ps -p "$pid" > /dev/null 2>&1; then
            echo "Force killing training process..."
            kill -9 "$pid"
        fi
        
        rm -f "$PID_FILE"
        echo "Training stopped."
    else
        echo "Training process not found (PID: $pid)"
        rm -f "$PID_FILE"
    fi
}

# Show usage
train_help() {
    echo "Training utility commands:"
    echo "  source scripts/train_utils.sh  # Load utilities"
    echo "  train_log                      # Follow training log (tail -f)"
    echo "  train_log_last [N]             # Show last N lines (default: 50)"
    echo "  train_status                   # Check if training is running"
    echo "  train_stop                     # Stop training"
}

