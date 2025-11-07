#!/usr/bin/env bash
set -euo pipefail

# Segmented allocator keeps fragmentation down on multi-day runs.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

: "${MODEL:=yolov8m.pt}"
: "${RESUME:=False}"
: "${EPOCHS:=100}"
: "${IMG_SIZE:=1536}"
: "${BATCH:=16}"
: "${DEVICE:=0，1}"
: "${WORKERS:=4}"
: "${RUN_NAME:=mtsd_fully_v8m_p2}"

python -m src.detect.yolo \
  --model "${MODEL}" \
  --data configs/mtsd_fully_v8_p2.yaml \
  --project runs/mtsd \
  --name "${RUN_NAME}" \
  --override resume="${RESUME}" \
    epochs="${EPOCHS}" \
    imgsz="${IMG_SIZE}" \
    batch="${BATCH}" \
    device="${DEVICE}" \
    workers="${WORKERS}" \
    save_period=5 \
    cos_lr=True \
    val=True \
    amp=True
