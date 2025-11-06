#!/usr/bin/env bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

python -m src.detect.yolo \
  --model runs/mtsd/mtsd_fully_v8m_p1/weights/last.pt \
  --data configs/mtsd_fully_v8_p1.yaml \
  --project runs/mtsd \
  --name mtsd_fully_v8m_p1 \
  --override resume=True\
    epochs=100 \
    imgsz=960 \
    batch=12 \
    device=0 \
    workers=4 \
    save_period=5 \
    cos_lr=True \
    val=False


