from __future__ import annotations

from pathlib import Path

import numpy as np


def read_labels(path, width, height, class_count):
    classes, boxes = [], []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        cls, cx, cy, bw, bh = line.split()
        cls = int(cls)
        values = np.array([cx, cy, bw, bh], dtype=float)
        if not (
            0 <= cls < class_count
            and np.isfinite(values).all()
            and (values >= 0).all()
            and (values <= 1).all()
            and (values[2:] > 0).all()
        ):
            raise ValueError(f"Invalid YOLO label in {path}: {line}")
        cx, cy, bw, bh = values * [width, height, width, height]
        box = np.array([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])
        tolerance = max(width, height) * 1e-5
        if (box[:2] < -tolerance).any() or (
            box[2:] > [width + tolerance, height + tolerance]
        ).any():
            raise ValueError(f"Box exceeds source image in {path}: {line}")
        boxes.append(np.clip(box, [0, 0, 0, 0], [width, height, width, height]))
        classes.append(cls)
    return np.asarray(classes, dtype=int), np.asarray(boxes, dtype=float).reshape(-1, 4)


def write_labels(path, classes, boxes, size=1280):
    lines = []
    for cls, box in zip(classes, boxes):
        center = (box[:2] + box[2:]) / (2 * size)
        sides = (box[2:] - box[:2]) / size
        lines.append(f"{int(cls)} " + " ".join(f"{v:.8f}" for v in [*center, *sides]))
    Path(path).write_text("\n".join(lines) + ("\n" if lines else ""))
