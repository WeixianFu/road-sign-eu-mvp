from __future__ import annotations

import numpy as np
from PIL import Image


def starts(length, size=1280, overlap=0.2):
    end = max(0, length - size)
    points = list(range(0, end + 1, int(size * (1 - overlap))))
    return sorted(set(points + [end]))


def windows(width, height, size=1280, overlap=0.2):
    if size <= 0 or not 0 <= overlap < 1 or int(size * (1 - overlap)) == 0:
        raise ValueError("Tile size and overlap must produce a positive step")
    for y in starts(height, size, overlap):
        for x in starts(width, size, overlap):
            yield (x, y, min(x + size, width), min(y + size, height))


def area(boxes):
    return np.prod(np.maximum(boxes[:, 2:] - boxes[:, :2], 0), axis=1)


def clip_boxes(boxes, region, min_area=100, min_visibility=0.2, min_side=0):
    clipped = boxes.copy()
    clipped[:, :2] = np.maximum(clipped[:, :2], region[:2])
    clipped[:, 2:] = np.minimum(clipped[:, 2:], region[2:])
    sides = clipped[:, 2:] - clipped[:, :2]
    keep = (
        (sides > 0).all(axis=1)
        & (area(clipped) >= min_area)
        & (area(clipped) >= area(boxes) * min_visibility)
        & (sides.min(axis=1) >= min_side)
    )
    return clipped[keep], keep


def padded_tile(image, window, size=1280):
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(image.crop(window), (0, 0))
    return canvas


def letterbox(image, boxes, size=1280):
    width, height = image.size
    scale = min(size / width, size / height)
    new_width, new_height = round(width * scale), round(height * scale)
    left, top = (size - new_width) // 2, (size - new_height) // 2
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(image.resize((new_width, new_height), Image.Resampling.BILINEAR), (left, top))
    factors = np.array([new_width / width, new_height / height] * 2)
    shifted = boxes * factors + np.array([left, top] * 2)
    return canvas, shifted, {"scale_xy": factors[:2].tolist(), "pad_xy": [left, top]}


def iou(box, boxes):
    intersection = np.prod(
        np.maximum(np.minimum(box[2:], boxes[:, 2:]) - np.maximum(box[:2], boxes[:, :2]), 0), axis=1
    )
    return intersection / (area(boxes) + np.prod(box[2:] - box[:2]) - intersection)


def nms(boxes, scores, classes, threshold=0.5):
    keep = []
    order = np.argsort(-scores, kind="stable")
    while len(order):
        first, rest = order[0], order[1:]
        keep.append(int(first))
        overlaps = iou(boxes[first], boxes[rest])
        order = rest[(classes[rest] != classes[first]) | (overlaps <= threshold)]
    return keep
