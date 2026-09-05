from __future__ import annotations

import random

import numpy as np
from PIL import Image

from roadsigns.image_tools.geometry import clip_boxes


def scale_for(boxes, rng, scale=0.1, tiny_side=32):
    tiny = len(boxes) and (boxes[:, 2:] - boxes[:, :2]).min() < tiny_side
    return rng.uniform(1.0 if tiny else 1.0 - scale, 1.0 + scale)


def stitch(
    samples,
    size=1280,
    scale=0.1,
    tiny_side=32,
    min_side=16,
    min_visibility=0.6,
    rng=random,
    center=None,
):
    xc, yc = (
        center
        if center is not None
        else (rng.randint(size // 4, 3 * size // 4), rng.randint(size // 4, 3 * size // 4))
    )
    regions = [(0, 0, xc, yc), (xc, 0, size, yc), (0, yc, xc, size), (xc, yc, size, size)]
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    result_boxes, result_classes, scales = [], [], []
    for index, ((image, classes, boxes), region) in enumerate(zip(samples, regions)):
        sampled_scale = scale_for(boxes, rng, scale, tiny_side)
        edge = round(size * sampled_scale)
        actual_scale = edge / size
        resized = np.asarray(Image.fromarray(image).resize((edge, edge), Image.Resampling.BILINEAR))
        x = xc - edge if index in (0, 2) else xc
        y = yc - edge if index in (0, 1) else yc
        x1, y1 = max(region[0], x), max(region[1], y)
        x2, y2 = min(region[2], x + edge), min(region[3], y + edge)
        canvas[y1:y2, x1:x2] = resized[y1 - y : y2 - y, x1 - x : x2 - x]
        transformed = boxes * actual_scale + np.array([x, y, x, y])
        clipped, keep = clip_boxes(transformed, region, 0, min_visibility, min_side)
        result_boxes.append(clipped)
        result_classes.append(classes[keep])
        scales.append(actual_scale)
    return (
        canvas,
        np.concatenate(result_classes),
        np.concatenate(result_boxes),
        {"center": [xc, yc], "scales": scales},
    )
