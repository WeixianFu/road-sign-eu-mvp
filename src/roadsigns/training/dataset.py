from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from ultralytics.data.augment import Compose, Format, RandomHSV
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils.instance import Instances

from roadsigns.training.mosaic import stitch


class TinyMosaic:
    def __init__(self, dataset, hyp):
        self.dataset = dataset
        self.hyp = hyp

    def __call__(self, label):
        label["sample_sources"] = [label["im_file"]]
        if random.random() >= self.hyp.mosaic:
            return label
        labels = [label] + [self.dataset.sample() for _ in range(3)]
        samples = []
        for item in labels:
            instances = item["instances"]
            instances.convert_bbox("xyxy")
            instances.denormalize(self.dataset.imgsz, self.dataset.imgsz)
            samples.append((item["img"], item["cls"], instances.bboxes))
        cfg = self.dataset.data["tiny_training"]
        image, classes, boxes, _ = stitch(
            samples,
            self.dataset.imgsz,
            self.hyp.scale,
            cfg["tiny_side"],
            cfg["mosaic_min_side"],
            cfg["mosaic_min_visibility"],
        )
        label["img"] = image
        label["cls"] = classes
        label["instances"] = Instances(
            boxes,
            segments=np.zeros((0, 1000, 2), dtype=np.float32),
            bbox_format="xyxy",
            normalized=False,
        )
        label["sample_sources"] = [item["im_file"] for item in labels]
        return label


class HybridDataset(YOLODataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.full = [i for i, p in enumerate(self.im_files) if Path(p).stem.endswith("__full")]
        self.tiles = [i for i, p in enumerate(self.im_files) if "__tile_" in Path(p).stem]
        self.full_ratio = self.data["tiny_training"]["full_ratio"]
        if not self.tiles or (self.full_ratio > 0 and not self.full):
            raise ValueError(
                "Hybrid sampling requires tiles and eligible full images; check manifest"
            )

    def sample(self):
        pool = self.full if random.random() < self.full_ratio else self.tiles
        return self.get_image_and_label(random.choice(pool))

    def __getitem__(self, index):
        # Sampling with replacement makes the 70/30 ratio independent of tile counts.
        return self.transforms(self.sample())

    def build_transforms(self, hyp=None):
        return Compose(
            [
                TinyMosaic(self, hyp),
                RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
                Format(bbox_format="xywh", normalize=True, batch_idx=True, bgr=0.0),
            ]
        )
