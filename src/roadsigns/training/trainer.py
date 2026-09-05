from __future__ import annotations

import logging

import torch
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import RANK

from roadsigns.common import event, setup_logging, write_json
from roadsigns.training.dataset import HybridDataset


def check_loss(trainer):
    if not torch.isfinite(trainer.loss).all():
        path = trainer.save_dir / f"nonfinite-rank{RANK}.json"
        write_json(
            path,
            {
                "epoch": trainer.epoch + 1,
                "sources": trainer.current_batch["sample_sources"],
                "classes": trainer.current_batch["cls"].cpu().tolist(),
                "boxes": trainer.current_batch["bboxes"].cpu().tolist(),
                "batch_idx": trainer.current_batch["batch_idx"].cpu().tolist(),
            },
        )
        torch.save(trainer.current_batch["img"].cpu(), path.with_suffix(".pt"))
        raise FloatingPointError(f"Non-finite training loss; batch evidence saved to {path}")


def record_epoch(trainer):
    if RANK in (-1, 0):
        event(
            trainer.save_dir.parent / "operations.jsonl",
            "epoch",
            epoch=trainer.epoch + 1,
            metrics={k: float(v) for k, v in trainer.metrics.items()},
        )
        logging.getLogger("roadsigns").info("Completed epoch %d", trainer.epoch + 1)


class TrafficTrainer(DetectionTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if RANK == 0:
            setup_logging(self.save_dir.parent / "train.log")
        # Register here so Ultralytics' spawned DDP trainers use the same callbacks.
        self.add_callback("on_train_batch_end", check_loss)
        self.add_callback("on_fit_epoch_end", record_epoch)

    def build_dataset(self, img_path, mode="train", batch=None):
        dataset_class = HybridDataset if mode == "train" else YOLODataset
        return dataset_class(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=False,
            cache=False,
            data=self.data,
            task="detect",
            stride=32,
            pad=0.0,
            prefix=f"{mode}: ",
        )

    def preprocess_batch(self, batch):
        result = super().preprocess_batch(batch)
        self.current_batch = result
        return result
