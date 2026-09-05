from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from roadsigns.common import (
    digest,
    environment,
    event,
    file_hash,
    read_yaml,
    setup_logging,
    write_json,
)
from roadsigns.image_tools.geometry import clip_boxes, nms, padded_tile, windows


class YoloBackend:
    def __init__(self, weights, cfg, names):
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        if [self.model.names[i] for i in range(len(self.model.names))] != names:
            raise ValueError("Model class names do not match this run's ontology")
        self.cfg = cfg

    def __call__(self, images):
        results = self.model.predict(
            images,
            imgsz=self.cfg["imgsz"],
            conf=self.cfg["confidence"],
            iou=self.cfg["tile_iou"],
            device=self.cfg["device"],
            rect=False,
            max_det=self.cfg["max_detections"],
            verbose=False,
        )
        return [
            np.column_stack(
                (r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy())
            )
            for r in results
        ]


def predict_image(image, backend, cfg):
    size = cfg["imgsz"]
    regions = list(windows(*image.size, size, cfg["overlap"]))
    predictions, sources = [], []
    for start in range(0, len(regions), cfg["tile_batch"]):
        batch_regions = regions[start : start + cfg["tile_batch"]]
        tiles = [
            np.asarray(padded_tile(image, roi, size))[:, :, ::-1].copy() for roi in batch_regions
        ]
        outputs = backend(tiles)
        if len(outputs) != len(tiles):
            raise ValueError("Backend returned a different number of tiles")
        for roi, detections in zip(batch_regions, outputs):
            boxes = detections[:, :4] + np.array(roi[:2] * 2)
            clipped, keep = clip_boxes(boxes, roi, min_area=0, min_visibility=0)
            predictions.append(np.column_stack((clipped, detections[keep, 4:])))
            sources.extend([list(roi)] * int(keep.sum()))
    merged = np.concatenate(predictions)
    keep = nms(merged[:, :4], merged[:, 4], merged[:, 5], cfg["merge_iou"])
    records = []
    for index in keep[: cfg["max_detections"]]:
        x1, y1, x2, y2, score, cls = merged[index]
        records.append(
            {
                "category_id": int(cls),
                "score": float(score),
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "tile_roi": sources[index],
            }
        )
    return records


def predict(coco_path, weights, cfg, output, resume=False, backend=None):
    coco_path, weights, output = Path(coco_path), Path(weights), Path(output)
    if cfg["imgsz"] != 1280 or cfg["overlap"] != 0.2 or cfg["tile_batch"] <= 0:
        raise ValueError("Prediction requires 1280 / 20% slicing and a positive tile batch")
    coco = json.loads(coco_path.read_text())
    run = json.loads((weights.parents[2] / "run.json").read_text())
    names = run["ontology"]["names"]
    if coco["info"]["ontology_fingerprint"] != run["ontology"]["fingerprint"] or coco[
        "categories"
    ] != [{"id": i, "name": n} for i, n in enumerate(names)]:
        raise ValueError("Evaluation GT and model ontology do not match")
    identity = {
        "gt_sha256": file_hash(coco_path),
        "model_sha256": file_hash(weights),
        "ontology_fingerprint": run["ontology"]["fingerprint"],
        "config": cfg,
    }
    run_id = digest(identity)
    if resume:
        if json.loads((output / "prediction.json").read_text())["run_id"] != run_id:
            raise ValueError("Resume requires unchanged model, GT and inference configuration")
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output / "parts").mkdir()
        write_json(
            output / "prediction.json",
            {"run_id": run_id, **identity, "environment": environment(), "status": "running"},
        )
    logger = setup_logging(output / "predict.log")
    backend = backend if backend is not None else YoloBackend(weights, cfg, names)
    all_predictions = []
    started = time.perf_counter()
    for image in coco["images"]:
        part = output / "parts" / f"{image['id']:08d}.json"
        if part.exists():
            predictions = json.loads(part.read_text())
        else:
            source = Path(image["file_name"])
            if file_hash(source) != image["sha256"]:
                raise ValueError(f"Source image changed since preparation: {source}")
            frame = Image.open(source).convert("RGB")
            tick = time.perf_counter()
            predictions = [
                {"image_id": image["id"], **p} for p in predict_image(frame, backend, cfg)
            ]
            temporary = part.with_suffix(".tmp")
            write_json(temporary, predictions)
            temporary.replace(part)
            event(
                output / "operations.jsonl",
                "image",
                image_id=image["id"],
                detections=len(predictions),
                seconds=time.perf_counter() - tick,
            )
        all_predictions.extend(predictions)
        logger.info("Image %s: %d detections", image["source_id"], len(predictions))
    write_json(output / "predictions.coco.json", all_predictions)
    metadata = json.loads((output / "prediction.json").read_text())
    metadata.update(
        status="complete",
        images=len(coco["images"]),
        image_ids=[im["id"] for im in coco["images"]],
        seconds_this_invocation=time.perf_counter() - started,
    )
    write_json(output / "prediction.json", metadata)
    return all_predictions


def main():
    parser = argparse.ArgumentParser(description="Predict MTSD original images with tiled YOLO")
    parser.add_argument("--coco", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/predict.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    predict(args.coco, args.weights, read_yaml(args.config), args.output, args.resume)


if __name__ == "__main__":
    main()
