from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from roadsigns.common import file_hash, write_json
from roadsigns.image_tools.geometry import iou


def xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]


def working_point(coco, predictions, confidence=0.25, iou_threshold=0.5):
    gt, pred = defaultdict(list), defaultdict(list)
    for box in coco["annotations"]:
        gt[box["image_id"]].append(box)
    for box in predictions:
        if box["score"] >= confidence:
            pred[box["image_id"]].append(box)
    counts, errors = defaultdict(Counter), []
    small_gt = small_tp = 0
    for image in coco["images"]:
        image_id = image["id"]
        targets = gt[image_id]
        matched = set()
        for target in targets:
            counts[target["category_id"]]["gt"] += 1
            small_gt += min(target["bbox"][2:]) < 32
        for detection in sorted(pred[image_id], key=lambda d: -d["score"]):
            cls = detection["category_id"]
            candidates = [
                i for i, t in enumerate(targets) if i not in matched and t["category_id"] == cls
            ]
            overlaps = iou(
                np.asarray(xyxy(detection["bbox"])),
                np.asarray([xyxy(targets[i]["bbox"]) for i in candidates]).reshape(-1, 4),
            )
            if len(overlaps) and overlaps.max() >= iou_threshold:
                index = candidates[int(overlaps.argmax())]
                matched.add(index)
                counts[cls]["tp"] += 1
                small_tp += min(targets[index]["bbox"][2:]) < 32
            else:
                counts[cls]["fp"] += 1
                errors.append({"kind": "FP", **detection})
        for index, target in enumerate(targets):
            if index not in matched:
                errors.append(
                    {
                        "kind": "FN",
                        "image_id": image_id,
                        "category_id": target["category_id"],
                        "bbox": target["bbox"],
                    }
                )
    rows = []
    for category in coco["categories"]:
        c = counts[category["id"]]
        tp, fp, ngt = c["tp"], c["fp"], c["gt"]
        rows.append(
            {
                "class_id": category["id"],
                "class_name": category["name"],
                "gt": ngt,
                "tp": tp,
                "fp": fp,
                "fn": ngt - tp,
                "precision": tp / (tp + fp) if tp + fp else None,
                "recall": tp / ngt if ngt else None,
            }
        )
    small = {
        "definition": "original bbox short side <32 px, class-aware matching",
        "gt": small_gt,
        "tp": small_tp,
        "recall": small_tp / small_gt if small_gt else None,
    }
    return rows, errors, small


def evaluate(coco, predictions, confidence=0.25):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    image_ids = {im["id"] for im in coco["images"]}
    class_ids = {cat["id"] for cat in coco["categories"]}
    for p in predictions:
        if (
            p["image_id"] not in image_ids
            or p["category_id"] not in class_ids
            or not np.isfinite(p["bbox"] + [p["score"]]).all()
            or min(p["bbox"][2:]) <= 0
            or not 0 <= p["score"] <= 1
        ):
            raise ValueError("Prediction has an unknown image/class or invalid geometry/score")
    gt = COCO()
    gt.dataset = coco
    gt.createIndex()
    if predictions:
        dt = gt.loadRes(predictions)
    else:
        dt = COCO()
        dt.dataset = {"images": coco["images"], "categories": coco["categories"], "annotations": []}
        dt.createIndex()
    evaluator = COCOeval(gt, dt, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    labels = [
        "AP",
        "AP50",
        "AP75",
        "AP_small",
        "AP_medium",
        "AP_large",
        "AR_1",
        "AR_10",
        "AR_100",
        "AR_small",
        "AR_medium",
        "AR_large",
    ]
    metrics = {
        label: float(value) if value >= 0 else None for label, value in zip(labels, evaluator.stats)
    }
    rows, errors, small = working_point(coco, predictions, confidence)
    for row in rows:
        index = evaluator.params.catIds.index(row["class_id"])
        values = evaluator.eval["precision"][:, :, index, 0, -1]
        ap50 = values[0]
        row["AP"] = float(values[values > -1].mean()) if (values > -1).any() else None
        row["AP50"] = float(ap50[ap50 > -1].mean()) if (ap50 > -1).any() else None
    return {
        "protocol": "COCO bbox, 101 recall points, IoU .50:.05:.95, maxDets [1,10,100]",
        "metrics": metrics,
        "operating_confidence": confidence,
        "operating_iou": 0.5,
        "small_signs": small,
        "images": len(coco["images"]),
        "gt_instances": len(coco["annotations"]),
        "per_class": rows,
    }, errors


def main():
    parser = argparse.ArgumentParser(description="Standard COCO original-image evaluation")
    parser.add_argument("--coco", type=Path, required=True)
    parser.add_argument(
        "--predictions", type=Path, required=True, help="rs-predict output directory"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.25)
    args = parser.parse_args()
    coco = json.loads(args.coco.read_text())
    meta = json.loads((args.predictions / "prediction.json").read_text())
    if (
        meta["status"] != "complete"
        or meta["gt_sha256"] != file_hash(args.coco)
        or meta["image_ids"] != [im["id"] for im in coco["images"]]
    ):
        raise ValueError("Evaluation requires completed predictions for exactly this GT dataset")
    if not meta["config"]["confidence"] <= args.confidence <= 1:
        raise ValueError("Operating confidence must be between the prediction floor and 1")
    predictions = json.loads((args.predictions / "predictions.coco.json").read_text())
    report, errors = evaluate(coco, predictions, args.confidence)
    report["prediction_provenance"] = meta
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "metrics.json", report)
    write_json(args.output / "errors.json", errors)
    with (args.output / "per_class.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=report["per_class"][0].keys())
        writer.writeheader()
        writer.writerows(report["per_class"])


if __name__ == "__main__":
    main()
