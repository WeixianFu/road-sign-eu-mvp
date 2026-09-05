from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from roadsigns.common import write_json
from roadsigns.evaluation.metrics import xyxy
from roadsigns.image_tools.preview import draw_boxes


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def dataset_report(manifest_path, output):
    manifest_path, output = Path(manifest_path), Path(output)
    manifest = json.loads(manifest_path.read_text())
    ontology = json.loads((manifest_path.parent / "ontology.json").read_text())
    output.mkdir(parents=True, exist_ok=False)
    counts = {s: manifest["summary"][s]["original_class_counts"] for s in ("train", "val")}
    rows = [
        {
            "class_id": i,
            "class_name": name,
            "train_original_instances": counts["train"][str(i)],
            "val_original_instances": counts["val"][str(i)],
        }
        for i, name in enumerate(ontology["names"])
    ]
    write_csv(output / "class_counts.csv", rows)
    pending = [r for r in manifest["records"] if r["status"] == "review"]
    status = Counter(r["geography"] for r in manifest["records"])
    text = [
        "# Dataset analysis",
        "",
        f"Fingerprint: `{manifest['fingerprint']}`",
        "",
        "Counts below refer to original boxes, before overlapping tiles duplicate them.",
        "",
        "| Split | Original images | Training/validation tiles and full views |",
        "|---|---:|---:|",
    ]
    for split, summary in manifest["summary"].items():
        text.append(f"| {split} | {summary['original_images']} | {summary['samples']} |")
    text += [
        "",
        f"Images held for ambiguous-label review: {len(pending)}.",
        f"Images excluded by geography allowlist: {len(manifest['geography_excluded'])}.",
        f"Geography status: {dict(status)}.",
        "",
        "| Class | Train instances | Val instances |",
        "|---|---:|---:|",
    ]
    for row in sorted(rows, key=lambda r: r["train_original_instances"]):
        text.append(
            f"| {row['class_name']} | {row['train_original_instances']} | "
            f"{row['val_original_instances']} |"
        )
    text += [
        "",
        "Missing validation classes have undefined AP, not an AP of zero.",
        "Small sample counts are a review signal; they do not automatically delete a class.",
    ]
    (output / "report.md").write_text("\n".join(text) + "\n")
    write_json(output / "review_images.json", pending)
    return rows


def training_report(csv_path, output):
    with Path(csv_path).open(newline="") as stream:
        rows = [{k.strip(): float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
    metric = "metrics/mAP50-95(B)"
    finite = [row for row in rows if math.isfinite(row[metric])]
    best = max(finite, key=lambda r: r[metric]) if finite else None
    nonfinite = [
        int(row["epoch"])
        for row in rows
        if any(not math.isfinite(v) for key, v in row.items() if "loss" in key)
    ]
    summary = {
        "epochs_recorded": len(rows),
        "best_tile_epoch": int(best["epoch"]) if best else None,
        "best_tile_AP": best[metric] if best else None,
        "nonfinite_loss_epochs": nonfinite,
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "summary.json", summary)
    text = [
        "# Training analysis",
        "",
        f"Recorded epochs: {len(rows)}.",
        f"Best tile AP: {summary['best_tile_AP']} at epoch {summary['best_tile_epoch']}.",
        f"Non-finite loss epochs: {nonfinite}.",
        "",
        "These are tile-validation metrics. Compare model quality using the separate "
        "original-image COCO report. A single curve does not establish an accuracy ceiling "
        "or the cause of a numerical failure.",
    ]
    (output / "report.md").write_text("\n".join(text) + "\n")
    return summary


def error_gallery(coco_path, errors_path, output, limit=20):
    coco = json.loads(Path(coco_path).read_text())
    errors = json.loads(Path(errors_path).read_text())
    images = {i["id"]: i for i in coco["images"]}
    names = {c["id"]: c["name"] for c in coco["categories"]}
    gt = defaultdict(list)
    for annotation in coco["annotations"]:
        gt[annotation["image_id"]].append(annotation)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    selected = sorted(errors, key=lambda e: (e["kind"], -e.get("score", 0)))[:limit]
    for index, error in enumerate(selected):
        image = Image.open(images[error["image_id"]]["file_name"]).convert("RGB")
        targets = gt[error["image_id"]]
        canvas = draw_boxes(
            image, [t["category_id"] for t in targets], [xyxy(t["bbox"]) for t in targets], names
        )
        canvas = draw_boxes(canvas, [error["category_id"]], [xyxy(error["bbox"])], names, "red")
        canvas.save(output / f"{index:03d}_{error['kind']}_{error['image_id']}.jpg", quality=95)
    write_json(output / "index.json", selected)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze datasets, training curves and FP/FN cases"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    dataset = commands.add_parser("dataset")
    dataset.add_argument("--manifest", type=Path, required=True)
    dataset.add_argument("--output", type=Path, required=True)
    training = commands.add_parser("training")
    training.add_argument("--csv", type=Path, required=True)
    training.add_argument("--output", type=Path, required=True)
    errors = commands.add_parser("errors")
    errors.add_argument("--coco", type=Path, required=True)
    errors.add_argument("--errors", type=Path, required=True)
    errors.add_argument("--output", type=Path, required=True)
    errors.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    if args.command == "dataset":
        dataset_report(args.manifest, args.output)
    elif args.command == "training":
        training_report(args.csv, args.output)
    else:
        error_gallery(args.coco, args.errors, args.output, args.limit)


if __name__ == "__main__":
    main()
