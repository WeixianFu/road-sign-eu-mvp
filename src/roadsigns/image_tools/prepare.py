from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import yaml
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
from roadsigns.image_tools.geometry import clip_boxes, letterbox, padded_tile, windows
from roadsigns.image_tools.labels import read_labels, write_labels
from roadsigns.ontology import Ontology


def source_id(stem):
    return re.sub(r"^(fully|partially)_(train|val|test)_", "", stem)


def process_image(task, cfg, mapping, review_ids):
    split, image_path, country = task
    image_path = Path(image_path)
    label_path = image_path.parent.parent / "labels" / (image_path.stem + ".txt")
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    source_classes, source_boxes = read_labels(label_path, width, height, 401)
    record = {
        "source_id": source_id(image_path.stem),
        "source_path": str(image_path),
        "split": split,
        "width": width,
        "height": height,
        "image_sha256": file_hash(image_path),
        "label_sha256": file_hash(label_path),
        "country": country,
        "geography": "allowlisted" if country else "unverified",
        "source_counts": dict(Counter(map(int, source_classes))),
    }
    pending = sorted(set(source_classes.tolist()) & review_ids)
    if pending:
        return {**record, "status": "review", "review_classes": pending, "samples": []}
    keep = np.array([int(cls) in mapping for cls in source_classes], dtype=bool)
    classes = np.array([mapping[int(c)] for c in source_classes[keep]], dtype=int)
    boxes = source_boxes[keep]
    output = Path(cfg["output_root"])
    tile = cfg["tiling"]
    samples = []

    def save(name, canvas, local_classes, local_boxes, kind, **geometry):
        image_file = output / "images" / split / (name + ".jpg")
        label_file = output / "labels" / split / (name + ".txt")
        canvas.save(image_file, quality=95, subsampling=0)
        write_labels(label_file, local_classes, local_boxes, tile["size"])
        samples.append(
            {
                "image": str(image_file.relative_to(output)),
                "kind": kind,
                "counts": dict(Counter(map(int, local_classes))),
                "image_sha256": file_hash(image_file),
                "label_sha256": file_hash(label_file),
                **geometry,
            }
        )

    for index, window in enumerate(windows(width, height, tile["size"], tile["overlap"])):
        clipped, mask = clip_boxes(boxes, window, tile["min_area"], tile["min_visibility"])
        local = clipped - np.array(window[:2] * 2)
        save(
            f"{image_path.stem}__tile_{index:03d}",
            padded_tile(image, window, tile["size"]),
            classes[mask],
            local,
            "tile",
            roi=window,
        )

    full_status = "disabled"
    if cfg["full_images"]["enabled"] and split == "train":
        canvas, full_boxes, geometry = letterbox(image, boxes, tile["size"])
        # Omit the full view when resizing would erase a labelled target.
        if ((full_boxes[:, 2:] - full_boxes[:, :2]) >= cfg["full_images"]["min_side"]).all():
            save(f"{image_path.stem}__full", canvas, classes, full_boxes, "full", **geometry)
            full_status = "kept"
        else:
            full_status = "tiny_target"
    return {
        **record,
        "status": "prepared",
        "classes": classes.tolist(),
        "boxes": boxes.tolist(),
        "samples": samples,
        "full_status": full_status,
    }


def prepare(cfg):
    source, output = Path(cfg["source_root"]), Path(cfg["output_root"])
    ontology = Ontology(cfg["ontology"])
    ontology.check_source_names(json.loads((source / "classes.json").read_text())["names"])
    if cfg["splits"] != {"train": "train_full", "val": "val"}:
        raise ValueError(
            "This training stage requires the fully labelled train_full and val splits"
        )
    if cfg["tiling"]["size"] != 1280 or cfg["tiling"]["overlap"] != 0.2:
        raise ValueError("Preparation requires 1280-pixel tiles with 20% overlap")
    allowlist = None
    if cfg["geography_allowlist"]:
        with Path(cfg["geography_allowlist"]).open(newline="", encoding="utf-8") as stream:
            allowlist = {row["source_id"]: row["country"] for row in csv.DictReader(stream)}
        if any(not value for value in allowlist.values()):
            raise ValueError("Every allowlisted image requires an explicitly supplied country")

    tasks, identities = [], set()
    excluded = []
    for split, source_split in cfg["splits"].items():
        image_dir = source / source_split / "images"
        label_dir = source / source_split / "labels"
        images = sorted(p for p in image_dir.glob("*.jpg") if not p.name.startswith("._"))
        if not images:
            raise ValueError(f"No source JPEG images in {image_dir}")
        labels = {p.stem for p in label_dir.glob("*.txt") if not p.name.startswith("._")}
        if {p.stem for p in images} != labels:
            raise ValueError(f"Image/label pairs differ in {source_split}; missing is not negative")
        for image_path in images:
            identity = source_id(image_path.stem)
            if identity in identities:
                raise ValueError(f"Repeated original-image ID across dataset: {identity}")
            identities.add(identity)
            if allowlist is not None and identity not in allowlist:
                excluded.append({"source_id": identity, "split": split, "reason": "geography"})
                continue
            country = allowlist[identity] if allowlist is not None else None
            tasks.append((split, str(image_path), country))
    output.mkdir(parents=True, exist_ok=False)
    log = setup_logging(output / "prepare.log")
    write_json(output / "recipe.json", cfg)
    write_json(output / "environment.json", environment())
    write_json(output / "ontology.json", ontology.snapshot())
    for split in cfg["splits"]:
        for kind in ("images", "labels"):
            (output / kind / split).mkdir(parents=True)
    worker = partial(
        process_image, cfg=cfg, mapping=ontology.mapping, review_ids=ontology.review_ids
    )
    records = []
    with ProcessPoolExecutor(max_workers=cfg["workers"]) as pool:
        for record in pool.map(worker, tasks, chunksize=1):
            records.append(record)
            event(output / "operations.jsonl", "image", **record)
            if len(records) % 100 == 0:
                log.info("Prepared %d / %d source images", len(records), len(tasks))
    hashes = {}
    for record in records:
        image_hash = record["image_sha256"]
        if image_hash in hashes and hashes[image_hash] != record["split"]:
            raise ValueError(
                f"Identical source image content crosses train/val: {record['source_id']}"
            )
        hashes[image_hash] = record["split"]
    prepared = [r for r in records if r["status"] == "prepared"]
    summary = {}
    for split in cfg["splits"]:
        selected = [r for r in prepared if r["split"] == split]
        if not selected:
            raise ValueError(f"No eligible source images remain in {split}")
        samples = [s for r in selected for s in r["samples"]]
        counts = Counter()
        for record in selected:
            counts.update(record["classes"])
        summary[split] = {
            "original_images": len(selected),
            "samples": len(samples),
            "kinds": dict(Counter(s["kind"] for s in samples)),
            "original_class_counts": {i: counts[i] for i in range(len(ontology.names))},
        }
    stable_recipe = {k: cfg[k] for k in ("tiling", "full_images", "training", "splits")}
    fingerprint = digest(
        {
            "recipe": stable_recipe,
            "ontology": ontology.fingerprint,
            "sources": [
                {
                    k: r[k]
                    for k in (
                        "source_id",
                        "split",
                        "image_sha256",
                        "label_sha256",
                        "country",
                        "status",
                    )
                }
                for r in records
            ],
            "sample_hashes": [
                s["image_sha256"] + s["label_sha256"] for r in records for s in r["samples"]
            ],
        }
    )
    manifest = {
        "schema_version": 1,
        "fingerprint": fingerprint,
        "summary": summary,
        "geography_excluded": excluded,
        "recipe": stable_recipe,
        "records": records,
    }
    write_json(output / "manifest.json", manifest)
    val = [r for r in prepared if r["split"] == "val"]
    annotations = []
    images = []
    for image_id, record in enumerate(val, 1):
        images.append(
            {
                "id": image_id,
                "file_name": record["source_path"],
                "width": record["width"],
                "height": record["height"],
                "source_id": record["source_id"],
                "sha256": record["image_sha256"],
            }
        )
        for cls, box in zip(record["classes"], record["boxes"]):
            x1, y1, x2, y2 = box
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": cls,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": (x2 - x1) * (y2 - y1),
                    "iscrowd": 0,
                }
            )
    write_json(
        output / "val.coco.json",
        {
            "info": {
                "dataset_fingerprint": fingerprint,
                "ontology_fingerprint": ontology.fingerprint,
                "scope": "original images from the selected MTSD validation subset",
            },
            "images": images,
            "annotations": annotations,
            "categories": [{"id": i, "name": n} for i, n in enumerate(ontology.names)],
        },
    )
    data = {
        "path": str(output),
        "train": "images/train",
        "val": "images/val",
        "names": ontology.names,
        "nc": len(ontology.names),
        "dataset_fingerprint": fingerprint,
        "ontology_fingerprint": ontology.fingerprint,
        "tiling": cfg["tiling"],
        "tiny_training": cfg["training"],
    }
    (output / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    event(output / "operations.jsonl", "complete", fingerprint=fingerprint, summary=summary)
    log.info("Dataset ready: %s (%s)", output, fingerprint)
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Prepare original 401-class YOLO MTSD images")
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    args = parser.parse_args()
    cfg = read_yaml(args.config)
    for key in ("source_root", "output_root", "ontology", "geography_allowlist"):
        if cfg[key] is not None:
            cfg[key] = str((args.config.resolve().parent / cfg[key]).resolve())
    prepare(cfg)


if __name__ == "__main__":
    main()
