from __future__ import annotations

import argparse
import json
from pathlib import Path

from roadsigns.common import environment, event, file_hash, read_yaml, setup_logging, write_json


def training_config(path, profile):
    cfg = read_yaml(path)
    profiles = cfg.pop("profiles")
    cfg.update(profiles[profile])
    required = {
        "imgsz": 1280,
        "scale": 0.1,
        "mosaic": 1.0,
        "close_mosaic": 10,
        "fliplr": 0.0,
        "flipud": 0.0,
        "degrees": 0.0,
        "translate": 0.0,
        "mixup": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "multi_scale": False,
        "copy_paste": 0.0,
        "cutmix": 0.0,
    }
    for key, value in required.items():
        if cfg[key] != value:
            raise ValueError(f"The tiny-sign training recipe requires {key}={value}")
    return cfg


def train(data_path, config_path, profile, output, resume=None):
    import torch
    from ultralytics import YOLO

    from roadsigns.training.trainer import TrafficTrainer

    data_path = Path(data_path).resolve()
    data = read_yaml(data_path)
    dataset_dir = data_path.parent
    manifest = json.loads((dataset_dir / "manifest.json").read_text())
    ontology = json.loads((dataset_dir / "ontology.json").read_text())
    if (
        data["dataset_fingerprint"] != manifest["fingerprint"]
        or data["ontology_fingerprint"] != ontology["fingerprint"]
        or data["names"] != ontology["names"]
    ):
        raise ValueError("Prepared data, manifest and ontology do not agree")
    cfg = training_config(config_path, profile)
    output = Path(output).resolve()
    if resume is None:
        output.mkdir(parents=True, exist_ok=False)
    else:
        resume = Path(resume).resolve()
        previous = json.loads((resume.parents[2] / "run.json").read_text())
        if (
            previous["dataset_fingerprint"] != manifest["fingerprint"]
            or output != resume.parents[2]
            or str(data_path) != previous["data"]
        ):
            raise ValueError("Resume requires the same dataset and original run directory")
        fixed = set(cfg) - {"device", "batch", "model"}
        if any(cfg[key] != previous["config"][key] for key in fixed):
            raise ValueError(
                "Resume keeps the saved training recipe; only device and batch may change"
            )
    logger = setup_logging(output / "train.log")
    model_path = cfg.pop("model")
    model = YOLO(str(resume) if resume else model_path)
    metadata = {
        "dataset_fingerprint": manifest["fingerprint"],
        "ontology": ontology,
        "config": cfg,
        "model": model_path,
        "data": str(data_path),
        "initial_weights_sha256": file_hash(model.ckpt_path),
        "environment": environment(),
        "cuda": torch.version.cuda,
        "gpus": [
            {
                "name": torch.cuda.get_device_name(i),
                "capability": list(torch.cuda.get_device_capability(i)),
                "memory_bytes": torch.cuda.get_device_properties(i).total_memory,
            }
            for i in range(torch.cuda.device_count())
        ],
    }
    if resume is None:
        write_json(output / "run.json", metadata)
    event(output / "operations.jsonl", "resume" if resume else "start", **metadata)
    logger.info("Training %s on %s; dataset=%s", model_path, cfg["device"], manifest["fingerprint"])
    model.train(
        trainer=TrafficTrainer,
        data=str(data_path),
        project=str(output),
        name="model",
        exist_ok=True,
        resume=str(resume) if resume else False,
        **cfg,
    )
    event(
        output / "operations.jsonl",
        "complete",
        best_sha256=file_hash(output / "model/weights/best.pt"),
    )
    logger.info("Training complete. Evaluate original images with rs-predict and rs-evaluate.")


def main():
    parser = argparse.ArgumentParser(description="Train on prepared MTSD tiles and safe full views")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--profile", choices=["desktop", "server"], default="desktop")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    train(args.data, args.config, args.profile, args.output, args.resume)


if __name__ == "__main__":
    main()
