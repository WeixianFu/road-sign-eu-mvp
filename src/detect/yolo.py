from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from ultralytics import YOLO


def train(
    model: str | Path,
    data: str | Path,
    project: str = "runs/mtsd",
    name: str = "yolov8_401cls",
    overrides: Dict[str, Any] | None = None,
) -> None:
    """Train a YOLOv8 model with the given Ultralytics overrides."""
    overrides = overrides or {}
    yolo = YOLO(str(model))
    yolo.train(data=str(data), project=project, name=name, **overrides)
    yolo.val(data=str(data), batch=8)


def parse_overrides(pairs: list[str]) -> Dict[str, Any]:
    """Parse CLI key=value pairs into a dict with best-effort typing."""
    parsed: Dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise argparse.ArgumentTypeError(f"Expected KEY=VALUE, got '{pair}'")
        key, value = pair.split("=", 1)
        if value.isdigit():
            parsed[key] = int(value)
            continue
        try:
            parsed[key] = float(value)
            continue
        except ValueError:
            pass
        if value.lower() in {"true", "false"}:
            parsed[key] = value.lower() == "true"
        else:
            parsed[key] = value
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLOv8 training wrapper for MTSD.")
    parser.add_argument("--model", required=True, help="Base model weight, e.g. yolov8m.pt")
    parser.add_argument("--data", required=True, help="Dataset YAML, e.g. configs/mtsd.yaml")
    parser.add_argument("--project", default="runs/mtsd", help="Ultralytics project directory")
    parser.add_argument("--name", default="yolov8_401cls", help="Run name under the project directory")
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Additional Ultralytics overrides, e.g. epochs=100 imgsz=640 batch=16",
    )
    args = parser.parse_args()
    overrides = parse_overrides(args.override)
    train(args.model, args.data, project=args.project, name=args.name, overrides=overrides)


if __name__ == "__main__":
    main()
