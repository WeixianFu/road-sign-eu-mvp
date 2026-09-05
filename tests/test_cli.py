import json
import subprocess
import sys
from pathlib import Path

import yaml

from roadsigns.common import file_hash, write_json

ROOT = Path(__file__).resolve().parents[1]


def command(module, *args, success=True):
    result = subprocess.run(
        [sys.executable, "-m", module, *map(str, args)], cwd=ROOT, capture_output=True, text=True
    )
    assert (result.returncode == 0) == success, result.stdout + result.stderr
    return result


def test_image_preparation_to_analysis_through_cli(source_dataset):
    prepared = Path(source_dataset["output_root"])
    config = prepared.parent / "prepare.yaml"
    config.write_text(yaml.safe_dump(source_dataset))
    command("roadsigns.image_tools.prepare", "--config", config)
    command(
        "roadsigns.analysis.report",
        "dataset",
        "--manifest",
        prepared / "manifest.json",
        "--output",
        prepared.parent / "dataset-report",
    )
    assert (prepared.parent / "dataset-report/class_counts.csv").is_file()
    source = Path(source_dataset["source_root"]) / "train_full"
    command(
        "roadsigns.image_tools.preview",
        "--image",
        source / "images/fully_train_A.jpg",
        "--labels",
        source / "labels/fully_train_A.txt",
        "--ontology",
        source_dataset["ontology"],
        "--tiles",
        "--output",
        prepared.parent / "preview.png",
    )
    assert (prepared.parent / "preview.png").is_file()
    coco = prepared / "val.coco.json"
    images = json.loads(coco.read_text())["images"]
    predictions = prepared.parent / "empty-predictions"
    predictions.mkdir()
    write_json(
        predictions / "prediction.json",
        {
            "status": "complete",
            "gt_sha256": file_hash(coco),
            "image_ids": [image["id"] for image in images],
            "config": {"confidence": 0.001},
        },
    )
    write_json(predictions / "predictions.coco.json", [])
    report = prepared.parent / "evaluation"
    command(
        "roadsigns.evaluation.metrics",
        "--coco",
        coco,
        "--predictions",
        predictions,
        "--output",
        report,
    )
    assert json.loads((report / "metrics.json").read_text())["metrics"]["AP"] == 0
    command(
        "roadsigns.analysis.report",
        "errors",
        "--coco",
        coco,
        "--errors",
        report / "errors.json",
        "--output",
        report / "gallery",
    )
    assert len(list((report / "gallery").glob("*.jpg"))) == 1
    failed = command(
        "roadsigns.evaluation.metrics",
        "--coco",
        coco,
        "--predictions",
        predictions,
        "--output",
        report / "invalid",
        "--confidence",
        "0.0001",
        success=False,
    )
    assert "prediction floor" in failed.stderr
