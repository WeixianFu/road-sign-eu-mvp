import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from roadsigns.common import file_hash, read_yaml, write_json
from roadsigns.evaluation.metrics import evaluate, working_point
from roadsigns.evaluation.predict import predict, predict_image
from roadsigns.image_tools.prepare import prepare

ROOT = Path(__file__).resolve().parents[1]


def toy_coco():
    return {
        "info": {},
        "images": [{"id": 1, "width": 600, "height": 600}],
        "categories": [
            {"id": 0, "name": "common"},
            {"id": 1, "name": "rare"},
            {"id": 2, "name": "absent"},
        ],
        "annotations": [
            {
                "id": i + 1,
                "image_id": 1,
                "category_id": 0 if i < 9 else 1,
                "bbox": [40 * i, 50, 22, 22],
                "area": 484,
                "iscrowd": 0,
            }
            for i in range(10)
        ],
    }


def test_coco_averages_classes_not_pooled_detections():
    coco = toy_coco()
    predictions = [
        {"image_id": 1, "category_id": 0, "bbox": gt["bbox"], "score": 0.9}
        for gt in coco["annotations"]
        if gt["category_id"] == 0
    ]
    report, errors = evaluate(coco, predictions)
    assert report["metrics"]["AP50"] == pytest.approx(0.5)
    assert report["metrics"]["AP"] == pytest.approx(0.5)
    assert report["per_class"][2]["AP"] is None
    assert report["per_class"][1]["recall"] == 0
    assert report["small_signs"]["recall"] == pytest.approx(0.9)
    assert [e["kind"] for e in errors] == ["FN"]


def test_empty_predictions_are_valid_and_score_zero_when_gt_exists():
    report, errors = evaluate(toy_coco(), [])
    assert report["metrics"]["AP"] == 0
    assert len(errors) == 10


def test_working_point_counts_duplicates_and_wrong_classes_as_false_positives():
    coco = toy_coco()
    prediction = {"image_id": 1, "category_id": 0, "bbox": [0, 50, 22, 22], "score": 0.9}
    rows, errors, _ = working_point(
        coco, [prediction, prediction, {**prediction, "category_id": 2}], 0.25
    )
    assert (rows[0]["tp"], rows[0]["fp"], rows[0]["fn"]) == (1, 1, 8)
    assert rows[2]["fp"] == 1
    assert sum(e["kind"] == "FP" for e in errors) == 2


def test_tiled_predictions_restore_coordinates_and_merge_by_class():
    cfg = read_yaml(ROOT / "configs/predict.yaml")

    def backend(tiles):
        assert len(tiles) == 2
        assert all(tile.shape == (1280, 1280, 3) for tile in tiles)
        return [
            np.array([[1000, 100, 1022, 122, 0.9, 0]], float),
            np.array([[280, 100, 302, 122, 0.8, 0], [280, 100, 302, 122, 0.7, 1]], float),
        ]

    predictions = predict_image(Image.new("RGB", (2000, 900)), backend, cfg)
    assert len(predictions) == 2
    assert predictions[0]["bbox"] == [1000, 100, 22, 22]
    assert [p["category_id"] for p in predictions] == [0, 1]


def test_predictions_in_padding_are_removed():
    cfg = read_yaml(ROOT / "configs/predict.yaml")

    def backend(_):
        return [np.array([[700, 100, 720, 120, 0.9, 0]], float)]

    assert predict_image(Image.new("RGB", (640, 480)), backend, cfg) == []


def test_prediction_resume_commits_empty_images_and_prevents_duplicate_results(source_dataset):
    prepare(source_dataset)
    prepared = Path(source_dataset["output_root"])
    run = prepared.parent / "run"
    (run / "model/weights").mkdir(parents=True)
    weights = run / "model/weights/best.pt"
    weights.write_bytes(b"synthetic model identity; no model execution")
    write_json(run / "run.json", {"ontology": json.loads((prepared / "ontology.json").read_text())})
    cfg = read_yaml(ROOT / "configs/predict.yaml")
    output = prepared.parent / "predictions"
    calls = []

    def interrupted(tiles):
        calls.append(len(tiles))
        if len(calls) == 2:
            raise RuntimeError("simulated interruption")
        return [np.empty((0, 6)) for _ in tiles]

    with pytest.raises(RuntimeError, match="interruption"):
        predict(prepared / "val.coco.json", weights, cfg, output, backend=interrupted)
    assert len(list((output / "parts").glob("*.json"))) == 1
    completed_calls = []

    def completed(tiles):
        completed_calls.append(len(tiles))
        return [np.empty((0, 6)) for _ in tiles]

    result = predict(prepared / "val.coco.json", weights, cfg, output, True, completed)
    assert result == []
    assert len(completed_calls) == 1
    meta = json.loads((output / "prediction.json").read_text())
    assert meta["status"] == "complete" and meta["images"] == 2
    assert meta["model_sha256"] == file_hash(weights)
    with pytest.raises(ValueError, match="unchanged"):
        predict(
            prepared / "val.coco.json", weights, {**cfg, "merge_iou": 0.4}, output, True, completed
        )
