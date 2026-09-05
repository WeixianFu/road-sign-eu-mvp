import json
import shutil
from pathlib import Path

import pytest
from PIL import Image

from roadsigns.common import read_yaml
from roadsigns.image_tools.prepare import prepare, process_image
from roadsigns.ontology import Ontology


def test_reproducible_preparation_keeps_negatives_and_original_gt(source_dataset):
    first = prepare(source_dataset)
    out = Path(source_dataset["output_root"])
    data = read_yaml(out / "data.yaml")
    gt = json.loads((out / "val.coco.json").read_text())
    assert data["dataset_fingerprint"] == first["fingerprint"]
    assert first["summary"]["train"]["kinds"] == {"tile": 3, "full": 2}
    assert (out / "labels/train/fully_train_B__tile_000.txt").read_text() == ""
    assert len(gt["images"]) == 2
    assert len(gt["annotations"]) == 1
    assert gt["annotations"][0]["bbox"] == pytest.approx([100, 100, 22, 22])
    assert all(r["geography"] == "unverified" for r in first["records"])
    source_dataset["output_root"] = str(out.with_name("prepared-again"))
    second = prepare(source_dataset)
    assert second["fingerprint"] == first["fingerprint"]


def test_missing_label_is_not_a_negative(source_dataset):
    (Path(source_dataset["source_root"]) / "val/labels/fully_val_D.txt").unlink()
    with pytest.raises(ValueError, match="missing is not negative"):
        prepare(source_dataset)


def test_same_pixels_cannot_cross_splits(source_dataset):
    root = Path(source_dataset["source_root"])
    shutil.copy(root / "train_full/images/fully_train_B.jpg", root / "val/images/fully_val_D.jpg")
    with pytest.raises(ValueError, match="content crosses"):
        prepare(source_dataset)
    assert not (Path(source_dataset["output_root"]) / "data.yaml").exists()


def test_unknown_source_id_order_is_rejected(source_dataset):
    path = Path(source_dataset["source_root"]) / "classes.json"
    names = json.loads(path.read_text())
    names["names"].reverse()
    path.write_text(json.dumps(names))
    with pytest.raises(ValueError, match="ID order"):
        prepare(source_dataset)


def test_review_labels_hold_entire_images_out(source_dataset):
    path = Path(source_dataset["source_root"]) / "train_full/labels/fully_train_A.txt"
    path.write_text(path.read_text() + "304 0.5 0.5 0.1 0.1\n")
    manifest = prepare(source_dataset)
    record = next(r for r in manifest["records"] if r["source_id"] == "A")
    assert record["status"] == "review"
    assert record["samples"] == []
    assert manifest["summary"]["train"]["original_images"] == 1


def test_full_view_is_omitted_when_it_would_erase_a_tiny_positive(source_dataset):
    cfg = source_dataset
    source = Path(cfg["source_root"]) / "train_full"
    image = source / "images/fully_train_A.jpg"
    Image.new("RGB", (3000, 2000)).save(image)
    (source / "labels/fully_train_A.txt").write_text(f"8 0.5 0.5 {22 / 3000} {22 / 2000}\n")
    out = Path(cfg["output_root"])
    (out / "images/train").mkdir(parents=True)
    (out / "labels/train").mkdir(parents=True)
    ontology = Ontology(cfg["ontology"])
    result = process_image(("train", str(image), None), cfg, ontology.mapping, ontology.review_ids)
    assert result["full_status"] == "tiny_target"
    assert {s["kind"] for s in result["samples"]} == {"tile"}


def test_geography_allowlist_is_independent_from_the_ontology(source_dataset, tmp_path):
    allowlist = tmp_path / "europe.csv"
    allowlist.write_text("source_id,country\nA,GB\nC,CH\n")
    source_dataset["geography_allowlist"] = str(allowlist)
    manifest = prepare(source_dataset)
    assert {r["source_id"] for r in manifest["records"]} == {"A", "C"}
    assert len(manifest["geography_excluded"]) == 2
    assert {r["country"] for r in manifest["records"]} == {"GB", "CH"}
