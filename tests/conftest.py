import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from roadsigns.common import read_yaml
from roadsigns.ontology import Ontology

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def source_dataset(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    ontology = Ontology(ROOT / "configs/ontology.csv")
    (source / "classes.json").write_text(
        json.dumps({"names": [r["source_name"] for r in ontology.rows]})
    )
    for split in ("train_full", "val"):
        (source / split / "images").mkdir(parents=True)
        (source / split / "labels").mkdir()
    samples = [
        ("train_full", "fully_train_A", (1500, 1000), (30, 50, 80), True),
        ("train_full", "fully_train_B", (640, 480), (80, 50, 30), False),
        ("val", "fully_val_C", (700, 500), (20, 50, 30), True),
        ("val", "fully_val_D", (800, 600), (30, 20, 80), False),
    ]
    for split, stem, size, color, positive in samples:
        image = Image.new("RGB", size, color)
        if positive:
            ImageDraw.Draw(image).rectangle((100, 100, 122, 122), fill="red")
        image.save(source / split / "images" / f"{stem}.jpg")
        text = (
            f"8 {111 / size[0]} {111 / size[1]} {22 / size[0]} {22 / size[1]}\n" if positive else ""
        )
        (source / split / "labels" / f"{stem}.txt").write_text(text)
    cfg = read_yaml(ROOT / "configs/data.yaml")
    cfg.update(
        source_root=str(source),
        output_root=str(tmp_path / "prepared"),
        ontology=str(ROOT / "configs/ontology.csv"),
        workers=1,
    )
    return cfg
