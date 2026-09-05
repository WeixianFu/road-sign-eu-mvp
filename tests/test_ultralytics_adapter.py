import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("ultralytics")

from ultralytics.cfg import get_cfg

from roadsigns.common import read_yaml
from roadsigns.image_tools.prepare import prepare
from roadsigns.training.dataset import HybridDataset
from roadsigns.training.run import training_config

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[1]


def test_real_ultralytics_dataset_format_collation_and_mosaic_close(source_dataset):
    prepare(source_dataset)
    output = Path(source_dataset["output_root"])
    cfg = training_config(ROOT / "configs/train.yaml", "desktop")
    cfg.pop("model")
    hyp = get_cfg(overrides=cfg)
    data = read_yaml(output / "data.yaml")
    dataset = HybridDataset(
        img_path=str(output / "images/train"),
        imgsz=1280,
        batch_size=2,
        augment=True,
        hyp=hyp,
        rect=False,
        cache=False,
        data=data,
        task="detect",
    )
    random.seed(42)
    samples = [dataset[i] for i in range(2)]
    batch = dataset.collate_fn(samples)
    assert tuple(batch["img"].shape) == (2, 3, 1280, 1280)
    assert len(batch["sample_sources"][0]) == 4
    assert (batch["bboxes"] >= 0).all() and (batch["bboxes"] <= 1).all()
    dataset.close_mosaic(hyp)
    assert len(dataset[0]["sample_sources"]) == 1
    assert json.loads((output / "ontology.json").read_text())["names"] == data["names"]


def test_training_entry_and_resume_keep_outer_provenance(source_dataset, monkeypatch):
    import ultralytics

    from roadsigns.training.run import train
    from roadsigns.training.trainer import TrafficTrainer

    prepare(source_dataset)
    data = Path(source_dataset["output_root"]) / "data.yaml"
    output = data.parent.parent / "run"
    initial = data.parent.parent / "initial.pt"
    initial.write_bytes(b"synthetic weights, never loaded")
    calls = []

    class ModelStub:
        def __init__(self, path):
            self.ckpt_path = initial if path == "yolov8m.pt" else Path(path)

        def train(self, **kwargs):
            calls.append(kwargs)
            assert kwargs["trainer"] is TrafficTrainer
            assert (Path(kwargs["project"]) / "run.json").exists()
            native = Path(kwargs["project"]) / kwargs["name"] / "weights"
            native.mkdir(parents=True, exist_ok=True)
            (native / "best.pt").write_bytes(b"synthetic best")
            (native / "last.pt").write_bytes(b"synthetic last")

    monkeypatch.setattr(ultralytics, "YOLO", ModelStub)
    config = ROOT / "configs/train.yaml"
    train(data, config, "desktop", output)
    original_metadata = (output / "run.json").read_bytes()
    checkpoint = output / "model/weights/last.pt"
    train(data, config, "desktop", output, checkpoint)
    assert calls[1]["resume"] == str(checkpoint)
    assert calls[0]["imgsz"] == 1280 and calls[0]["name"] == "model"
    assert (output / "run.json").read_bytes() == original_metadata
    with pytest.raises(ValueError, match="original run directory"):
        train(data, config, "desktop", output.parent / "different", checkpoint)


def test_nonfinite_loss_saves_actual_batch_before_stopping(tmp_path):
    import torch

    from roadsigns.training.trainer import check_loss

    batch = {
        "sample_sources": (["tile_a.jpg", "tile_b.jpg"],),
        "cls": torch.tensor([[1.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.02, 0.02]]),
        "batch_idx": torch.tensor([0.0]),
        "img": torch.full((1, 3, 16, 16), 0.4),
    }
    trainer = SimpleNamespace(
        loss=torch.tensor(float("nan")), save_dir=tmp_path, epoch=3, current_batch=batch
    )
    with pytest.raises(FloatingPointError, match="evidence saved"):
        check_loss(trainer)
    diagnostic = next(tmp_path.glob("nonfinite-*.json"))
    evidence = json.loads(diagnostic.read_text())
    assert evidence["epoch"] == 4
    assert evidence["sources"] == [["tile_a.jpg", "tile_b.jpg"]]
    assert torch.equal(torch.load(diagnostic.with_suffix(".pt")), batch["img"])
