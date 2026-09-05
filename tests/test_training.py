from pathlib import Path

import pytest
import yaml

from roadsigns.analysis.report import training_report
from roadsigns.training.run import training_config

ROOT = Path(__file__).resolve().parents[1]


def test_directional_signs_cannot_silently_use_horizontal_flip(tmp_path):
    cfg = yaml.safe_load((ROOT / "configs/train.yaml").read_text())
    cfg["fliplr"] = 0.5
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="fliplr"):
        training_config(path, "desktop")


def test_desktop_is_a_small_explicit_cuda_batch():
    cfg = training_config(ROOT / "configs/train.yaml", "desktop")
    assert cfg["device"] == 0
    assert cfg["batch"] == 2
    assert "profiles" not in cfg


def test_curve_analysis_records_nan_without_claiming_its_cause(tmp_path):
    path = tmp_path / "results.csv"
    path.write_text(
        "epoch,metrics/mAP50-95(B),train/box_loss,val/box_loss\n"
        "1,0.4,1,1\n2,0.6,0.8,0.9\n3,0.1,nan,nan\n"
    )
    summary = training_report(path, tmp_path / "report")
    assert summary["best_tile_epoch"] == 2
    assert summary["nonfinite_loss_epochs"] == [3]
