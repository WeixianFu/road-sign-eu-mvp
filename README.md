# European Traffic Sign Image Training

Prepare data from original MTSD images, train YOLOv8, run tiled prediction on original images, and evaluate results with COCO metrics and false-positive / false-negative analysis. This iteration covers images. Video, tracking, OCR and semi-supervised training are deferred.

The European scope includes Switzerland and the UK. The current ontology has **153 target classes**. Geographic provenance is tracked separately; MTSD has not yet been filtered to verified European images.

```text
configs/                 Data, training and prediction settings; canonical class mapping
src/roadsigns/
  image_tools/           Labels, tiling, dataset preparation and image previews
  training/              70/30 sampling, tiny-safe Mosaic and YOLO training
  evaluation/            Tiled prediction on original images and standard COCO metrics
  analysis/              Class counts, training summaries and FP/FN galleries
tests/                   Synthetic image and adapter tests
tools/                   Full command logs and lightweight checks
docs/refactor/           Audit, operation logs, progress reports and validation records
```

| Device | Workload |
|---|---|
| MacBook | Code review, lightweight tests and reports exported from the desktop |
| Ubuntu / RTX 5070 Ti | Dataset preparation, training, original-image prediction and evaluation |
| Remote Linux server | The same commands, with data paths, device and batch size configured for the server |

The lightweight Mac environment does not require PyTorch:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,evaluate]' -c requirements/constraints.txt
python tools/check.py
```

See the [training guide](docs/training.md) for Ubuntu installation and training. Input uses the original project's **401-class YOLO format with original images**. The new class distinctions cannot be recovered from labels already merged into core146.

Run these commands in order. Each supports `--help`:

1. `rs-prepare`: Convert original images and labels into 1280-pixel tiles, eligible full-image views, original-image ground truth and a dataset fingerprint.
2. `rs-preview`: Inspect source or prepared labels and tile layouts.
3. `rs-train`: Train on the Ubuntu desktop or Linux server.
4. `rs-predict`: Run tiled prediction on original validation images, with recovery from completed images.
5. `rs-evaluate`: Compute COCO AP, per-class metrics, small-sign recall and FP/FN lists.
6. `rs-analyze`: Generate dataset statistics, training summaries and error galleries.

[Design and data contract](docs/design.md) · [Class mapping changes](docs/ontology.md) · [Evaluation and analysis](docs/evaluation.md) · [Refactor audit](docs/refactor/audit.md)

Code checks use synthetic data. Real MTSD processing, RTX 5070 Ti memory usage and throughput, multi-GPU training and model accuracy still require validation on the target hardware. Passing synthetic tests does not mean a model has been trained.
