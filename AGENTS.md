# Yolo Trafiic Sign Detection Dvelopment Guide

## Project Overview
EU-only road-sign detection using custom-trained YOLOv8 on MTSD.
Inputs: dashcam videos (target 1440p/30 fps; support mixed resolutions/VFR). Decode via PyAV/FFmpeg and compute true timestamps from PTS/time_base. Detection: YOLOv8 (train on MTSD; keep YOLOv11 as a later A/B). Phase 1 excludes SAHI and ByteTrack—focus on getting end-to-end training/inference working. Validate on MTSD val (and optionally cross-check on GTSDB). Export NDJSON/Parquet with timestamp, class, confidence, bbox, and provenance fields.

Love the repo layout—here’s a tight **Tech Stack** and an updated **Project Structure** you can paste into README/AGENTS.md. I’ve kept Phase-1 focused on training/inference with YOLOv8 on MTSD; SAHI/ByteTrack/OCR are noted as Phase-2 add-ons.

## Tech Stack (Phase-1)
* **Detector:** Ultralytics **YOLOv8** (train on MTSD; keep a flag to A/B YOLO11 later).
* **Video I/O & timestamps:** **PyAV** on top of FFmpeg; compute timestamps from **`frame.pts * stream.time_base`** (VFR-safe). 
* **Dataset:** **MTSD** (primary). Optionally sanity-check on **GTSDB** later. 
* **(Phase-2) Small objects:** **SAHI** tiled inference. 
* **(Phase-2) Tracking:** **ByteTrack** for stable `track_id`. 
* **(Phase-2) OCR:** **PaddleOCR** (multilingual EU scripts). 

## Project Structure (what each folder owns)

```
configs/
  data.yaml           # MTSD paths, class names
  train.yaml          # YOLOv8 hyperparams (imgsz, batch, epochs)
  infer.yaml          # thresholds, NMS, device
data/
  mtsd/               # MTSD in YOLO format (images/{train,val}, labels/{train,val})
  sample/             # tiny EU samples for smoke tests
docs/
  datasets.md         # MTSD notes, license, splits
  ontology.md         # class mapping notes (MTSD→canonical)
src/
  io/                 # video reader using PyAV (PTS → ms), image loaders
  detect/             # YOLOv8 train/infer wrappers (phase-1)
  export/             # NDJSON/Parquet writers + schema check
  ocr/                # (phase-2) PaddleOCR crops
  post/               # (phase-2) eventization, smoothing
  track/              # (phase-2) ByteTrack adapter
tests/
  test_io_timestamps.py  # asserts PTS/time_base timing
  test_schema.py         # validates output records
AGENTS.md             # module “agents” contracts (see below)
README.md
```
## Development Guide — EU-only Road-Sign Detection (YOLOv8 on MTSD)

### 1) Scope & milestones

**Goal (Phase-1 / P0):** train and run **YOLOv8** on **MTSD**, then perform video inference with true timestamps and export results. Defer SAHI/ByteTrack/OCR to Phase-2.

* **P0 (this week):** dataset in YOLO format → YOLOv8 train/val → video inference via PyAV/FFmpeg with PTS-based timestamps → NDJSON/Parquet export → smoke tests.
* **P1 (next):** add SAHI (tiny/far signs), ByteTrack (stable track_id), optional OCR (textual signs), A/B YOLOv8 vs YOLOv11.

Ultralytics training/CLI docs are your primary reference for P0. 

### 2) Prerequisites

* **Python 3.12**, CUDA toolchain for your GPU （5070Ti 16GB）, **PyTorch**.
* **FFmpeg** available on PATH (PyAV uses it).
* Key libs: `ultralytics`, `av` (PyAV), `opencv-python`, `pandas`, `pyarrow` (Parquet).

Ultralytics provides the `yolo` CLI for training/val/inference. 

---

### 3) Dataset: MTSD (Mapillary Traffic Sign Dataset)

* Use MTSD as the **primary EU-centric dataset**. Note that MTSD is distributed under **CC BY-NC-SA** (non-commercial) — keep a license note in `docs/datasets.md`. ([mapillary.com][3])

### 3.1 Convert to YOLO format

If your MTSD annotations are COCO-style JSON, convert them to YOLO with the **Ultralytics converter**:

* Python API: `ultralytics.data.converter.convert_coco(...)` (COCO→YOLO). 
* Alternatives: Ultralytics **JSON2YOLO** toolkit or third-party COCO→YOLO utilities. 

Organize as:

```
datasets/mtsd-yolo/
  images/{train,val}/...jpg
  labels/{train,val}/...txt
  mtsd.yaml   # data config with paths and class names
```

---

### 4) Training (YOLOv8)

Start with a medium/small model (vRAM-friendly) and imgsz 640; tune later.

* **Docs:** YOLO training quickstart & reference.
* **CLI syntax:** `yolo TASK MODE ARGS` (e.g., `yolo detect train ...`). 

**Checklist**

1. `mtsd.yaml` points to `images/{train,val}` & class list.
2. Run a tiny “overfit a batch” (e.g., `epochs=1`, small subset) to sanity-check labels.
3. Baseline train run: record mAP on MTSD val; save `runs/detect/train*/results.csv`.

---

### 5) Inference on videos with true timestamps

Use **PyAV** to decode frames and compute timestamps from **PTS × time_base** (don’t assume constant fps).

* PyAV docs explain **time_base** and frame **pts** semantics; ffmpeg docs describe PTS/DTS. 

**Contract for your reader module**

* Yield `(frame_bgr, timestamp_ms, (W,H))` where `timestamp_ms = round(frame.pts * stream.time_base * 1000)`.
* Handle VFR and missing timestamps robustly (skip or warn). PyAV known discussions on time_base pitfalls can help debugging. 

---

### 6) Outputs & data formats

Produce both:

* **NDJSON** (one JSON object per line; great for streaming/debugging). 
* **Parquet** (analytics-friendly columnar storage via PyArrow). 

**Frame-level schema (P0)**
`video_id, frame_idx, timestamp_ms, bbox_xyxy, bbox_norm, class_id, class_name, confidence, model_name, model_version, imgsz`
*(Tracking/OCR fields land in P1.)*

---

### 7) Quality gates (P0)

* **Data sanity:** spot-check a few images: label txt ↔ image size consistent.
* **Train sanity:** non-NaN loss, improving mAP on MTSD val. 
* **Inference sanity:** timestamps strictly increasing per stream (from PTS); bboxes match original resolution; outputs pass NDJSON/Parquet schema checks. 

---

### 8) Phase-2 roadmap (when P0 is stable)

* **SAHI** (Slicing Aided Hyper Inference) for tiny/far signs; integrates with Ultralytics. Expect recall gains at cost of throughput; make it a config toggle. 
* **ByteTrack** for persistent `track_id` & eventization (per-sign start/end times). Use the official repo as reference. 
* **OCR** for textual signs (street/direction boards): **PaddleOCR** multilingual models. 
* Optional **YOLOv11** A/B once P0 is reproducible (Ultralytics guides list v11 topics & SAHI doc). 

---

### 9) Reproducibility & logging

* Save: exact dataset split, `mtsd.yaml`, training args, model hash, environment (pip freeze), and seed.
* Store training curves and confusion matrix from Ultralytics runs directory for later comparison. 

---

### 10) Common pitfalls & fixes

* **Bad timestamps / drift:** ensure you use **frame.pts × stream.time_base**; don’t trust `CAP_PROP_POS_MSEC`. 
* **Label mismatch:** confirm YOLO txt normalized coords match image sizes post-conversion (re-run COCO→YOLO if needed). 
* **Large outputs:** prefer **Parquet** for analytics; keep NDJSON for logs/samples. 

