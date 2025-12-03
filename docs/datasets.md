# MTSD Dataset Documentation

## Overview

The **Mapillary Traffic Sign Dataset (MTSD)** is used as the primary dataset for training and validation of the EU road sign detection model.

## License

**Important**: MTSD is distributed under **CC BY-NC-SA** (Creative Commons Attribution-NonCommercial-ShareAlike) license. This means:
- ✅ Non-commercial use is allowed
- ❌ Commercial use requires permission
- ✅ Modifications and sharing are allowed under the same license

For more information, visit: [mapillary.com](https://www.mapillary.com/)

## Dataset Structure

The MTSD dataset is located at: `/Users/weixianfu/Documents/Datas/mtsd`

### Directory Structure

```
mtsd/
├── classes.json              # Class name mappings (401 classes)
├── train_full/               # Full training set
│   ├── images/              # Training images
│   └── labels/               # YOLO format labels
├── train_partial/           # Partial training set (subset)
│   ├── images/
│   └── labels/
├── val/                     # Validation set
│   ├── images/              # Validation images
│   └── labels/               # Validation labels
└── sample/                  # Sample data for testing
    ├── images/
    └── labels/
```

## Class Information

The dataset contains **401 classes** of EU road signs, covering:
- Regulatory signs
- Warning signs
- Information signs
- Complementary signs

Class names follow the MTSD naming convention (e.g., `regulatory--stop--g1`, `warning--pedestrians-crossing--g10`).

The complete class list is stored in `classes.json` at the dataset root.

## Data Format

- **Images**: JPEG format
- **Labels**: YOLO format (normalized coordinates)
  - Format: `class_id x_center y_center width height`
  - Coordinates are normalized (0.0 - 1.0)

## Usage

The dataset is configured in `configs/data.yaml`:

```yaml
path: /Users/weixianfu/Documents/Datas/mtsd
train: train_full/images
val: val/images
```

## Validation

Before training, ensure:
1. Image and label files are paired correctly
2. Label coordinates match image dimensions
3. Class IDs are consistent with `classes.json`
