import shutil
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
MIN_ABS_AREA_DEFAULT = 100


def _find_image(image_dir: Path, name: str) -> Path:
    for ext in IMAGE_EXTS:
        cand = image_dir / f"{name}{ext}"
        if cand.exists():
            return cand
    return next(image_dir.glob(f"{name}.*"))


def _read_labels(labels_dir: Path, name: str) -> List[str]:
    path = labels_dir / f"{name}.txt"
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def _prepare_output(output_dir: Path) -> Tuple[Path, Path]:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    return images_dir, labels_dir


def slice_image(img: np.ndarray, slice_size: int, overlap: float) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    h, w = img.shape[:2]
    step = int(slice_size * (1 - overlap))
    slices: List[Tuple[np.ndarray, Tuple[int, int, int, int]]] = []
    y = 0
    while y < h:
        x = 0
        while x < w:
            x2 = min(x + slice_size, w)
            y2 = min(y + slice_size, h)
            x1 = max(0, x2 - slice_size)
            y1 = max(0, y2 - slice_size)
            slices.append((img[y1:y2, x1:x2], (x1, y1, x2, y2)))
            if x2 >= w:
                break
            x += step
        if y2 >= h:
            break
        y += step
    return slices


def _iter_labels(label_lines: Iterable[str]) -> Iterable[Tuple[str, float, float, float, float]]:
    for line in label_lines:
        parts = line.split()
        if len(parts) != 5:
            continue
        cls, cx, cy, bw, bh = parts[0], *map(float, parts[1:])
        yield cls, cx, cy, bw, bh


def _project_labels(label_lines: List[str], image_shape: Tuple[int, int], slice_box: Tuple[int, int, int, int], min_abs_area: int, min_area_ratio: float) -> List[str]:
    h, w = image_shape
    sx1, sy1, sx2, sy2 = slice_box
    sw = sx2 - sx1
    sh = sy2 - sy1
    out: List[str] = []
    for cls, cx, cy, bw, bh in _iter_labels(label_lines):
        bx1 = (cx - bw / 2) * w
        by1 = (cy - bh / 2) * h
        bx2 = (cx + bw / 2) * w
        by2 = (cy + bh / 2) * h
        orig_area = (bx2 - bx1) * (by2 - by1)
        ix1 = max(bx1, sx1)
        iy1 = max(by1, sy1)
        ix2 = min(bx2, sx2)
        iy2 = min(by2, sy2)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        inter_area = (ix2 - ix1) * (iy2 - iy1)
        if inter_area < min_abs_area or inter_area < min_area_ratio * orig_area:
            continue
        rx1 = ix1 - sx1
        ry1 = iy1 - sy1
        rx2 = ix2 - sx1
        ry2 = iy2 - sy1
        rcx = (rx1 + rx2) / 2 / sw
        rcy = (ry1 + ry2) / 2 / sh
        rbw = (rx2 - rx1) / sw
        rbh = (ry2 - ry1) / sh
        out.append(f"{cls} {rcx:.6f} {rcy:.6f} {rbw:.6f} {rbh:.6f}")
    return out


def _letterbox(img: np.ndarray, target: int) -> Tuple[np.ndarray, float, float, float]:
    h, w = img.shape[:2]
    scale = min(target / w, target / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_x = (target - new_w) / 2
    pad_y = (target - new_h) / 2
    top, bottom = int(np.floor(pad_y)), int(np.ceil(pad_y))
    left, right = int(np.floor(pad_x)), int(np.ceil(pad_x))
    out = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return out, scale, pad_x, pad_y


def _write_if_labels(base_name: str, image: np.ndarray, labels: List[str], suffix: str, images_dir: Path, labels_dir: Path) -> None:
    if not labels:
        return
    img_path = images_dir / f"{base_name}{suffix}"
    cv2.imwrite(str(img_path), image)
    label_path = labels_dir / f"{base_name}.txt"
    with open(label_path, "w", encoding="utf-8") as f:
        f.write("\n".join(labels))


def process_image(
    image_dir: str,
    labels_dir: str,
    output_dir: str,
    name: str,
    slice_size: int = 1280,
    overlap: float = 0.2,
    resize_size: int = 1280,
    min_abs_area: int = MIN_ABS_AREA_DEFAULT,
    min_area_ratio: float = 0.6,
) -> None:
    image_dir_path = Path(image_dir)
    labels_dir_path = Path(labels_dir)
    output_dir_path = Path(output_dir)

    image_path = _find_image(image_dir_path, name)
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]
    label_lines = _read_labels(labels_dir_path, name)

    images_out, labels_out = _prepare_output(output_dir_path)

    slices = slice_image(img, slice_size, overlap)
    for idx, (patch, box) in enumerate(slices):
        patch_labels = _project_labels(label_lines, (h, w), box, min_abs_area, min_area_ratio)
        _write_if_labels(f"{name}_{idx}", patch, patch_labels, image_path.suffix, images_out, labels_out)

    resized, scale, pad_x, pad_y = _letterbox(img, resize_size)
    resized_labels: List[str] = []
    for cls, cx, cy, bw, bh in _iter_labels(label_lines):
        abs_cx = cx * w
        abs_cy = cy * h
        abs_bw = bw * w
        abs_bh = bh * h
        abs_cx = abs_cx * scale + pad_x
        abs_cy = abs_cy * scale + pad_y
        abs_bw = abs_bw * scale
        abs_bh = abs_bh * scale
        if abs_bw * abs_bh < min_abs_area:
            continue
        new_cx = abs_cx / resize_size
        new_cy = abs_cy / resize_size
        new_bw = abs_bw / resize_size
        new_bh = abs_bh / resize_size
        resized_labels.append(f"{cls} {new_cx:.6f} {new_cy:.6f} {new_bw:.6f} {new_bh:.6f}")

    _write_if_labels(f"{name}_F", resized, resized_labels, image_path.suffix, images_out, labels_out)
