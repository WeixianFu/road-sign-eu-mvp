"""
Image slicing module for tiling large images into smaller patches.

Purpose:
    Slice high-resolution images (e.g., 4K 4032x3024) into smaller images (1280x1280)
    with overlap to prevent cutting objects at edges. This is required for YOLOv8
    training on MTSD where small targets (～20px) would be lost if resized directly.

Key parameters:
    - Slice size: 1280x1280 (retains ~2.75px feature response at P3 layer for 20px targets)
    - Overlap: 20% (prevents edge-cutting artifacts)
"""

import cv2
import numpy as np
import shutil
from pathlib import Path
from typing import List, Tuple


def slice_image(
    image_path: str,
    slice_size: int = 1280,
    overlap: float = 0.2
) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """
    Slice a large image into overlapping patches.
    
    Parameters:
    -----------
    image_path : str
        Path to the input image file
    slice_size : int
        Size of each slice (width and height). Default 1280 to retain small target features.
    overlap : float
        Overlap ratio between adjacent slices (0.0-1.0). Default 0.2 (20%) to prevent edge-cutting.
        
    Returns:
    --------
    List[Tuple[np.ndarray, Tuple[int, int, int, int]]]
        List of (slice_image, (x1, y1, x2, y2)) tuples where coordinates are in original image space.
    """
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]
    
    step = int(slice_size * (1 - overlap))
    
    slices = []
    
    y = 0
    while y < h:
        x = 0
        while x < w:
            x2 = min(x + slice_size, w)
            y2 = min(y + slice_size, h)
            
            x1 = max(0, x2 - slice_size)
            y1 = max(0, y2 - slice_size)
            
            slice_img = img[y1:y2, x1:x2]
            
            if slice_img.size > 0:
                slices.append((slice_img, (x1, y1, x2, y2)))
            
            if x2 >= w:
                break
            
            x += step
            if x >= w:
                break
        
        if y2 >= h:
            break
        
        y += step
        if y >= h:
            break
    
    return slices


def slice_labels(
    image_path: str,
    slice_box: Tuple[int, int, int, int],
    image_shape: Tuple[int, int] | None = None,
    label_lines: List[str] | None = None,
    min_abs_area: int = 100,
    min_area_ratio: float = 0.6
) -> List[str]:
    """
    Project original YOLO labels into a sliced patch without saving files.

    Parameters:
    -----------
    image_path : str
        Path to the original image; label file is inferred by stem + ".txt".
    slice_box : (x1, y1, x2, y2)
        Absolute slice box in original image coordinates.
    min_abs_area : int
        Discard boxes whose remaining area after clipping is < min_abs_area pixels.
    min_area_ratio : float
        Discard boxes whose remaining area is < min_area_ratio of original box area.

    Returns:
    --------
    List[str]
        YOLO-format lines (class cx cy w h) normalized to the slice frame.

    Notes:
    - Keeps patch size consistent (uses slice_box width/height for normalization).
    - Filters tiny fragments to avoid noisy labels (per slicing rules).
    """
    if image_shape is None:
        img = cv2.imread(str(image_path))
        h, w = img.shape[:2]
    else:
        h, w = image_shape

    label_path = _resolve_label_path(Path(image_path))
    if label_lines is None:
        if not label_path.exists():
            return []
        with open(label_path, "r", encoding="utf-8") as f:
            label_lines = f.readlines()
    else:
        if not label_lines:
            return []

    sx1, sy1, sx2, sy2 = slice_box
    slice_w = sx2 - sx1
    slice_h = sy2 - sy1

    sliced_labels: List[str] = []

    for line in label_lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        cls = parts[0]
        cx, cy, bw, bh = map(float, parts[1:])

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

        rel_x1 = ix1 - sx1
        rel_y1 = iy1 - sy1
        rel_x2 = ix2 - sx1
        rel_y2 = iy2 - sy1

        rel_cx = (rel_x1 + rel_x2) / 2 / slice_w
        rel_cy = (rel_y1 + rel_y2) / 2 / slice_h
        rel_bw = (rel_x2 - rel_x1) / slice_w
        rel_bh = (rel_y2 - rel_y1) / slice_h

        sliced_labels.append(f"{cls} {rel_cx:.6f} {rel_cy:.6f} {rel_bw:.6f} {rel_bh:.6f}")

    return sliced_labels


def _letterbox(img: np.ndarray, target: int) -> Tuple[np.ndarray, float, float, float]:
    """
    Resize keeping aspect ratio; pad to target x target. Returns (image, scale, pad_x, pad_y).
    """
    h, w = img.shape[:2]
    scale = min(target / w, target / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_x = (target - new_w) / 2
    pad_y = (target - new_h) / 2
    top, bottom = int(np.floor(pad_y)), int(np.ceil(pad_y))
    left, right = int(np.floor(pad_x)), int(np.ceil(pad_x))
    out = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    return out, scale, pad_x, pad_y


def _resolve_label_path(image_path: Path) -> Path:
    """
    Resolve label path. If image is under .../images/, prefer sibling .../labels/<stem>.txt.
    Otherwise fall back to same directory with .txt suffix.
    """
    if image_path.parent.name == "images":
        cand = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
        if cand.exists():
            return cand
    return image_path.with_suffix(".txt")


def save_slices_and_resized(
    image_path: str,
    output_dir: str,
    slice_size: int = 1280,
    overlap: float = 0.2,
    min_abs_area: int = 100,
    min_area_ratio: float = 0.6,
    resize_size: int = 1280
) -> None:
    """
    Slice image into patches and save both slices and a resized copy with labels.

    Output layout under output_dir:
        images/  -> slice images + resized image
        labels/  -> corresponding YOLO label txt

    Naming:
        - Slice k:  {stem}_{k}.<ext>   and {stem}_{k}.txt
        - Resized:  {stem}_F.<ext>     and {stem}_F.txt
    """
    image_path = Path(image_path)
    out_dir = Path(output_dir)
    images_dir = out_dir / "images"
    labels_dir = out_dir / "labels"

    # Prepare output dirs (clear if exist)
    for d in (images_dir, labels_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]

    label_path = _resolve_label_path(image_path)
    label_lines: List[str] = []
    if label_path.exists():
        with open(label_path, "r", encoding="utf-8") as f:
            label_lines = f.readlines()

    # Save slices
    slices = slice_image(str(image_path), slice_size=slice_size, overlap=overlap)
    for idx, (slice_img, box) in enumerate(slices):
        name = f"{image_path.stem}_{idx}"
        img_out_path = images_dir / f"{name}{image_path.suffix}"
        cv2.imwrite(str(img_out_path), slice_img)

        slice_label_lines = slice_labels(
            str(image_path),
            box,
            image_shape=(h, w),
            label_lines=label_lines,
            min_abs_area=min_abs_area,
            min_area_ratio=min_area_ratio
        )
        label_out_path = labels_dir / f"{name}.txt"
        with open(label_out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(slice_label_lines))

    # Save resized full image
    resized, scale, pad_x, pad_y = _letterbox(img, resize_size)
    resized_name = f"{image_path.stem}_F"
    resized_img_path = images_dir / f"{resized_name}{image_path.suffix}"
    cv2.imwrite(str(resized_img_path), resized)

    resized_labels: List[str] = []
    for line in label_lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls = parts[0]
        cx, cy, bw, bh = map(float, parts[1:])
        # denormalize to abs coords
        abs_cx = cx * w
        abs_cy = cy * h
        abs_bw = bw * w
        abs_bh = bh * h

        # scale and pad (letterbox)
        abs_cx = abs_cx * scale + pad_x
        abs_cy = abs_cy * scale + pad_y
        abs_bw = abs_bw * scale
        abs_bh = abs_bh * scale

        if abs_bw * abs_bh < min_abs_area:
            continue

        # re-normalize to target canvas (resize_size x resize_size)
        new_cx = abs_cx / resize_size
        new_cy = abs_cy / resize_size
        new_bw = abs_bw / resize_size
        new_bh = abs_bh / resize_size
        resized_labels.append(f"{cls} {new_cx:.6f} {new_cy:.6f} {new_bw:.6f} {new_bh:.6f}")

    resized_label_path = labels_dir / f"{resized_name}.txt"
    with open(resized_label_path, "w", encoding="utf-8") as f:
        f.write("\n".join(resized_labels))

