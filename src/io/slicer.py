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
                x = w
                break
            x += step

        
        if y2 >= h:
            y = h
            break
        
        y += step
    
    return slices

