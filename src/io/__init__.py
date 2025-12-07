"""
Video I/O module for reading video frames with accurate timestamps.

Purpose:
    This module provides functionality to read video files using PyAV and extract
    frames with precise timestamps calculated from PTS (Presentation Time Stamp)
    and time_base, ensuring accurate timing even for variable frame rate (VFR) videos.

Future implementation:
    - read_video: Read video frames with PTS-based timestamps
    - Image loading utilities (if needed)
"""

from .slicer import slice_image
from .slicer import slice_labels

__all__ = ['slice_image', 'slice_labels']
