"""
Video reader module using PyAV for frame extraction with accurate timestamps.

Purpose:
    This module provides video reading functionality that extracts frames from
    video files and computes accurate timestamps using PTS (Presentation Time Stamp)
    and time_base. This approach is VFR-safe and more accurate than frame-based
    timestamp calculation.

Key features to implement:
    - Uses PyAV (FFmpeg bindings) for video decoding
    - Calculates timestamps from PTS × time_base (not frame index)
    - Handles variable frame rate (VFR) videos correctly
    - Returns frames in BGR format (OpenCV compatible)
    - Provides frame dimensions for coordinate scaling

Contract:
    Yield (frame_bgr, timestamp_ms, (W,H)) where timestamp_ms = round(frame.pts * stream.time_base * 1000)
    Handle VFR and missing timestamps robustly (skip or warn)
"""
