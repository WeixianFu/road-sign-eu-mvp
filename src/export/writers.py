"""
Export detection results to NDJSON and Parquet formats.

Purpose:
    This module provides functionality to export YOLOv8 detection results
    to structured data formats. NDJSON (Newline Delimited JSON) is useful
    for streaming and debugging, while Parquet is optimized for analytics
    and large-scale data processing.

Key features to implement:
    - Frame-level export with complete schema
    - NDJSON format for line-by-line processing
    - Parquet format for columnar analytics
    - Schema validation
    
Frame-level schema (Phase-1):
    - video_id: Identifier for the source video
    - frame_idx: Frame index in the video
    - timestamp_ms: Timestamp in milliseconds (from PTS)
    - bbox_xyxy: Bounding box in absolute coordinates [x1, y1, x2, y2]
    - bbox_norm: Bounding box in normalized coordinates [x_center, y_center, w, h]
    - class_id: Class index
    - class_name: Class name string
    - confidence: Detection confidence score
    - model_name: Model name/identifier
    - model_version: Model version
    - imgsz: Image size used for inference
"""
