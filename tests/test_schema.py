"""
Test module for detection result schema validation.

Purpose:
    This module contains tests to validate that exported detection results
    conform to the expected schema. This ensures data consistency and
    compatibility with downstream processing tools.

Schema fields to validate (Phase-1):
    - video_id: str
    - frame_idx: int
    - timestamp_ms: int
    - bbox_xyxy: List[float] (4 elements)
    - bbox_norm: List[float] (4 elements)
    - class_id: int
    - class_name: str
    - confidence: float (0.0-1.0)
    - model_name: str
    - model_version: str
    - imgsz: int

Future test cases:
    - validate_record_schema: Validate single detection record
    - test_schema_validation: Test valid and invalid records
"""
