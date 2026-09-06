# Project contract

- Train and evaluate European traffic-sign detectors from MTSD images. Europe includes Switzerland and the UK. Geography and class semantics are separate.
- Use YOLOv8; keep model selection configurable so YOLO11 can be compared later.
- Keep image tools, training, evaluation and analysis in separate folders under `src/roadsigns`.
- Preserve small signs with 1280-pixel tiles and 20% overlap. Original-image evaluation uses the same slicing and class-aware merging as image prediction.
- Use fully labelled training data. Keep 70% tile / 30% eligible full-image sampling and tiny-safe Mosaic: scale 0.9–1.1, no shrinking when any box is below 32 pixels, at least 60% visible area and a 16-pixel short side after stitching. Close Mosaic for the last 10 epochs. Disable flips, rotation and MixUp.
- Class names, IDs and ontology fingerprints must agree across prepared data, training artifacts, prediction and evaluation. Keep direction, numeric speed and distinct sign meanings separate. Do not infer a speed unit or country from a class name.
- The Mac is for code review and lightweight synthetic tests. Dataset processing and training run on the Ubuntu RTX 5070 Ti desktop or a configured Linux server. Do not run training or process the real dataset on the Mac.
- Write concise Python, simple functions and only necessary comments. Validate external inputs once; let errors propagate. Do not add silent fallbacks, broad exception handling or speculative abstractions.
- Write README files, code comments, docstrings and AGENTS.md in English.
- Record operations and verification results. During a long refactor, write a progress report every 20 minutes.
- Video, tracking, OCR and semi-supervised training are outside this iteration. Ask before expanding scope.
