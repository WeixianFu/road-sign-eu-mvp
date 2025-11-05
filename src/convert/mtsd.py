"""
Utilities for converting Mapillary Traffic Sign Dataset (MTSD) data into YOLOv8-compatible data.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

LOGGER = logging.getLogger(__name__)

DEFAULT_IMAGE_EXTS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


@dataclass
class ConversionConfig:
    """Configuration describing how MTSD annotations should be converted."""

    source_dir: Path
    output_dir: Path
    split: str = "val"
    class_names: Optional[Sequence[str]] = None
    accepted_image_exts: Tuple[str, ...] = DEFAULT_IMAGE_EXTS
    copy_images: bool = True
    filter_ambiguous: bool = True
    filter_dummy: bool = True

    #: Whether to skip annotations that are flagged as outside of the frame.
    filter_out_of_frame: bool = True

    #: Persist a simple class-name lookup table next to the labels.
    write_class_map: bool = True

    #: File stem prefix to prepend when copying images/labels (useful for split prefixes).
    stem_prefix: str = ""

    def image_dir(self) -> Path:
        return self.output_dir / "images" / self.split

    def label_dir(self) -> Path:
        return self.output_dir / "labels" / self.split


def convert_mtsd_to_yolo(config: ConversionConfig) -> Mapping[str, int]:
    """
    Convert MTSD JSON annotations under ``config.source_dir`` into YOLO-style label files.

    Parameters
    ----------
    config:
        Settings describing source paths, the destination layout, and filtering rules.

    Returns
    -------
    Mapping[str, int]
        Dictionary mapping MTSD class names to YOLO class indices produced during conversion.
    """

    source_dir = config.source_dir
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    json_paths = sorted(p for p in source_dir.rglob("*.json") if p.is_file())
    if not json_paths:
        raise RuntimeError(f"No MTSD JSON files found under {source_dir}")

    image_dir = config.image_dir()
    label_dir = config.label_dir()
    label_dir.mkdir(parents=True, exist_ok=True)
    if config.copy_images:
        image_dir.mkdir(parents=True, exist_ok=True)

    class_to_id: MutableMapping[str, int] = (
        {name: idx for idx, name in enumerate(config.class_names)}
        if config.class_names
        else {}
    )

    for json_path in json_paths:
        with json_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)

        width = float(payload.get("width") or payload.get("image_width") or 0)
        height = float(payload.get("height") or payload.get("image_height") or 0)
        if not width or not height:
            raise ValueError(f"Missing width/height for {json_path}")

        stem = f"{config.stem_prefix}{json_path.stem}"
        label_path = label_dir / f"{stem}.txt"

        records = _objects_to_yolo_rows(
            payload.get("objects", []),
            class_to_id,
            width,
            height,
            filter_ambiguous=config.filter_ambiguous,
            filter_dummy=config.filter_dummy,
            filter_out_of_frame=config.filter_out_of_frame,
        )

        _write_label_file(label_path, records)

        if config.copy_images:
            image_path = _resolve_image_path(json_path, config.accepted_image_exts)
            if image_path is None:
                LOGGER.warning("No image found for %s; skipping copy", json_path)
            else:
                target_path = image_dir / f"{stem}{image_path.suffix.lower()}"
                shutil.copy2(image_path, target_path)

    if config.write_class_map:
        _persist_class_map(config.output_dir, class_to_id)

    return dict(class_to_id)


def _objects_to_yolo_rows(
    objects: Iterable[Mapping[str, object]],
    class_to_id: MutableMapping[str, int],
    width: float,
    height: float,
    *,
    filter_ambiguous: bool,
    filter_dummy: bool,
    filter_out_of_frame: bool,
) -> List[str]:
    rows: List[str] = []
    for obj in objects:
        label = obj.get("label")
        if not isinstance(label, str):
            continue

        props = obj.get("properties") or {}
        if filter_ambiguous and bool(props.get("ambiguous")):
            continue
        if filter_dummy and bool(props.get("dummy")):
            continue
        if filter_out_of_frame and bool(props.get("out-of-frame")):
            continue

        bbox = obj.get("bbox") or {}
        xmin, ymin, xmax, ymax = (
            float(bbox.get("xmin", 0.0)),
            float(bbox.get("ymin", 0.0)),
            float(bbox.get("xmax", 0.0)),
            float(bbox.get("ymax", 0.0)),
        )

        # Clamp to image bounds to avoid invalid normalization.
        xmin = max(0.0, min(xmin, width))
        ymin = max(0.0, min(ymin, height))
        xmax = max(0.0, min(xmax, width))
        ymax = max(0.0, min(ymax, height))

        box_w = max(0.0, xmax - xmin)
        box_h = max(0.0, ymax - ymin)
        if box_w <= 0 or box_h <= 0:
            continue

        x_center = (xmin + xmax) / 2.0 / width
        y_center = (ymin + ymax) / 2.0 / height
        box_w_norm = box_w / width
        box_h_norm = box_h / height

        class_id = class_to_id.setdefault(label, len(class_to_id))
        rows.append(
            f"{class_id} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{box_w_norm:.6f} "
            f"{box_h_norm:.6f}"
        )

    return rows


def _write_label_file(path: Path, rows: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        if rows:
            fh.write("\n".join(rows) + "\n")
        else:
            fh.write("")


def _resolve_image_path(json_path: Path, accepted_exts: Sequence[str]) -> Optional[Path]:
    for ext in accepted_exts:
        candidate = json_path.with_suffix(ext)
        if candidate.exists():
            return candidate
    # Some MTSD dumps include multiple derivatives (e.g., JPG + PNG). Prefer JPG if present.
    alternatives = list(json_path.parent.glob(f"{json_path.stem}.*"))
    for candidate in alternatives:
        if candidate.suffix.lower() in accepted_exts and candidate.is_file():
            return candidate
    return None


@dataclass(frozen=True)
class PackageSummary:
    """Summary describing how a downloaded MTSD image bundle maps onto annotations."""

    package_dir: Path
    dataset: Optional[str]
    splits: Tuple[str, ...]
    total_images: int
    matched_images: int
    unmatched_count: int
    unmatched_examples: Tuple[str, ...]

    def as_dict(self) -> Dict[str, object]:
        return {
            "package": str(self.package_dir),
            "dataset": self.dataset,
            "splits": list(self.splits),
            "total_images": self.total_images,
            "matched_images": self.matched_images,
            "unmatched_count": self.unmatched_count,
            "unmatched_examples": list(self.unmatched_examples),
        }


def map_mtsd_image_packages(
    packages_root: Path,
    *,
    fully_root: Path,
    partially_root: Path,
    accepted_image_exts: Sequence[str] = DEFAULT_IMAGE_EXTS,
    unmatched_preview: int = 5,
) -> List[PackageSummary]:
    """
    Match scrambled MTSD image bundle folders to the fully/partially annotated datasets.

    Parameters
    ----------
    packages_root:
        Directory containing the downloaded image bundles (folders with hashed names).
    fully_root:
        Root folder of the ``mtsd_v2_fully_annotated`` annotations.
    partially_root:
        Root folder of the ``mtsd_v2_partially_annotated`` annotations.
    accepted_image_exts:
        File extensions that should be treated as images when scanning the bundles.
    unmatched_preview:
        Maximum number of unmatched filenames to keep per package summary (for inspection).
    """

    accepted_exts = {ext.lower() for ext in accepted_image_exts}

    fully_annotations = fully_root / "annotations"
    partial_annotations = partially_root / "annotations"
    if not fully_annotations.exists():
        raise FileNotFoundError(f"Fully annotated annotations dir missing: {fully_annotations}")
    if not partial_annotations.exists():
        raise FileNotFoundError(f"Partially annotated annotations dir missing: {partial_annotations}")

    fully_index = _build_annotation_index(fully_annotations)
    partial_index = _build_annotation_index(partial_annotations)
    fully_split_index = _build_split_index(fully_root / "splits")
    partial_split_index = _build_split_index(partially_root / "splits")

    ignore_names = {
        "sample",
        fully_root.name,
        partially_root.name,
        "mtsd_v2_fully_annotated",
        "mtsd_v2_partially_annotated",
    }

    summaries: List[PackageSummary] = []
    for package_dir in sorted(packages_root.iterdir()):
        if not package_dir.is_dir():
            continue
        if package_dir.name in ignore_names or package_dir.name.startswith("mtsd_"):
            continue

        images_dir = package_dir / "images"
        if not images_dir.is_dir():
            LOGGER.debug("Skipping %s: missing images/ directory", package_dir)
            continue

        stems: List[str] = [
            image_path.stem
            for image_path in images_dir.iterdir()
            if image_path.is_file() and image_path.suffix.lower() in accepted_exts
        ]
        if not stems:
            LOGGER.debug("Skipping %s: no images with accepted extensions", package_dir)
            continue

        matched_fully = {stem for stem in stems if stem in fully_index}
        matched_partial = {stem for stem in stems if stem in partial_index}
        matched_union = matched_fully | matched_partial
        unmatched = [stem for stem in stems if stem not in matched_union]

        if matched_fully and matched_partial:
            dataset = "mixed"
        elif matched_fully:
            dataset = "fully"
        elif matched_partial:
            dataset = "partial"
        else:
            dataset = None

        split_labels: set[str] = set()
        for stem in matched_fully:
            split_name = fully_split_index.get(stem, "unknown")
            split_labels.add(f"fully:{split_name}")
        for stem in matched_partial:
            split_name = partial_split_index.get(stem, "unknown")
            split_labels.add(f"partial:{split_name}")

        summaries.append(
            PackageSummary(
                package_dir=package_dir,
                dataset=dataset,
                splits=tuple(sorted(split_labels)),
                total_images=len(stems),
                matched_images=len(matched_union),
                unmatched_count=len(unmatched),
                unmatched_examples=tuple(unmatched[:unmatched_preview]),
            )
        )

    return summaries


def _build_annotation_index(annotations_dir: Path) -> Dict[str, Path]:
    if not annotations_dir.is_dir():
        raise FileNotFoundError(f"Annotations directory missing: {annotations_dir}")
    return {path.stem: path for path in annotations_dir.glob("*.json")}


def _build_split_index(splits_dir: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not splits_dir.is_dir():
        LOGGER.warning("Split directory missing: %s", splits_dir)
        return mapping

    for split_file in splits_dir.glob("*.txt"):
        split_name = split_file.stem
        with split_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                image_id = line.strip()
                if image_id:
                    mapping[image_id] = split_name
    return mapping


def _format_package_summary(summary: PackageSummary) -> str:
    dataset = summary.dataset or "unknown"
    splits = ", ".join(summary.splits) if summary.splits else "n/a"
    unmatched = (
        f"{summary.unmatched_count} (e.g. {', '.join(summary.unmatched_examples)})"
        if summary.unmatched_count
        else "0"
    )
    return (
        f"{summary.package_dir.name:>64}  "
        f"dataset={dataset:<7}  "
        f"splits={splits:<20}  "
        f"matched={summary.matched_images}/{summary.total_images}  "
        f"unmatched={unmatched}"
    )


def _run_map_packages_cli(argv: Optional[Sequence[str]] = None) -> List[PackageSummary]:
    parser = argparse.ArgumentParser(
        description=(
            "Map MTSD image bundle folders (hashed names) to the fully/partially "
            "annotated datasets by inspecting annotation filenames."
        )
    )
    parser.add_argument(
        "--packages-root",
        type=Path,
        default=Path("/mnt/e/tsr"),
        help="Directory that contains the downloaded image bundles.",
    )
    parser.add_argument(
        "--fully-root",
        type=Path,
        default=None,
        help="Path to mtsd_v2_fully_annotated (defaults to PACKAGES_ROOT/mtsd_v2_fully_annotated).",
    )
    parser.add_argument(
        "--partially-root",
        type=Path,
        default=None,
        help="Path to mtsd_v2_partially_annotated (defaults to PACKAGES_ROOT/mtsd_v2_partially_annotated).",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=5,
        help="Number of unmatched filenames to show per package (default: 5).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress non-essential log messages.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if not args.quiet else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    fully_root = args.fully_root or (args.packages_root / "mtsd_v2_fully_annotated")
    partially_root = args.partially_root or (args.packages_root / "mtsd_v2_partially_annotated")

    summaries = map_mtsd_image_packages(
        args.packages_root,
        fully_root=fully_root,
        partially_root=partially_root,
        unmatched_preview=max(0, args.preview),
    )

    if not summaries:
        LOGGER.warning("No matching MTSD image bundles were found under %s", args.packages_root)
        return

    for summary in summaries:
        print(_format_package_summary(summary))

    print("\nHint: use --preview 0 to suppress unmatched example filenames.")
    return summaries


def _build_image_index(
    packages_root: Path,
    *,
    accepted_image_exts: Sequence[str],
    ignore_dirs: Optional[Set[str]] = None,
) -> Dict[str, Path]:
    ignore = {
        "sample",
        "mtsd_v2_fully_annotated",
        "mtsd_v2_partially_annotated",
    }
    if ignore_dirs:
        ignore.update({name for name in ignore_dirs if name})

    accepted = {ext.lower() for ext in accepted_image_exts}
    index: Dict[str, Path] = {}
    for package_dir in sorted(packages_root.iterdir()):
        if not package_dir.is_dir():
            continue
        if package_dir.name in ignore or package_dir.name.startswith("mtsd_"):
            continue

        images_dir = package_dir / "images"
        if not images_dir.is_dir():
            continue

        for image_path in images_dir.rglob("*"):
            if not image_path.is_file():
                continue
            if image_path.suffix.lower() not in accepted:
                continue
            stem = image_path.stem
            existing = index.get(stem)
            if existing and existing != image_path:
                raise RuntimeError(
                    f"Duplicate image stem detected: {stem} appears in {existing} and {image_path}"
                )
            index[stem] = image_path

    if not index:
        raise RuntimeError(f"No MTSD images found under {packages_root}")
    return index


def _load_split_ids(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _normalise_split_name(dataset_label: str, split_name: str) -> str:
    if dataset_label == "partial":
        return f"{split_name}_partial"
    return split_name


def convert_all_mtsd(
    *,
    packages_root: Path,
    fully_root: Path,
    partially_root: Optional[Path],
    output_dir: Path,
    copy_images: bool = True,
    accepted_image_exts: Sequence[str] = DEFAULT_IMAGE_EXTS,
    filter_ambiguous: bool = True,
    filter_dummy: bool = True,
    filter_out_of_frame: bool = True,
) -> Mapping[str, int]:
    """
    Convert the full MTSD dataset (fully + partially annotated) into YOLO directory structure.
    """

    packages_root = packages_root.expanduser().resolve()
    fully_root = fully_root.expanduser().resolve()
    partially_root = partially_root.expanduser().resolve() if partially_root else None
    output_dir = output_dir.expanduser().resolve()

    if not fully_root.exists():
        raise FileNotFoundError(f"Fully annotated dataset directory not found: {fully_root}")

    if partially_root is not None and not partially_root.exists():
        LOGGER.warning("Partially annotated dataset directory missing: %s", partially_root)
        partially_root = None

    ignore_dirs: Set[str] = {fully_root.name}
    if partially_root is not None:
        ignore_dirs.add(partially_root.name)

    image_index = _build_image_index(
        packages_root,
        accepted_image_exts=accepted_image_exts,
        ignore_dirs=ignore_dirs,
    )

    datasets: List[Tuple[str, Path]] = [("fully", fully_root)]
    if partially_root is not None:
        datasets.append(("partial", partially_root))

    class_to_id: MutableMapping[str, int] = {}
    output_images_root = output_dir / "images"
    output_labels_root = output_dir / "labels"
    if copy_images:
        output_images_root.mkdir(parents=True, exist_ok=True)
    output_labels_root.mkdir(parents=True, exist_ok=True)

    processed = 0
    warnings = 0

    for dataset_label, dataset_root in datasets:
        annotations_dir = dataset_root / "annotations"
        if not annotations_dir.is_dir():
            LOGGER.warning("Annotations directory missing for %s dataset: %s", dataset_label, annotations_dir)
            continue

        splits_dir = dataset_root / "splits"
        split_files = sorted(splits_dir.glob("*.txt")) if splits_dir.is_dir() else []
        if not split_files:
            LOGGER.warning("No split files found for %s dataset under %s", dataset_label, splits_dir)
            continue

        for split_file in split_files:
            split_name = split_file.stem
            dest_split = _normalise_split_name(dataset_label, split_name)

            image_ids = _load_split_ids(split_file)
            if not image_ids:
                LOGGER.info("Split %s/%s is empty; skipping", dataset_label, split_name)
                continue

            label_split_dir = output_labels_root / dest_split
            label_split_dir.mkdir(parents=True, exist_ok=True)
            image_split_dir = None
            if copy_images:
                image_split_dir = output_images_root / dest_split
                image_split_dir.mkdir(parents=True, exist_ok=True)

            converted_split = 0
            missing_ann_count = 0
            missing_img_count = 0
            missing_ann_examples: List[str] = []
            missing_img_examples: List[str] = []

            for image_id in image_ids:
                annotation_path = annotations_dir / f"{image_id}.json"
                if not annotation_path.exists():
                    missing_ann_count += 1
                    if len(missing_ann_examples) < 5:
                        missing_ann_examples.append(image_id)
                    continue

                image_path = image_index.get(image_id)
                if image_path is None:
                    missing_img_count += 1
                    if len(missing_img_examples) < 5:
                        missing_img_examples.append(image_id)
                    continue

                with annotation_path.open("r", encoding="utf-8") as fh:
                    payload = json.load(fh)

                width = float(payload.get("width") or payload.get("image_width") or 0)
                height = float(payload.get("height") or payload.get("image_height") or 0)
                if not width or not height:
                    LOGGER.warning(
                        "Invalid dimensions for %s split %s image %s",
                        dataset_label,
                        split_name,
                        image_id,
                    )
                    warnings += 1
                    continue

                rows = _objects_to_yolo_rows(
                    payload.get("objects", []),
                    class_to_id,
                    width,
                    height,
                    filter_ambiguous=filter_ambiguous,
                    filter_dummy=filter_dummy,
                    filter_out_of_frame=filter_out_of_frame,
                )

                stem = f"{dataset_label}_{dest_split}_{image_id}"
                label_path = label_split_dir / f"{stem}.txt"
                _write_label_file(label_path, rows)

                if copy_images and image_split_dir is not None:
                    target_image_path = image_split_dir / f"{stem}{image_path.suffix.lower()}"
                    shutil.copy2(image_path, target_image_path)

                processed += 1
                converted_split += 1

            if missing_ann_count:
                warnings += missing_ann_count
                LOGGER.warning(
                    "Missing %d annotations for dataset=%s split=%s (e.g. %s)",
                    missing_ann_count,
                    dataset_label,
                    split_name,
                    ", ".join(missing_ann_examples) if missing_ann_examples else "n/a",
                )
            if missing_img_count:
                warnings += missing_img_count
                LOGGER.warning(
                    "Missing %d images for dataset=%s split=%s (e.g. %s)",
                    missing_img_count,
                    dataset_label,
                    split_name,
                    ", ".join(missing_img_examples) if missing_img_examples else "n/a",
                )

            LOGGER.info(
                "Converted %d/%d samples for dataset=%s split=%s (dest=%s)",
                converted_split,
                len(image_ids),
                dataset_label,
                split_name,
                dest_split,
            )

    if processed == 0:
        LOGGER.warning("No MTSD samples were converted; check your input directories.")
    else:
        LOGGER.info("Converted %d MTSD samples with %d warnings.", processed, warnings)

    _persist_class_map(output_dir, class_to_id)
    return dict(class_to_id)


def _run_convert_all_cli(argv: Optional[Sequence[str]] = None) -> Mapping[str, int]:
    parser = argparse.ArgumentParser(
        description="Convert all MTSD datasets (fully + partial) into a YOLO directory layout."
    )
    parser.add_argument(
        "--packages-root",
        type=Path,
        default=Path("/mnt/e/tsr"),
        help="Directory containing the downloaded image bundles (default: /mnt/e/tsr).",
    )
    parser.add_argument(
        "--fully-root",
        type=Path,
        default=None,
        help="Path to mtsd_v2_fully_annotated (defaults to PACKAGES_ROOT/mtsd_v2_fully_annotated).",
    )
    parser.add_argument(
        "--partially-root",
        type=Path,
        default=None,
        help="Path to mtsd_v2_partially_annotated (defaults to PACKAGES_ROOT/mtsd_v2_partially_annotated).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/mtsd"),
        help="Destination directory for YOLO-formatted data (default: data/mtsd).",
    )
    parser.add_argument(
        "--labels-only",
        action="store_true",
        help="Generate YOLO labels without copying images.",
    )
    parser.add_argument(
        "--skip-partial",
        action="store_true",
        help="Skip the partially annotated dataset.",
    )
    parser.add_argument(
        "--include-ambiguous",
        action="store_true",
        help="Keep annotations flagged as ambiguous.",
    )
    parser.add_argument(
        "--include-dummy",
        action="store_true",
        help="Keep annotations marked as dummy.",
    )
    parser.add_argument(
        "--include-out-of-frame",
        action="store_true",
        help="Keep annotations extending outside the frame.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress INFO-level logging.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if not args.quiet else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    fully_root = args.fully_root or (args.packages_root / "mtsd_v2_fully_annotated")
    partially_root = None
    if not args.skip_partial:
        partially_root = args.partially_root or (args.packages_root / "mtsd_v2_partially_annotated")

    class_map = convert_all_mtsd(
        packages_root=args.packages_root,
        fully_root=fully_root,
        partially_root=partially_root,
        output_dir=args.output_dir,
        copy_images=not args.labels_only,
        filter_ambiguous=not args.include_ambiguous,
        filter_dummy=not args.include_dummy,
        filter_out_of_frame=not args.include_out_of_frame,
    )
    LOGGER.info("Wrote %d class definitions to %s", len(class_map), args.output_dir / "classes.json")
    return class_map


def _persist_class_map(output_dir: Path, class_to_id: Mapping[str, int]) -> None:
    names = [name for name, _ in sorted(class_to_id.items(), key=lambda item: item[1])]
    target = output_dir / "classes.json"
    with target.open("w", encoding="utf-8") as fh:
        json.dump({"names": names}, fh, ensure_ascii=False, indent=2)


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )


def _parse_args(argv: Optional[Sequence[str]] = None):
    import argparse

    parser = argparse.ArgumentParser(description="Convert MTSD annotations to YOLO format.")
    parser.add_argument("source", type=Path, help="Directory containing MTSD JSON annotations.")
    parser.add_argument(
        "output",
        type=Path,
        help="Destination directory where YOLO images/labels will be written.",
    )
    parser.add_argument("--split", default="val", help="Dataset split identifier (default: val).")
    parser.add_argument(
        "--no-copy-images",
        action="store_true",
        help="Write labels only without copying images.",
    )
    parser.add_argument(
        "--include-ambiguous",
        action="store_true",
        help="Keep annotations flagged as ambiguous.",
    )
    parser.add_argument(
        "--include-dummy",
        action="store_true",
        help="Keep annotations marked as dummy placeholders.",
    )
    parser.add_argument(
        "--include-out-of-frame",
        action="store_true",
        help="Keep annotations that extend outside the frame.",
    )
    parser.add_argument(
        "--class-list",
        type=Path,
        help="Optional path to a text file listing class names (one per line).",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable logging.")

    args = parser.parse_args(argv)
    class_names = None
    if args.class_list:
        with args.class_list.open("r", encoding="utf-8") as fh:
            class_names = [line.strip() for line in fh if line.strip()]

    config = ConversionConfig(
        source_dir=args.source,
        output_dir=args.output,
        split=args.split,
        copy_images=not args.no_copy_images,
        filter_ambiguous=not args.include_ambiguous,
        filter_dummy=not args.include_dummy,
        filter_out_of_frame=not args.include_out_of_frame,
        class_names=class_names,
    )
    return args, config


def main(argv: Optional[Sequence[str]] = None) -> Mapping[str, int]:
    if argv is None:
        argv = sys.argv[1:]
    argv = list(argv)

    if argv and argv[0] == "map-packages":
        _run_map_packages_cli(argv[1:])
        return {}

    if argv and argv[0] == "convert-all":
        return _run_convert_all_cli(argv[1:])

    args, config = _parse_args(argv)
    _configure_logging(args.verbose)
    class_map = convert_mtsd_to_yolo(config)
    LOGGER.info("Converted %d classes", len(class_map))
    return class_map


if __name__ == "__main__":
    main()
