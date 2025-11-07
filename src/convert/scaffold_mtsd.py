"""
Utilities for scaffolding an MTSD directory tree for YOLOv8 training assets.
"""

from __future__ import annotations

import argparse
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

LOGGER = logging.getLogger(__name__)

DATASET_FOLDERS: Sequence[str] = ("full", "paritl", "sample")
SPLIT_FOLDERS: Sequence[str] = ("train", "val")
LEAF_FOLDERS: Sequence[str] = ("images", "labels")

DEFAULT_SOURCE_ROOT = Path("data/mtsd")
DEFAULT_SAMPLE_ROOT = Path("data/sample/mtsd_yolo")


@dataclass
class CopySummary:
    """Simple statistics for copy operations."""

    planned: int = 0
    copied: int = 0
    skipped_existing: int = 0
    missing_sources: List[Tuple[str, Path]] = field(default_factory=list)
    non_directory_sources: List[Tuple[str, Path]] = field(default_factory=list)

    def __iadd__(self, other: "CopySummary") -> "CopySummary":
        self.planned += other.planned
        self.copied += other.copied
        self.skipped_existing += other.skipped_existing
        self.missing_sources.extend(other.missing_sources)
        self.non_directory_sources.extend(other.non_directory_sources)
        return self


def create_mtsd_scaffold(
    root: Path,
    *,
    datasets: Iterable[str] = DATASET_FOLDERS,
    splits: Iterable[str] = SPLIT_FOLDERS,
    leaves: Iterable[str] = LEAF_FOLDERS,
    dry_run: bool = False,
) -> List[Path]:
    """
    Create (or preview) the nested directory layout expected for MTSD YOLO data.
    """

    root = Path(root).expanduser()
    created: List[Path] = []

    if not dry_run:
        root.mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
        dataset_dir = root / dataset
        for split in splits:
            for leaf in leaves:
                target = dataset_dir / split / leaf
                created.append(target)
                if dry_run:
                    continue
                target.mkdir(parents=True, exist_ok=True)

    return created


def build_default_source_map(
    source_root: Path,
    sample_root: Path | None = None,
) -> Dict[Tuple[str, str, str], Path]:
    """
    Produce a default mapping from (dataset, split, leaf) -> source directory.

    Parameters
    ----------
    source_root:
        Base directory containing MTSD YOLO-formatted ``images`` and ``labels`` folders.
    sample_root:
        Optional directory with sample assets; if omitted the sample split is left empty.
    """

    source_root = Path(source_root).expanduser()
    mapping: Dict[Tuple[str, str, str], Path] = {
        ("full", "train", "images"): source_root / "images" / "train",
        ("full", "train", "labels"): source_root / "labels" / "train",
        ("full", "val", "images"): source_root / "images" / "val",
        ("full", "val", "labels"): source_root / "labels" / "val",
        ("paritl", "train", "images"): source_root / "images" / "train_partial",
        ("paritl", "train", "labels"): source_root / "labels" / "train_partial",
    }

    if sample_root is not None:
        sample_root = Path(sample_root).expanduser()
        for split in SPLIT_FOLDERS:
            for leaf in LEAF_FOLDERS:
                candidate = sample_root / leaf / split
                if candidate.exists():
                    mapping[("sample", split, leaf)] = candidate

    return mapping


def populate_from_sources(
    root: Path,
    source_map: Mapping[Tuple[str, str, str], Path],
    *,
    dry_run: bool = False,
    overwrite: bool = False,
) -> CopySummary:
    """
    Copy YOLO-formatted files from ``source_map`` into the scaffolded structure.
    """

    root = Path(root).expanduser()
    summary = CopySummary()

    for (dataset, split, leaf), src in source_map.items():
        key = f"{dataset}/{split}/{leaf}"
        src = Path(src).expanduser()
        if not src.exists():
            summary.missing_sources.append((key, src))
            LOGGER.warning("Source directory missing for %s: %s", key, src)
            continue
        if not src.is_dir():
            summary.non_directory_sources.append((key, src))
            LOGGER.warning("Source path is not a directory for %s: %s", key, src)
            continue

        dst = root / dataset / split / leaf
        if not dry_run:
            dst.mkdir(parents=True, exist_ok=True)

        for item in src.iterdir():
            if not item.is_file():
                continue

            summary.planned += 1
            target = dst / item.name
            if target.exists() and not overwrite:
                summary.skipped_existing += 1
                continue

            if dry_run:
                summary.copied += 1
                continue

            shutil.copy2(item, target)
            summary.copied += 1

    return summary


def _run_cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Scaffold an MTSD directory structure under the target root and "
            "optionally populate it from the project-local MTSD datasets."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/mnt/g/mtsd"),
        help="Base directory to create if missing (default: /mnt/g/mtsd).",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="Root containing MTSD YOLO images/labels (default: data/mtsd).",
    )
    parser.add_argument(
        "--sample-root",
        type=Path,
        default=DEFAULT_SAMPLE_ROOT,
        help="Optional sample dataset root with images/labels (default: data/sample/mtsd_yolo).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the directories and copy operations without touching the filesystem.",
    )
    parser.add_argument(
        "--skip-copy",
        action="store_true",
        help="Only create the folder structure; do not copy any data.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files in the target tree if they already exist.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console output (useful if integrating elsewhere).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(
        level=logging.INFO if not args.quiet else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    created = create_mtsd_scaffold(
        args.root,
        dry_run=args.dry_run,
    )

    summary = CopySummary()
    if not args.skip_copy:
        sample_root = args.sample_root if args.sample_root is not None else None
        source_map = build_default_source_map(
            args.source_root,
            sample_root=sample_root,
        )
        summary = populate_from_sources(
            args.root,
            source_map,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )

    if not args.quiet:
        header = "Would create" if args.dry_run else "Created"
        for path in created:
            print(f"{header}: {path}")

        if not args.skip_copy:
            verb = "would copy" if args.dry_run else "copied"
            print(
                f"{verb} {summary.copied}/{summary.planned} files "
                f"(skipped existing: {summary.skipped_existing})"
            )
            if summary.missing_sources:
                print("Missing sources:")
                for key, src in summary.missing_sources:
                    print(f"  {key}: {src}")
            if summary.non_directory_sources:
                print("Non-directory sources:")
                for key, src in summary.non_directory_sources:
                    print(f"  {key}: {src}")

    return 0


if __name__ == "__main__":
    raise SystemExit(_run_cli())
