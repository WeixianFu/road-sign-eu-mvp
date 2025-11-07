# convert MTSD data to YOLO format

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Tuple

__all__ = [
    "ConversionConfig",
    "CopySummary",
    "build_default_source_map",
    "convert_mtsd_to_yolo",
    "create_mtsd_scaffold",
    "populate_from_sources",
]

_ATTR_TO_MODULE: Dict[str, Tuple[str, str]] = {
    "ConversionConfig": (".mtsd", "ConversionConfig"),
    "convert_mtsd_to_yolo": (".mtsd", "convert_mtsd_to_yolo"),
    "CopySummary": (".scaffold_mtsd", "CopySummary"),
    "build_default_source_map": (".scaffold_mtsd", "build_default_source_map"),
    "create_mtsd_scaffold": (".scaffold_mtsd", "create_mtsd_scaffold"),
    "populate_from_sources": (".scaffold_mtsd", "populate_from_sources"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _ATTR_TO_MODULE[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
