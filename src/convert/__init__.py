# convert MTSD data to YOLO format

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["ConversionConfig", "convert_mtsd_to_yolo"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        module = import_module(".mtsd", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
