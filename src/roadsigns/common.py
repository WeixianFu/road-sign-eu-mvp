from __future__ import annotations

import hashlib
import json
import logging
import platform
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import yaml


def read_yaml(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def environment():
    packages = {d.metadata["Name"]: d.version for d in metadata.distributions()}
    package = Path(__file__).parent
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=package, capture_output=True, text=True, check=False
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=package, capture_output=True, text=True, check=False
    )
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "git_commit": commit.stdout.strip() or None,
        "git_dirty": bool(status.stdout) if status.returncode == 0 else None,
        "code_sha256": digest(
            {str(p.relative_to(package)): file_hash(p) for p in sorted(package.rglob("*.py"))}
        ),
    }


def setup_logging(path):
    logger = logging.getLogger("roadsigns")
    logger.setLevel(logging.INFO)
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(path, encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def event(path, action, **fields):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {"time": datetime.now(timezone.utc).isoformat(), "action": action, **fields},
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        )
