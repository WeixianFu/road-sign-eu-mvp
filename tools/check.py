"""Run lightweight checks and preserve their complete output for review."""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(__file__).resolve().parents[1]
output = root / "docs/refactor/checks"
output.mkdir(parents=True, exist_ok=True)
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
commands = [
    [sys.executable, "-m", "ruff", "check", "src", "tests", "tools"],
    [sys.executable, "-m", "pytest", "-q"],
]
failed = False
for index, command in enumerate(commands):
    result = subprocess.run(
        command, cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False
    )
    log = output / f"{stamp}-{index}.txt"
    log.write_text(result.stdout)
    entry = {
        "time": stamp,
        "command": command,
        "cwd": str(root),
        "exit_code": result.returncode,
        "output": str(log.relative_to(root)),
    }
    with (root / "docs/refactor/operations.jsonl").open("a") as stream:
        stream.write(json.dumps(entry) + "\n")
    print(result.stdout)
    failed |= result.returncode != 0
sys.exit(int(failed))
