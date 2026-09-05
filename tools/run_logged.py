"""Capture a command's complete console output and exit status."""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--log", type=Path, required=True)
parser.add_argument("command", nargs=argparse.REMAINDER)
args = parser.parse_args()
args.log.parent.mkdir(parents=True, exist_ok=True)
record = {
    "started_at": datetime.now(timezone.utc).isoformat(),
    "command": args.command,
    "cwd": str(Path.cwd()),
}
with args.log.open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(record) + "\n")
    stream.flush()
    with subprocess.Popen(
        args.command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    ) as process:
        for line in process.stdout:
            stream.write(line)
            stream.flush()
            print(line, end="", flush=True)
    record.update(finished_at=datetime.now(timezone.utc).isoformat(), exit_code=process.returncode)
    stream.write(json.dumps(record) + "\n")
sys.exit(process.returncode)
