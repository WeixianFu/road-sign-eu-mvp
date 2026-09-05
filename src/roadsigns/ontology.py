from __future__ import annotations

import csv
from pathlib import Path

from roadsigns.common import digest


class Ontology:
    def __init__(self, path):
        with Path(path).open(encoding="utf-8", newline="") as stream:
            self.rows = list(csv.DictReader(stream))
        ids = [int(row["source_id"]) for row in self.rows]
        if ids != list(range(401)):
            raise ValueError("Ontology must list source IDs 0–400 exactly once, in source order")
        names = sorted({r["target"] for r in self.rows if r["action"] == "train"})
        self.names = names
        self.mapping = {}
        self.review_ids = set()
        for row in self.rows:
            source_id = int(row["source_id"])
            if row["action"] == "train":
                self.mapping[source_id] = names.index(row["target"])
            elif row["action"] == "review":
                self.review_ids.add(source_id)
            elif row["action"] != "exclude":
                raise ValueError(f"Unknown ontology action: {row['action']}")
        self.fingerprint = digest(self.rows)

    def check_source_names(self, names):
        if names != [r["source_name"] for r in self.rows]:
            raise ValueError("classes.json does not match the original 401-class ID order")

    def snapshot(self):
        return {"fingerprint": self.fingerprint, "names": self.names, "sources": self.rows}
