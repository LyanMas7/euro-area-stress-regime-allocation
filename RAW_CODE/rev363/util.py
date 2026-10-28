"""Shared helpers: hashing, run identity, canonical json."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def sha256_json(value) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def run_id(tag: str) -> str:
    return f"{tag}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def jdump(obj, path: Path) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, default=str, ensure_ascii=False) + "\n",
                          encoding="utf-8")


def jload(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fmt(x, nd=6):
    """Deterministic numeric formatting for report cards."""
    if x is None:
        return "None"
    if isinstance(x, (bool, np.bool_)):
        return str(bool(x))
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        if not np.isfinite(x):
            return str(float(x))
        return f"{float(x):.{nd}f}"
    return str(x)