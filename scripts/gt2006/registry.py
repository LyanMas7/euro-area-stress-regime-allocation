"""Load and validate the GT2006 academic presentation style register."""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
REGISTER_PATH = ROOT / "style_register.json"


@functools.lru_cache(maxsize=1)
def load_register() -> dict:
    with REGISTER_PATH.open("r", encoding="utf-8") as fh:
        register = json.load(fh)
    _validate(register)
    return register


def _validate(register: dict) -> None:
    required = [
        "register_version",
        "reference",
        "page",
        "typography",
        "headings",
        "panels",
        "axes",
        "lines",
        "stacked_areas",
        "table",
        "statistics",
        "notes",
        "export",
        "qa",
    ]
    missing = [key for key in required if key not in register]
    if missing:
        raise ValueError(f"Style register missing sections: {missing}")
    weights = register["qa"]["criteria"]
    if abs(sum(weights.values()) - 1.0) > 1e-9:
        raise ValueError("QA criteria weights must sum to 1.0")
    sizes = register["typography"]["sizes_pt"]
    if sizes["tick_label"] < 6.0:
        raise ValueError("Tick labels below the legibility floor")


REGISTER = load_register()


def get(path: str, default: Any = None) -> Any:
    """Read a dotted path from the register."""
    node: Any = REGISTER
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


PT_PER_INCH = 72.0


def inches(points: float) -> float:
    return points / PT_PER_INCH


def points(inches_value: float) -> float:
    return inches_value * PT_PER_INCH
