"""Canonical threshold loader.

Reads the owner-selected RQ01/RQ02/RQ03 configurations from the sibling
``THRESHOLD`` directory and exposes them as validated, immutable objects.

Every selected number lives in THRESHOLD, never in this module. A missing or
malformed file raises :class:`ThresholdError` with the exact expected path so
that the pipeline fails loudly instead of falling back to a hard-coded value.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class ThresholdError(RuntimeError):
    """Raised when a selected threshold configuration is absent or invalid."""


REQUIRED = {
    "RQ01": ("RQ01", "rq01_selected.json"),
    "RQ02": ("RQ02", "rq02_selected.json"),
    "RQ03": ("RQ03", "rq03_selected.json"),
}


def _load_one(root: Path, rq: str) -> dict:
    sub, name = REQUIRED[rq]
    path = root / sub / name
    if not path.is_file():
        raise ThresholdError(
            f"missing selected {rq} configuration: {path} "
            f"(expected THRESHOLD/{sub}/{name})"
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # explicit, not silent
        raise ThresholdError(f"{path} is not valid JSON: {exc}") from exc
    if data.get("rq") != rq:
        raise ThresholdError(f"{path} declares rq={data.get('rq')!r}, expected {rq!r}")
    return data


@dataclass(frozen=True)
class RQ01Selection:
    model: str
    operator: str
    rule: str
    thresholds: Mapping[str, float]
    joint_or: tuple[str, ...]
    joint_not: tuple[str, ...]
    status: str
    provenance: Mapping[str, Any]

    def threshold(self, country: str) -> float:
        try:
            return float(self.thresholds[country])
        except KeyError as exc:
            raise ThresholdError(f"RQ01 has no selected threshold for {country!r}") from exc


@dataclass(frozen=True)
class RQ02Selection:
    entry_model: str
    entry_horizon: str
    entry_horizon_months: int
    entry_threshold: float
    exit_horizon_months: int
    exit_threshold: float
    horizon_register: Mapping[str, float]
    status: str
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class RQ03Selection:
    active_policy: str
    fee_one_way: float
    chosen_standard: Mapping[str, float]
    status: str
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class ThresholdBundle:
    root: Path
    rq01: RQ01Selection
    rq02: RQ02Selection
    rq03: RQ03Selection
    raw: Mapping[str, Any]

    def as_register(self) -> dict:
        """Flat, JSON-serialisable register written next to every run."""
        return {
            "threshold_root": str(self.root),
            "RQ01": {
                "model": self.rq01.model,
                "operator": self.rq01.operator,
                "rule": self.rq01.rule,
                "thresholds": dict(self.rq01.thresholds),
                "joint_or": list(self.rq01.joint_or),
                "joint_not": list(self.rq01.joint_not),
                "status": self.rq01.status,
            },
            "RQ02": {
                "entry_model": self.rq02.entry_model,
                "entry_horizon": self.rq02.entry_horizon,
                "entry_horizon_months": self.rq02.entry_horizon_months,
                "entry_threshold": self.rq02.entry_threshold,
                "exit_horizon_months": self.rq02.exit_horizon_months,
                "exit_threshold": self.rq02.exit_threshold,
                "horizon_register": dict(self.rq02.horizon_register),
                "status": self.rq02.status,
            },
            "RQ03": {
                "active_policy": self.rq03.active_policy,
                "fee_one_way": self.rq03.fee_one_way,
                "chosen_standard": dict(self.rq03.chosen_standard),
                "status": self.rq03.status,
            },
        }


def load(threshold_root: str | Path) -> ThresholdBundle:
    root = Path(threshold_root).expanduser().resolve()
    if not root.is_dir():
        raise ThresholdError(f"threshold root does not exist: {root}")

    r1 = _load_one(root, "RQ01")
    r2 = _load_one(root, "RQ02")
    r3 = _load_one(root, "RQ03")

    thresholds = {str(k): float(v) for k, v in r1["thresholds"].items()}
    if not thresholds:
        raise ThresholdError("RQ01 selection contains no country thresholds")
    for country, value in thresholds.items():
        if not 0.0 <= value <= 1.0:
            raise ThresholdError(f"RQ01 threshold for {country} outside [0,1]: {value}")

    rq01 = RQ01Selection(
        model=str(r1["model"]),
        operator=str(r1["operator"]),
        rule=str(r1["rule"]),
        thresholds=thresholds,
        joint_or=tuple(r1.get("joint_features", {}).get("or", ())),
        joint_not=tuple(r1.get("joint_features", {}).get("not", ())),
        status=str(r1.get("status", "")),
        provenance=dict(r1.get("provenance", {})),
    )

    entry = r2["entry"]
    exit_ = r2["exit"]
    rq02 = RQ02Selection(
        entry_model=str(entry["model"]),
        entry_horizon=str(entry["horizon"]),
        entry_horizon_months=int(entry["horizon_months"]),
        entry_threshold=float(entry["threshold"]),
        exit_horizon_months=int(exit_["horizon_months"]),
        exit_threshold=float(exit_["threshold"]),
        horizon_register={str(k): float(v) for k, v in r2.get("horizon_register", {}).items()},
        status=str(r2.get("status", "")),
        provenance=dict(r2.get("provenance", {})),
    )
    if not 0.0 <= rq02.entry_threshold <= 1.0:
        raise ThresholdError(f"RQ02 entry threshold outside [0,1]: {rq02.entry_threshold}")
    if not 0.0 <= rq02.exit_threshold <= 1.0:
        raise ThresholdError(f"RQ02 exit threshold outside [0,1]: {rq02.exit_threshold}")

    rq03 = RQ03Selection(
        active_policy=str(r3["active_policy"]),
        fee_one_way=float(r3["fee_one_way"]),
        chosen_standard={str(k): float(v) for k, v in r3.get("chosen_standard", {}).items()},
        status=str(r3.get("status", "")),
        provenance=dict(r3.get("provenance", {})),
    )

    return ThresholdBundle(root=root, rq01=rq01, rq02=rq02, rq03=rq03,
                           raw={"RQ01": r1, "RQ02": r2, "RQ03": r3})


def describe(bundle: ThresholdBundle) -> str:
    """One-screen provenance summary printed by the entrypoint."""
    r1, r2, r3 = bundle.rq01, bundle.rq02, bundle.rq03
    lines = [
        f"threshold root : {bundle.root}",
        f"RQ01           : {r1.model} | {len(r1.thresholds)} country thresholds | "
        f"{min(r1.thresholds.values()):.4f}..{max(r1.thresholds.values()):.4f} | {r1.status}",
        f"RQ02           : entry {r2.entry_horizon} ({r2.entry_horizon_months}m) "
        f"t={r2.entry_threshold:.6f} | exit {r2.exit_horizon_months}m "
        f"t={r2.exit_threshold:.6f} | {r2.status}",
        f"RQ03           : {r3.active_policy} | fee {r3.fee_one_way:.5f} | {r3.status}",
    ]
    return "\n".join(lines)
