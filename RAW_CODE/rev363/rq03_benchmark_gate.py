"""Authoritative RQ03 benchmark capital-weight trigger and base-snapshot lifecycle.

This module implements the user-confirmed R25 rule for the five benchmarks only:

    request a restructure iff at least two of the six current capital weights differ
    from that benchmark's stored base weights by at least 0.05 of total NAV.

The comparison is inclusive at exactly five percentage points (with a small absolute
floating-point tolerance).  Native benchmark geometry, risk, utility, RQ01/RQ02 and
Main lifecycle signals are deliberately absent from this API so they cannot become
additional discretionary activation conditions.

``BaseSnapshot`` is immutable.  ``BenchmarkBaseState`` replaces it only after a
completed, confirmed simulated fill.  HOLD, BLOCKED, no-fill and hard-risk repair do
not rebase the discretionary benchmark reference.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


ASSETS = ("E1", "E2", "E3", "B1", "B2", "C1")
BENCHMARKS = ("EW", "IVOL", "MVP", "ERC", "MaxDiv")
WEIGHT_BAND_THRESHOLD = 0.05
MIN_QUALIFYING_ASSETS = 2
COMPARISON_ATOL = 1e-12
WEIGHT_SUM_ATOL = 1e-8
FILL_ATOL = 1e-12
TRIGGER_RULE_ID = "RQ03_BENCHMARK_SIX_WEIGHT_5PP_MIN2_V1"


def _weight_error(weights: Sequence[float] | None) -> str | None:
    if weights is None:
        return "MISSING"
    try:
        w = np.asarray(weights, dtype=float)
    except (TypeError, ValueError):
        return "NON_NUMERIC"
    if w.shape != (len(ASSETS),):
        return f"EXPECTED_{len(ASSETS)}_WEIGHTS_GOT_{w.shape}"
    if not np.isfinite(w).all():
        return "NON_FINITE"
    if np.any(w < -COMPARISON_ATOL):
        return "NEGATIVE_WEIGHT"
    if np.any(w > 1.0 + COMPARISON_ATOL):
        return "WEIGHT_ABOVE_ONE"
    if not math.isclose(float(w.sum()), 1.0, rel_tol=0.0, abs_tol=WEIGHT_SUM_ATOL):
        return f"WEIGHTS_DO_NOT_SUM_TO_ONE:{float(w.sum()):.17g}"
    return None


def validate_weights(weights: Sequence[float], *, name: str = "weights") -> np.ndarray:
    """Return a defensive float copy or raise a precise validation error."""
    error = _weight_error(weights)
    if error is not None:
        raise ValueError(f"{name}:{error}")
    return np.asarray(weights, dtype=float).copy()


@dataclass(frozen=True)
class BaseSnapshot:
    """One benchmark's immutable reference between confirmed restructuring fills."""

    strategy: str
    effective_at: str
    weights: tuple[float, ...]
    source: str
    confirmed: bool
    nav: float
    holdings_value: tuple[float, ...]
    version: int
    provenance: str
    execution_id: str

    def __post_init__(self):
        if self.strategy not in BENCHMARKS:
            raise ValueError("BASE_UNKNOWN_BENCHMARK")
        validate_weights(self.weights, name="base_weights")
        h = np.asarray(self.holdings_value, dtype=float)
        if h.shape != (len(ASSETS),) or not np.isfinite(h).all() or np.any(h < -COMPARISON_ATOL):
            raise ValueError("BASE_INVALID_HOLDINGS_VALUE")
        if not (math.isfinite(self.nav) and self.nav > 0.0):
            raise ValueError("BASE_INVALID_NAV")
        if not math.isclose(float(h.sum()), float(self.nav), rel_tol=0.0,
                            abs_tol=max(1e-10, abs(float(self.nav)) * 1e-10)):
            raise ValueError("BASE_HOLDINGS_NAV_MISMATCH")
        if not isinstance(self.version, int) or self.version < 1:
            raise ValueError("BASE_INVALID_VERSION")
        if not self.effective_at or not self.source or not self.provenance or not self.execution_id:
            raise ValueError("BASE_MISSING_PROVENANCE")

    def record(self) -> dict:
        return {
            "strategy": self.strategy,
            "effective_at": self.effective_at,
            "weights": {a: float(self.weights[i]) for i, a in enumerate(ASSETS)},
            "source": self.source,
            "confirmed": bool(self.confirmed),
            "nav": float(self.nav),
            "holdings_value": {a: float(self.holdings_value[i]) for i, a in enumerate(ASSETS)},
            "version": int(self.version),
            "provenance": self.provenance,
            "execution_id": self.execution_id,
        }

    @property
    def snapshot_id(self) -> str:
        payload = json.dumps(self.record(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_snapshot(strategy: str, effective_at, weights: Sequence[float], *, source: str,
                  confirmed: bool, nav: float, version: int, provenance: str,
                  execution_id: str) -> BaseSnapshot:
    w = validate_weights(weights, name="base_weights")
    nav = float(nav)
    return BaseSnapshot(
        strategy=strategy,
        effective_at=str(effective_at),
        weights=tuple(float(x) for x in w),
        source=str(source),
        confirmed=bool(confirmed),
        nav=nav,
        holdings_value=tuple(float(x) for x in nav * w),
        version=int(version),
        provenance=str(provenance),
        execution_id=str(execution_id),
    )


def _failed_gate(strategy: str, status: str, reason: str, base: BaseSnapshot | None,
                 current_weights=None) -> dict:
    current = None
    if current_weights is not None:
        try:
            a = np.asarray(current_weights, dtype=float)
            if a.shape == (len(ASSETS),):
                current = tuple(float(x) if np.isfinite(x) else None for x in a)
        except (TypeError, ValueError):
            current = None
    return {
        "rule_id": TRIGGER_RULE_ID,
        "strategy": strategy,
        "status": status,
        "reason": reason,
        "triggered": False,
        "qualifying_count": None,
        "qualifying_assets": (),
        "qualifying_mask": (None,) * len(ASSETS),
        "deviations": (None,) * len(ASSETS),
        "base_weights": tuple(base.weights) if base is not None else None,
        "current_weights": current,
        "base_snapshot_id": base.snapshot_id if base is not None else None,
        "threshold": WEIGHT_BAND_THRESHOLD,
        "minimum_assets": MIN_QUALIFYING_ASSETS,
    }


def assess_weight_band(strategy: str, base: BaseSnapshot | None, current_weights: Sequence[float] | None,
                       *, current_confirmed: bool = True) -> dict:
    """Evaluate the fixed benchmark trigger; invalid inputs fail closed without a count."""
    if strategy not in BENCHMARKS:
        return _failed_gate(strategy, "FAIL_CLOSED_UNKNOWN_BENCHMARK", "UNKNOWN_BENCHMARK", base,
                            current_weights)
    if base is None:
        return _failed_gate(strategy, "FAIL_CLOSED_NO_BASE_SNAPSHOT", "NO_BASE_SNAPSHOT", None,
                            current_weights)
    if base.strategy != strategy:
        return _failed_gate(strategy, "FAIL_CLOSED_BASE_STRATEGY_MISMATCH", "BASE_STRATEGY_MISMATCH", base,
                            current_weights)
    if not base.confirmed:
        return _failed_gate(strategy, "FAIL_CLOSED_BASE_UNCONFIRMED", "BASE_UNCONFIRMED", base,
                            current_weights)
    if not current_confirmed:
        return _failed_gate(strategy, "FAIL_CLOSED_CURRENT_WEIGHTS_UNCONFIRMED",
                            "CURRENT_WEIGHTS_UNCONFIRMED", base, current_weights)
    error = _weight_error(current_weights)
    if error is not None:
        return _failed_gate(strategy, "FAIL_CLOSED_CURRENT_WEIGHTS_INVALID", error, base, current_weights)

    current = np.asarray(current_weights, dtype=float)
    base_w = np.asarray(base.weights, dtype=float)
    deviations = np.abs(current - base_w)
    qualifying = (deviations > WEIGHT_BAND_THRESHOLD) | np.isclose(
        deviations, WEIGHT_BAND_THRESHOLD, rtol=0.0, atol=COMPARISON_ATOL)
    count = int(qualifying.sum())
    triggered = count >= MIN_QUALIFYING_ASSETS
    return {
        "rule_id": TRIGGER_RULE_ID,
        "strategy": strategy,
        "status": "AUTO_RESTRUCTURE_REQUEST" if triggered else "HOLD_WEIGHT_BAND",
        "reason": "MINIMUM_TWO_ASSETS_CONFIRMED" if triggered else "FEWER_THAN_TWO_ASSETS_CONFIRMED",
        "triggered": bool(triggered),
        "qualifying_count": count,
        "qualifying_assets": tuple(a for a, q in zip(ASSETS, qualifying) if q),
        "qualifying_mask": tuple(bool(x) for x in qualifying),
        "deviations": tuple(float(x) for x in deviations),
        "base_weights": tuple(float(x) for x in base_w),
        "current_weights": tuple(float(x) for x in current),
        "base_snapshot_id": base.snapshot_id,
        "threshold": WEIGHT_BAND_THRESHOLD,
        "minimum_assets": MIN_QUALIFYING_ASSETS,
    }


@dataclass
class BenchmarkBaseState:
    """Mutable holder whose value is always an immutable ``BaseSnapshot``."""

    strategy: str
    snapshot: BaseSnapshot | None = None

    def __post_init__(self):
        if self.strategy not in BENCHMARKS:
            raise ValueError("STATE_UNKNOWN_BENCHMARK")
        if self.snapshot is not None and self.snapshot.strategy != self.strategy:
            raise ValueError("STATE_BASE_STRATEGY_MISMATCH")

    def record_completed_fill(self, weights: Sequence[float], *, effective_at, nav_after_cost: float,
                              turnover: float, execution_id: str, execution_kind: str,
                              trade_confirmed: bool, completed: bool, gate_status: str | None,
                              provenance: str) -> dict:
        """Establish/rebase only for an initial fill or a gate-requested completed fill."""
        before = self.snapshot
        base_result = {
            "updated": False,
            "rebase_applied": False,
            "update_kind": "NONE",
            "reason": None,
            "before_snapshot_id": before.snapshot_id if before is not None else None,
            "after_snapshot_id": before.snapshot_id if before is not None else None,
        }
        if not (trade_confirmed and completed):
            return {**base_result, "reason": "FILL_NOT_CONFIRMED_OR_INCOMPLETE"}
        if not (math.isfinite(float(turnover)) and float(turnover) > FILL_ATOL):
            return {**base_result, "reason": "NO_COMPLETED_TRADE"}
        if execution_kind == "INITIAL_ALLOCATION":
            if before is not None:
                return {**base_result, "reason": "INITIAL_BASE_ALREADY_EXISTS"}
            update_kind = "INITIAL_BASE_ESTABLISHED"
        elif execution_kind == "WEIGHT_BAND_RESTRUCTURE":
            if before is None:
                return {**base_result, "reason": "CANNOT_REBASE_WITHOUT_INITIAL_BASE"}
            if gate_status != "AUTO_RESTRUCTURE_REQUEST":
                return {**base_result, "reason": "NO_AUTHORITATIVE_GATE_REQUEST"}
            update_kind = "CONFIRMED_REBASE"
        else:
            return {**base_result, "reason": "EXECUTION_KIND_NOT_REBASE_ELIGIBLE"}

        version = 1 if before is None else before.version + 1
        snap = make_snapshot(
            self.strategy,
            effective_at,
            weights,
            source=f"{execution_kind}:SIMULATED_FILL_CONFIRMED_BY_ENGINE",
            confirmed=True,
            nav=float(nav_after_cost),
            version=version,
            provenance=provenance,
            execution_id=execution_id,
        )
        self.snapshot = snap
        return {
            "updated": True,
            "rebase_applied": execution_kind == "WEIGHT_BAND_RESTRUCTURE",
            "update_kind": update_kind,
            "reason": "COMPLETED_CONFIRMED_SIMULATED_FILL",
            "before_snapshot_id": before.snapshot_id if before is not None else None,
            "after_snapshot_id": snap.snapshot_id,
        }


def ledger_fields(gate: dict, base_before: BaseSnapshot | None, base_after: BaseSnapshot | None, *,
                  target_weights: Sequence[float] | None, execution_confirmed: bool,
                  execution_kind: str, base_update: dict) -> dict:
    """Flatten gate/state provenance for a weekly CSV ledger."""
    target = None
    if target_weights is not None:
        a = np.asarray(target_weights, dtype=float)
        if a.shape == (len(ASSETS),) and np.isfinite(a).all():
            target = a
    out = {
        "benchmark_trigger_rule": TRIGGER_RULE_ID,
        "weight_band_threshold": WEIGHT_BAND_THRESHOLD,
        "weight_band_threshold_pp": 100.0 * WEIGHT_BAND_THRESHOLD,
        "weight_band_min_assets": MIN_QUALIFYING_ASSETS,
        "weight_gate_status": gate["status"],
        "weight_gate_reason": gate["reason"],
        "restructure_request": bool(gate["triggered"]),
        "qualifying_count": gate["qualifying_count"],
        "qualifying_assets": ";".join(gate["qualifying_assets"]),
        "base_snapshot_id_before": base_before.snapshot_id if base_before is not None else None,
        "base_version_before": base_before.version if base_before is not None else None,
        "base_effective_at_before": base_before.effective_at if base_before is not None else None,
        "base_source_before": base_before.source if base_before is not None else None,
        "base_confirmed_before": base_before.confirmed if base_before is not None else None,
        "base_nav_before": base_before.nav if base_before is not None else None,
        "execution_confirmed": bool(execution_confirmed),
        "execution_kind": execution_kind,
        "base_updated": bool(base_update.get("updated", False)),
        "rebase_applied": bool(base_update.get("rebase_applied", False)),
        "base_update_kind": base_update.get("update_kind", "NONE"),
        "base_update_reason": base_update.get("reason"),
        "base_snapshot_id_after": base_after.snapshot_id if base_after is not None else None,
        "base_version_after": base_after.version if base_after is not None else None,
        "base_effective_at_after": base_after.effective_at if base_after is not None else None,
        "base_source_after": base_after.source if base_after is not None else None,
        "base_confirmed_after": base_after.confirmed if base_after is not None else None,
        "base_nav_after": base_after.nav if base_after is not None else None,
    }
    base_w = np.asarray(base_before.weights, float) if base_before is not None else None
    after_w = np.asarray(base_after.weights, float) if base_after is not None else None
    current = gate.get("current_weights")
    dev = gate.get("deviations")
    qmask = gate.get("qualifying_mask")
    for i, asset in enumerate(ASSETS):
        out[f"base_w_{asset}"] = float(base_w[i]) if base_w is not None else None
        out[f"current_w_{asset}"] = current[i] if current is not None else None
        out[f"abs_dev_{asset}"] = dev[i] if dev is not None else None
        out[f"qualifies_{asset}"] = qmask[i] if qmask is not None else None
        out[f"target_w_{asset}"] = float(target[i]) if target is not None else None
        out[f"base_after_w_{asset}"] = float(after_w[i]) if after_w is not None else None
    return out
