"""Canonical IJP application to RQ02 S5: 7 Individual Short + 21 Joint Short + Pool.

The S5 research engine needs a binary Short confirmation at each issue.  The
integrated runner previously used a source-count proxy, bypassing the paired
Joint layer.  This module builds the required IJP decision from released native
rows:

* every source uses only native observations with ``available_at <= issue``;
* Individual Short = Strong OR (Weak AND Count);
* each of the 21 unordered pairs is paired on the same native anchors;
* the Pool is the registered ``COMPLEMENT_AND_JOINT`` research policy by default;
* missing/insufficient evidence yields ``None`` (NOT_EVALUATED), never 0.

Thresholds are a deterministic research profile derived from *prior* adjacent
native deltas. They are explicitly not owner-approved. The module is a bridge
over ``research_adapters.ijp_kernel``; it does not duplicate triangle/joint/pool.
"""
from __future__ import annotations

import calendar
from itertools import combinations

import numpy as np
import pandas as pd

from research_adapters.ijp_kernel import (EligibilityError, NativeLevel, individual,
                                          joint, pool)
from research_adapters.rq01.pre_onset import RISK_SIGN, align_exact_native

SOURCES7 = ("CISS", "CLIFS", "DFC", "GFC", "VIX", "CG", "DSR-P")
SOURCE_COLUMNS = {"CISS": "ciss", "CLIFS": "clifs", "DFC": "dfc", "GFC": "gfc",
                  "VIX": "vix", "CG": "cg", "DSR-P": "dsr"}
AVAILABLE_COLUMNS = {"CISS": "available_at_CISS", "CLIFS": "available_at_CLIFS",
                     "DFC": "available_at_DFC", "GFC": "available_at_GFC",
                     "VIX": "available_at_VIX"}
OBSERVED_COLUMNS = {"CISS": "observed_at_CISS", "CLIFS": "observed_at_CLIFS",
                    "DFC": "observed_at_DFC", "GFC": "observed_at_GFC",
                    "VIX": "observed_at_VIX"}
SIGNS = {s: RISK_SIGN.get(s, 1) for s in SOURCES7}
PAIRS21 = tuple(combinations(SOURCES7, 2))
WINDOW_NATIVE = 5
PROFILE_LEVELS = 9
MIN_PRIOR_DELTAS = 4
RESEARCH_OWNER = "RESEARCH_IJP_S5_TRAILING_DELTA_PROFILE_V1"


def _month_end(month: str) -> str:
    y, m = int(month[:4]), int(month[5:7])
    return f"{y:04d}-{m:02d}-{calendar.monthrange(y, m)[1]:02d}"


def _text(value) -> str | None:
    if pd.isna(value):
        return None
    text = str(value)
    return text[:10] if len(text) >= 10 else text


def native_rows_from_raw(raw: pd.DataFrame, country: str) -> dict[str, list[NativeLevel]]:
    """Convert the integrated monthly raw panel into released NativeLevel rows.

    Monthly CISS/VIX/CLIFS/GFC snapshots and the DFC native quarter are kept
    distinct. CG/DSR-P lack a release column in the locked raw panel, so their
    month-end is used and labelled a research release proxy.
    """
    sub = raw[raw["country"] == country].sort_values("month")
    out: dict[str, list[NativeLevel]] = {}
    for source in SOURCES7:
        col = SOURCE_COLUMNS[source]
        if col not in sub.columns:
            out[source] = []
            continue
        rows = []
        for _, r in sub.iterrows():
            value = r.get(col)
            if pd.isna(value) or not np.isfinite(float(value)):
                continue
            if source in AVAILABLE_COLUMNS:
                available = _text(r.get(AVAILABLE_COLUMNS[source]))
                observed = _text(r.get(OBSERVED_COLUMNS[source]))
            else:
                available = _month_end(str(r["month"]))
                observed = _month_end(str(r["month"]))
            if available is None:
                continue
            period_end = observed or _month_end(str(r["month"]))
            if source == "DFC":
                period_id = _text(r.get("native_period_DFC")) or f"DFC-{str(r['month'])}"
            else:
                period_id = f"{source}-{str(r['month'])}"
            rows.append(NativeLevel(period_id=period_id, period_end=period_end,
                                    available_at=available, value=float(value)))
        # DFC repeats one native quarter across monthly rows: keep its first release.
        unique = {}
        for row in sorted(rows, key=lambda z: (z.period_end, z.available_at, z.period_id)):
            unique.setdefault(row.period_id, row)
        out[source] = list(unique.values())
    return out


def _prior_deltas(rows: list[NativeLevel], issue_end: str) -> np.ndarray:
    released = [r for r in rows if r.available_at <= issue_end and r.period_end <= issue_end]
    released.sort(key=lambda r: (r.period_end, r.period_id))
    vals = [r.value for r in released[-PROFILE_LEVELS:]]
    return np.asarray([abs(b - a) for a, b in zip(vals[:-1], vals[1:])], dtype=float)


def _profile(rows_by_source: dict[str, list[NativeLevel]], issue_end: str):
    """Prior-only source thresholds; returns None if a source lacks evidence."""
    profile = {}
    for s in SOURCES7:
        deltas = _prior_deltas(rows_by_source.get(s, []), issue_end)
        if len(deltas) < MIN_PRIOR_DELTAS:
            return None
        hi = max(float(np.quantile(deltas, 0.75)), 1e-9)
        lo = max(float(np.quantile(deltas, 0.35)), 0.0)
        profile[s] = {"hi": hi, "lo": min(lo, hi)}
    return profile


def s5_short_issue(rows_by_source: dict[str, list[NativeLevel]], issue_month: str) -> dict:
    """One binary S5 Short decision (or NOT_EVALUATED) from the IJP kernel."""
    issue_end = _month_end(issue_month)
    profile = _profile(rows_by_source, issue_end)
    if profile is None:
        return {"status": "NOT_EVALUATED", "reason": "INSUFFICIENT_PRIOR_NATIVE_DELTAS",
                "vote": None, "issue": issue_end}
    inds = {}
    for s in SOURCES7:
        try:
            asof = [r for r in rows_by_source.get(s, [])
                    if r.available_at <= issue_end and r.period_end <= issue_end]
            asof.sort(key=lambda r: (r.period_end, r.period_id))
            window = asof[-WINDOW_NATIVE:]
            if len(window) < 2:
                raise EligibilityError("requires >= 2 genuine native levels")
            res = individual([r.value for r in window], SIGNS[s],
                             theta_hi=profile[s]["hi"], theta_lo=profile[s]["lo"],
                             min_bases=2, count_k=3)
            inds[s] = int(res["vote"])
        except EligibilityError:
            continue
    joints = {}
    for a, b in PAIRS21:
        al = align_exact_native(rows_by_source.get(a, []), rows_by_source.get(b, []),
                                issue_month, 0, back=24)
        anchors = al["anchors"]
        if len(anchors) < 2:
            continue
        try:
            res = joint([(x.value, y.value) for x, y in anchors], (SIGNS[a], SIGNS[b]),
                        high_a=profile[a]["hi"], high_b=profile[b]["hi"],
                        low_a=profile[a]["lo"], low_b=profile[b]["lo"],
                        gate1_a=profile[a]["lo"], gate1_b=profile[b]["lo"],
                        min_bases=1, count_k=1, timing_valid=True)
            joints[(a, b)] = int(res["vote"])
        except EligibilityError:
            continue
    try:
        vote = pool(inds, joints, "COMPLEMENT_AND_JOINT", registry=SOURCES7)
    except EligibilityError:
        return {"status": "NOT_EVALUATED", "reason": "POOL_NOT_DERIVABLE", "vote": None,
                "issue": issue_end, "n_individual": len(inds), "n_joint": len(joints)}
    return {"status": "EVALUATED", "issue": issue_end, "vote": int(vote),
            "n_individual": len(inds), "n_joint": len(joints), "pool_policy": "COMPLEMENT_AND_JOINT",
            "threshold_profile": "RESEARCH_TRAILING_DELTA_Q75_Q35_NOT_OWNER_APPROVED",
            "owner_id": RESEARCH_OWNER}


def s5_short_votes(native_rows: dict[str, list[NativeLevel]], months: list[str],
                   *, min_native: int = 1) -> pd.DataFrame:
    """Evaluate the IJP S5 Short vote at every monthly issue (month-end)."""
    rows = []
    for month in sorted(set(months)):
        rec = s5_short_issue(native_rows, month)
        rows.append({"issue": pd.Timestamp(rec["issue"]), "status": rec["status"],
                     "vote": rec["vote"], "reason": rec.get("reason"),
                     "n_individual": rec.get("n_individual"), "n_joint": rec.get("n_joint")})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame[frame["status"] == "EVALUATED"].dropna(subset=["vote"]).reset_index(drop=True)
