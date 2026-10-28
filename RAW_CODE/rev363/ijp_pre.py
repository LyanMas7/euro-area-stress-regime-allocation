"""Canonical RQ01 PRE-ONSET tape from the IJP kernel (research profile).

This is the RQ01 application of the same kernel used by RQ02 S5:

* five RQ01 sources, one Individual per source;
* ten unordered pairs, one registered EXACT(0, delta=0) Joint configuration each;
* six Pool masks with one locked research policy (COMPLEMENT_AND_JOINT);
* PRE confirmation requires two consecutive monthly votes with new native evidence.

Thresholds are a deterministic research profile derived from prior adjacent
native deltas. They are not owner-approved and every row is labelled accordingly.
The ENTRY..EXIT phase controller is not touched or replaced.
"""
from __future__ import annotations

import calendar

import numpy as np
import pandas as pd

from research_adapters.rq01.pre_onset import (PreIndividualOwner, PreJointOwner,
                                               PreOnsetDetector, pre_individual,
                                               pre_joint, pre_pools)
from research_adapters.rq01.pairs import LagPolicy
from research_adapters.rq01.motif_issue_packet import (evaluate_pair_motif,
                                                       research_pair_thresholds,
                                                       research_state_and_terminal)

from . import rq01_native_ijp as N
from .ijp_s5 import native_rows_from_raw

SOURCES5 = ("CISS", "CLIFS", "DFC", "GFC", "VIX")
PAIRS10 = tuple((SOURCES5[i], SOURCES5[j]) for i in range(5) for j in range(i + 1, 5))
PROFILE_LEVELS = 9
MIN_PRIOR_DELTAS = 4
OWNER = "RESEARCH_IJP_PRE_TRAILING_DELTA_PROFILE_V1"
POLICY = "COMPLEMENT_AND_JOINT"


def _month_end(month: str) -> str:
    y, m = int(month[:4]), int(month[5:7])
    return f"{y:04d}-{m:02d}-{calendar.monthrange(y, m)[1]:02d}"


def _profile(rows_by_source: dict, issue_end: str):
    profile = {}
    for s in SOURCES5:
        rows = [r for r in rows_by_source.get(s, [])
                if r.available_at <= issue_end and r.period_end <= issue_end]
        rows.sort(key=lambda r: (r.period_end, r.period_id))
        vals = [r.value for r in rows[-PROFILE_LEVELS:]]
        deltas = [abs(b - a) for a, b in zip(vals[:-1], vals[1:])]
        if len(deltas) < MIN_PRIOR_DELTAS:
            return None
        hi = max(float(np.quantile(deltas, 0.75)), 1e-9)
        lo = max(float(np.quantile(deltas, 0.35)), 0.0)
        profile[s] = {"hi": hi, "lo": min(lo, hi)}
    return profile


def pre_issue(rows_by_source: dict, issue_month: str, *, back: int = 24) -> dict:
    """One PRE issue from the IJP kernel; no phase/ENTRY state is read or changed."""
    issue_end = _month_end(issue_month)
    profile = _profile(rows_by_source, issue_end)
    if profile is None:
        return {"status": "NOT_EVALUATED", "execution_status": "NOT_EVALUATED",
                "reason": "INSUFFICIENT_PRIOR_NATIVE_DELTAS", "issue": issue_month, "native_ids": []}
    inds = {}
    for s in SOURCES5:
        owner = PreIndividualOwner(s, OWNER, profile[s]["hi"], profile[s]["lo"],
                                   min_bases=2, count_k=3, window_native=5)
        inds[s] = pre_individual(rows_by_source.get(s, []), issue_month, owner)
    # B2: paired Joint votes come from the motif packet; PRE Individual stays native.
    series_cache = {s: N._monthly_series(rows_by_source.get(s, []), issue_month)
                    for s in SOURCES5}
    joints = []
    for a, b in PAIRS10:
        sa, sb = series_cache[a], series_cache[b]
        _, used = N._matched(sa, sb, N.FREQUENCY[a], N.FREQUENCY[b])
        aa = [(r.period_id, r.value) for _, r, _ in used]
        bb = [(r.period_id, r.value) for _, _, r in used]
        state_a, term_a = research_state_and_terminal(
            a, [r.value for _, r in sa], profile[a])
        state_b, term_b = research_state_and_terminal(
            b, [r.value for _, r in sb], profile[b])
        th = research_pair_thresholds(min(len(aa), len(bb)), profile[a], profile[b])
        res = evaluate_pair_motif(
            pair=(a, b), series_a=aa, series_b=bb, motif="UU",
            lag=LagPolicy("EXACT", 0, 0, None, OWNER),
            state_a=state_a, state_b=state_b, term_a=term_a, term_b=term_b,
            gate1_a=th["gate1_a"], gate1_b=th["gate1_b"],
            high_a=th["high_a"], low_a=th["low_a"],
            high_b=th["high_b"], low_b=th["low_b"],
            min_bases=th["min_bases"], count_k=th["count_k"])
        evaluated = res["status"] in ("PASS", "FAIL")
        joints.append({
            "status": "EVALUATED" if evaluated else "NOT_EVALUATED",
            **({"vote": res["vote"]} if evaluated else {}),
            "pair": (a, b), "leader": a, "follower": b,
            "lag_months": 0, "lag_kind": "EXACT", "delta": 0,
            "gate1": res.get("gate1"), "gate2": res.get("gate2"),
            "paired_count": res.get("count_lower"),
            "native_ids": sorted({pid for pid, _ in aa} | {pid for pid, _ in bb}),
            "owner_hash": OWNER,
            "reason": None if res["status"] in ("PASS", "FAIL") else res.get("reason")})
    pooled = pre_pools(inds, joints, locked_policy=POLICY)
    native_ids = set()
    for rec in list(inds.values()) + joints:
        native_ids.update(rec.get("native_ids", []) or [])
    return {"status": pooled["status"], **({"vote": pooled["vote"]} if "vote" in pooled else {}),
            "execution_status": pooled["execution_status"],
            "reason": None if pooled["status"] == "EVALUATED" else "POOL_NOT_DERIVABLE",
            "issue": issue_month, "native_ids": sorted(native_ids),
            "n_individual": pooled["n_eligible_individuals"],
            "n_joint": pooled["n_eligible_joints"],
            "pool_policy": POLICY,
            "masks": pooled["masks"],
            "individuals": inds, "joints": joints, "pool": pooled,
            "threshold_profile": "RESEARCH_TRAILING_DELTA_Q75_Q35_NOT_OWNER_APPROVED",
            "owner_id": OWNER}


def pre_tape(raw: pd.DataFrame, panel: pd.DataFrame, out_dir) -> dict:
    """Write the RQ01 canonical IJP PRE tape beside the existing phase tape."""
    rows = []
    for country in sorted(panel["country"].unique()):
        native = native_rows_from_raw(raw, country)
        detector = PreOnsetDetector(require_new_native=True)
        months = sorted(panel.loc[panel["country"] == country, "month"].unique())
        for month in months:
            rec = pre_issue(native, month)
            obs = detector.observe(month, rec.get("vote"), rec.get("native_ids", []))
            # two-layer row: `pre_vote` is populated iff the PRE decision was EXECUTED
            rows.append({"country": country, "issue": month, "status": rec["status"],
                         "execution_status": "EXECUTED" if "vote" in rec else "NOT_EVALUATED",
                         "pre_vote": rec.get("vote"), "pre_confirmed": obs["confirmed"],
                         "pre_controller": obs["controller_vote"],
                         "pre_controller_status": obs["controller_execution_status"],
                         "pre_controller_reason": obs["controller_non_vote_reason"],
                         "reason": rec.get("reason") or obs.get("reason"),
                         "n_individual": rec.get("n_individual"),
                         "n_joint": rec.get("n_joint"),
                         "pool_policy": POLICY, "owner_id": OWNER,
                         "threshold_profile": rec.get("threshold_profile"),
                         "source_scope": "EA7", "role": "PRE_AUXILIARY_NOT_AN_ENTRY_GATE"})
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "RQ01_DEV_IJP_PRE.csv", index=False)
    return {"n_rows": int(len(frame)),
            "n_evaluated": int((frame.get("status") == "EVALUATED").sum()) if len(frame) else 0,
            "n_pre_confirmed": int(frame.get("pre_confirmed", pd.Series(dtype=bool)).sum()) if len(frame) else 0,
            "n_countries": int(frame["country"].nunique()) if len(frame) else 0,
            "evidence": "RQ01_DEV_IJP_PRE.csv",
            "pool_policy": POLICY,
            "role": "PRE_AUXILIARY_NOT_AN_ENTRY_GATE",
            "owner_status": "RESEARCH_PROFILE_NOT_OWNER_APPROVED"}
