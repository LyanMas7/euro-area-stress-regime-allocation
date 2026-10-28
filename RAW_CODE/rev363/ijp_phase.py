"""Canonical RQ01 six-phase tape (REV352/REV353) on released native rows.

This is the integration layer for `RQ01_6 pha`:

* five native source votes -> quorum (3,2,2,3,3) -> `PhaseController` persistence 2;
* the IJP PRE detector is passed as an auxiliary input, never an ENTRY gate;
* state flags, consensus, native ids, threshold profile and issue records are emitted;
* the legacy composite phase tape is not replaced.

The profile is the documented baseline percentile profile, frozen before replay and
labelled research-only. It is not an owner-approved threshold registry.
"""
from __future__ import annotations

import calendar
from collections import Counter

import pandas as pd

from research_adapters.rq01.five_state import (FiveStateEngine, PercentileProfile,
                                               SourceReference, ThresholdLedger,
                                               build_snapshot, compose_issue_record)
from research_adapters.rq01.pre_onset import RISK_SIGN
from research_adapters.rq01.common import SOURCES

from .ijp_pre import pre_issue
from .ijp_s5 import native_rows_from_raw

PROFILE_ID = "RQ01_SIX_PHASE_BASELINE_RESEARCH_V1"
THRESHOLDS = {"p_E": .90, "p_P": .75, "p_D": .50, "p_L": .25, "p_H": .75, "p_X": .75}
ROLE = "CANONICAL_RQ01_SIX_PHASE_RESEARCH_NOT_OWNER_APPROVED"


def _month(period_end: str) -> str:
    return str(period_end)[:7]


def build_country_engine(native: dict, panel: pd.DataFrame, country: str):
    """Build the frozen baseline snapshots from this country's labelled native rows."""
    labels = {(str(r["month"]), r["label"]) for _, r in panel[panel["country"] == country].iterrows()}
    lookup = {m: lab for m, lab in labels}
    snapshots = {}
    profile = PercentileProfile(PROFILE_ID, **THRESHOLDS)
    for source in SOURCES:
        vals_n, vals_s = [], []
        for row in native.get(source, []):
            label = lookup.get(_month(row.period_end))
            if label is None:
                continue
            value = RISK_SIGN[source] * float(row.value)
            (vals_n if label == "N" else vals_s).append(value)
        if not vals_n or not vals_s:
            raise ValueError(f"SNAPSHOT_CLASS_SUPPORT_MISSING:{country}:{source}")
        ref = SourceReference(source=source, values_N=tuple(vals_n), values_S=tuple(vals_s),
                              unit="native", quantile_method="linear")
        snapshots[source] = build_snapshot(ref, profile)
    ledger = ThresholdLedger(profile_by_source={s: profile for s in SOURCES}, locked_at="2005-01")
    return FiveStateEngine(snapshots=snapshots, ledger=ledger), profile


def _replay(raw: pd.DataFrame, panel_fit: pd.DataFrame, months_by_country: dict, out_dir,
            *, snapshot_source: str):
    rows, failures = [], {}
    for country in sorted(months_by_country):
        native = native_rows_from_raw(raw, country)
        try:
            engine, profile = build_country_engine(native, panel_fit, country)
        except ValueError as exc:
            failures[country] = str(exc)
            continue
        from research_adapters.rq01.pre_onset import PreOnsetDetector
        detector = PreOnsetDetector(require_new_native=True)
        for month in months_by_country[country]:
            pre = pre_issue(native, month)
            obs = detector.observe(month, pre.get("vote"), pre.get("native_ids", []))
            step = engine.step(month, native, pre_confirmation=obs["controller_confirmation"])
            rec = compose_issue_record(country, step, engine.controller,
                                       pre_individuals=pre.get("individuals"),
                                       pre_joints=pre.get("joints"),
                                       pre_pool=pre.get("pool"), pre_obs=obs)
            rec.update({"phase_engine": PROFILE_ID, "role": ROLE,
                        "threshold_profile_id": PROFILE_ID,
                        "profile_frozen_at": "2005-01",
                        "snapshot_source": snapshot_source,
                        "sample": "DEV" if str(month) <= "2019-12" else "OOS",
                        "pre_status": pre["status"], "pre_reason": pre.get("reason"),
                        "pre_vote": pre.get("vote")})
            rows.append(rec)
    return rows, failures


def phase_tape(raw: pd.DataFrame, panel: pd.DataFrame, out_dir) -> dict:
    """Write `RQ01_DEV_CANONICAL_PHASE.csv` beside the legacy phase path."""
    months_by_country = {c: sorted(panel.loc[panel["country"] == c, "month"].unique())
                         for c in sorted(panel["country"].unique())}
    rows, failures = _replay(raw, panel, months_by_country, out_dir,
                             snapshot_source="LABELLED_REPLAY_SAMPLE")
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "RQ01_DEV_CANONICAL_PHASE.csv", index=False)
    states = Counter(frame["state_after"]) if len(frame) else Counter()
    return {"n_rows": int(len(frame)), "n_countries": int(frame["country"].nunique()) if len(frame) else 0,
            "n_transitions": int((frame["state_before"] != frame["state_after"]).sum()) if len(frame) else 0,
            "state_counts": dict(states), "failures": failures,
            "profile_id": PROFILE_ID, "profile_status": "RESEARCH_BASELINE_NOT_OWNER_APPROVED",
            "role": ROLE, "evidence": "RQ01_DEV_CANONICAL_PHASE.csv"}


def phase_tape_split(raw: pd.DataFrame, panel_dev: pd.DataFrame, panel_oos: pd.DataFrame,
                     out_dir) -> dict:
    """DEV-frozen snapshot replay that continues through OOS without reading OOS labels.

    Snapshots and the threshold ledger are fitted on DEV native rows only; the
    state machine then runs continuously over DEV+OOS months.  OOS phase labels
    therefore never enter the snapshot, quantile or threshold construction.
    """
    scope = pd.concat([panel_dev[["country", "month"]], panel_oos[["country", "month"]]],
                      ignore_index=True).drop_duplicates(["country", "month"])
    months_by_country = {c: sorted(scope.loc[scope["country"] == c, "month"].unique())
                         for c in sorted(scope["country"].unique())}
    rows, failures = _replay(raw, panel_dev, months_by_country, out_dir,
                             snapshot_source="DEV_2005_07_TO_2019_12_FROZEN")
    frame = pd.DataFrame(rows)
    dev = frame[frame["sample"] == "DEV"] if len(frame) else frame
    oos = frame[frame["sample"] == "OOS"] if len(frame) else frame
    frame.to_csv(out_dir / "RQ01_CANONICAL_PHASE_FULL.csv", index=False)
    dev.to_csv(out_dir / "RQ01_DEV_CANONICAL_PHASE.csv", index=False)
    oos.to_csv(out_dir / "RQ01_OOS_CANONICAL_PHASE.csv", index=False)
    states = Counter(frame["state_after"]) if len(frame) else Counter()
    oos_states = Counter(oos["state_after"]) if len(oos) else Counter()
    return {"n_rows": int(len(frame)), "n_dev_rows": int(len(dev)), "n_oos_rows": int(len(oos)),
            "n_countries": int(frame["country"].nunique()) if len(frame) else 0,
            "n_transitions": int((frame["state_before"] != frame["state_after"]).sum()) if len(frame) else 0,
            "state_counts": dict(states), "oos_state_counts": dict(oos_states),
            "failures": failures, "profile_id": PROFILE_ID,
            "profile_status": "RESEARCH_BASELINE_NOT_OWNER_APPROVED",
            "oos_snapshot_policy": "DEV_FROZEN_NO_OOS_LABELS",
            "role": ROLE,
            "evidence": "RQ01_DEV_CANONICAL_PHASE.csv / RQ01_OOS_CANONICAL_PHASE.csv"}
