"""W02-only proof-carrying RQ02 label, native-frequency and threshold primitives.

Scientific classification: DEV_RESEARCH_PROXY / NOT_OWNER_CERTIFIED.
No OOS supervised fit, label-inferred pseudo issuance, or new RQ01/RQ03 owner logic.
All functions operate on explicit inputs; do not load sealed OOS data.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import isfinite

import numpy as np
import pandas as pd

DEV_START = "2005-01"
DEV_END = "2019-12"
OOS_START = "2020-01"
HORIZON_MONTHS = {"H15": 45, "H12": 36, "H10": 30, "H8": 24, "H6": 18}
NATIVE_FREQUENCY = {"CISS": "W", "CLIFS": "M", "DFC": "Q", "GFC": "M",
                    "VIX": "D", "CG": "Q", "DSR-P": "Q"}
LONG_SOURCES = ("CG", "DSR-P")
SHORT_SOURCES = ("CISS", "CLIFS", "DFC", "GFC", "VIX", "CG", "DSR-P")
RESEARCH_LABEL_LAG_MONTHS = 3  # NOT authentic official publication timing


def add_months(m: str, steps: int) -> str:
    p = pd.Period(m, freq="M") + int(steps)
    return str(p)


def target_end(m: str, horizon: int) -> str:
    return add_months(m, horizon)


def label_mature_by_freeze(m: str, *, dev_end: str = DEV_END,
                           research_lag: int = RESEARCH_LABEL_LAG_MONTHS) -> bool:
    """Conservative *research convention*, NOT proof of ESRB actual release."""
    return add_months(m, research_lag) <= dev_end


def make_dev_onset_keys(panel: pd.DataFrame, episodes: list[dict], *,
                        dev_end: str = DEV_END, start: str = DEV_START,
                        research_lag: int = RESEARCH_LABEL_LAG_MONTHS) -> pd.DataFrame:
    """Audit first onset (within (issue, issue+h]) separately from four-class state at h.

    H0 falsifier: appending ANY post-DEV onset/label/row must not change keys/targets.
    Cannot prove official issue-time PIT; output is diagnostic research eligibility.
    """
    dev = panel.loc[panel.month.between(start, dev_end)].copy()
    label_lut = {(str(r.country), str(r.month)): str(r.label)
                 for r in dev.itertuples(index=False)}
    starts = {}
    for e in episodes:
        if e.get("systemic") != 1 or e.get("dev_eligibility") == "PRE_DEV_ONLY":
            continue
        month = str(e["start"])[:7]
        if start <= month <= dev_end:
            starts.setdefault(str(e["country"]), set()).add(month)
    rows = []
    for r in dev.itertuples(index=False):
        issue = str(r.month)
        country = str(r.country)
        if str(r.label) != "N" or str(r.label_status) != "SCORABLE":
            continue
        for name, h in HORIZON_MONTHS.items():
            end = target_end(issue, h)
            if end > dev_end or not label_mature_by_freeze(end, dev_end=dev_end,
                                                             research_lag=research_lag):
                continue
            future_label = label_lut.get((country, end))
            if future_label not in ("N", "D", "G", "BOTH"):
                continue
            # Every calendar month must have an observed ex-post benchmark label,
            # otherwise the absence of an onset is not evidence of a negative.
            months = pd.period_range(pd.Period(issue, freq="M") + 1,
                                     pd.Period(end, freq="M"), freq="M")
            if not all((country, str(t)) in label_lut for t in months):
                continue
            onset = int(any(issue < t <= end for t in starts.get(country, set())))
            rows.append({"country": country, "issue": issue, "horizon": name,
                         "horizon_months": h, "target_end": end,
                         "research_label_mature_at": add_months(end, research_lag),
                         "first_onset": onset, "future_state": future_label,
                         "future_stress": int(future_label != "N"),
                         "status": "RESEARCH_PROXY_RELEASE_UNVERIFIED"})
    return pd.DataFrame(rows, columns=["country", "issue", "horizon", "horizon_months",
                                      "target_end", "research_label_mature_at", "first_onset",
                                      "future_state", "future_stress", "status"])


def make_dev_exit_keys(panel: pd.DataFrame, episodes: list[dict], *,
                       dev_end: str = DEV_END, start: str = DEV_START,
                       research_lag: int = RESEARCH_LABEL_LAG_MONTHS) -> pd.DataFrame:
    """First N from a currently stressed eligible risk set (EXIT).

    `first_n_exact_month` is the PRIMARY target (spec RQ02 EXIT 1.3): the first N after the issue falls in EXACTLY
    the target month.  `first_n` (first N within (issue, issue+h]) is the BY-HORIZON control; the two are never
    mixed in one loss table.  A matured negative (no first N yet) is 0; never silently drop it or call it U.
    """
    dev = panel.loc[panel.month.between(start, dev_end)].copy()
    lut = {(str(r.country), str(r.month)): str(r.label)
           for r in dev.itertuples(index=False)}
    normal_by_country = {}
    for e in episodes:
        if e.get("systemic") != 1 or e.get("dev_eligibility") == "PRE_DEV_ONLY":
            continue
        date = str(e["normal"])[:7]
        if start <= date <= dev_end:
            normal_by_country.setdefault(str(e["country"]), set()).add(date)
    rows = []
    for r in dev.itertuples(index=False):
        issue, country = str(r.month), str(r.country)
        if str(r.label) not in ("D", "G", "BOTH") or str(r.label_status) != "SCORABLE":
            continue
        for h in (18, 12, 6, 3):
            end = target_end(issue, h)
            if end > dev_end or not label_mature_by_freeze(end, dev_end=dev_end,
                                                             research_lag=research_lag):
                continue
            months = pd.period_range(pd.Period(issue, freq="M") + 1,
                                     pd.Period(end, freq="M"), freq="M")
            if not all((country, str(t)) in lut for t in months):
                continue
            normals = sorted(t for t in normal_by_country.get(country, set()) if issue < t <= end)
            rows.append({"country": country, "issue": issue, "exit_horizon": f"H{h}",
                         "target_end": end, "first_n": int(bool(normals)),
                         "first_n_exact_month": int(bool(normals) and normals[0] == end),
                         "normal_month": normals[0] if normals else None,
                         "research_label_mature_at": add_months(end, research_lag),
                         "status": "RESEARCH_PROXY_RELEASE_UNVERIFIED"})
    return pd.DataFrame(rows, columns=["country", "issue", "exit_horizon", "target_end",
                                      "first_n", "first_n_exact_month", "normal_month",
                                      "research_label_mature_at", "status"])


def native_history(raw_native: pd.DataFrame, *, country: str, source: str,
                   issue_at: str, n: int) -> pd.DataFrame:
    """Exact native obs per period. Same quarter in three carried months counts once.

    Required raw_native columns: country, source, native_period, available_at, value.
    A missing release timestamp is a hard invalidity, not 'available = period end'.
    """
    required = {"country", "source", "native_period", "available_at", "value"}
    if not required.issubset(raw_native):
        raise ValueError("NATIVE_SOURCE_ID_AND_RELEASE_REQUIRED:" + ",".join(sorted(required - set(raw_native))))
    x = raw_native.loc[(raw_native.country == country) & (raw_native.source == source)].copy()
    if x["available_at"].isna().any():
        raise ValueError("UNKNOWN_SOURCE_RELEASE_AT")
    x["available_at"] = pd.to_datetime(x["available_at"])
    cutoff = pd.Timestamp(issue_at)
    x = x[x.available_at <= cutoff].sort_values(["available_at", "native_period"])
    x = x.drop_duplicates("native_period", keep="last").tail(n).copy()
    x["value"] = pd.to_numeric(x["value"], errors="coerce")
    return x.reset_index(drop=True)


@dataclass(frozen=True)
class NativeThreshold:
    owner: str
    theta_L: float
    theta_H: float
    k_count: int
    b_terminal: int = 1
    sign: int = 1

    def __post_init__(self):
        if not (0 <= self.theta_L <= self.theta_H and self.k_count >= 0 and
                self.b_terminal >= 1 and self.sign in (-1, 1)):
            raise ValueError("INVALID_NATIVE_THRESHOLD_GEOMETRY")


def individual_native(raw_native: pd.DataFrame, theta: NativeThreshold, *,
                      country: str, source: str, issue_at: str, n: int = 7) -> dict:
    x = native_history(raw_native, country=country, source=source,
                       issue_at=issue_at, n=n)
    if len(x) < 2 or not np.isfinite(x.value.to_numpy(float)).all():
        return {"execution_status": "NOT_EVALUATED", "reason": "FEWER_THAN_TWO_FINITE_NATIVE_LEVELS",
                "n_native": len(x), "n_comparisons": 0, "n_terminal": 0}
    z = theta.sign * x.value.to_numpy(float)
    diffs = np.array([z[j] - z[i] for i, j in combinations(range(len(z)), 2)])
    terminal = z[-1] - z[:-1]
    count_up = int((diffs > 0).sum())
    strong = int(((terminal > 0) & (terminal >= theta.theta_H)).sum()) >= theta.b_terminal
    weak = int(((terminal > 0) & (terminal >= theta.theta_L)).sum()) >= theta.b_terminal
    vote = bool(strong or (weak and count_up >= theta.k_count))
    return {"vote": "PASS" if vote else "FAIL", "execution_status": "EXECUTED", "n_native": len(z),
            "n_comparisons": len(diffs), "n_terminal": len(terminal),
            "count_up": count_up, "count_down": int((diffs < 0).sum()),
            "count_sideways": int((diffs == 0).sum()),
            "strong": bool(strong), "weak": bool(weak), "owner_gate": "NOT_VERIFIED",
            "native_period_last": str(x.native_period.iloc[-1]),
            "release_at": str(x.available_at.iloc[-1])}


def joint_native(raw_native: pd.DataFrame, a: str, b: str,
                 ta: NativeThreshold, tb: NativeThreshold, *, country: str,
                 issue_at: str, n: int = 7, k_joint: int = 1,
                 gate1_pass: bool | None = None, selected: bool = False) -> dict:
    """Same-base Gate2; effective Joint cannot PASS without approved Gate1/selection."""
    aa = native_history(raw_native, country=country, source=a, issue_at=issue_at, n=n)
    bb = native_history(raw_native, country=country, source=b, issue_at=issue_at, n=n)
    shared = sorted(set(aa.native_period) & set(bb.native_period))
    if len(shared) < 2:
        return {"execution_status": "NOT_EVALUATED", "n_aligned_native": len(shared),
                "reason": "NO_SAME_NATIVE_BASE"}
    xa = aa.set_index("native_period").loc[shared, "value"].to_numpy(float) * ta.sign
    xb = bb.set_index("native_period").loc[shared, "value"].to_numpy(float) * tb.sign
    if not (np.isfinite(xa).all() and np.isfinite(xb).all()):
        return {"execution_status": "NOT_EVALUATED", "n_aligned_native": len(shared),
                "reason": "MISSING_NATIVE_VALUE"}
    ma = xa[-1] - xa[:-1]
    mb = xb[-1] - xb[:-1]
    simultaneous_strong = ((ma > 0) & (mb > 0) & (ma >= ta.theta_H) & (mb >= tb.theta_H))
    simultaneous_weak = ((ma > 0) & (mb > 0) & (ma >= ta.theta_L) & (mb >= tb.theta_L))
    both_up = sum((xa[j] > xa[i]) and (xb[j] > xb[i])
                  for i, j in combinations(range(len(shared)), 2))
    strong = int(simultaneous_strong.sum()) >= max(ta.b_terminal, tb.b_terminal)
    weak = int(simultaneous_weak.sum()) >= max(ta.b_terminal, tb.b_terminal)
    gate2 = bool(strong or (weak and both_up >= k_joint))
    # effective Joint needs an approved Gate1 AND a selected pair; otherwise it is BLOCKED (no vote at all)
    if gate1_pass is None or not selected:
        effective = {"execution_status": "BLOCKED", "reason": "GATE1_APPROVAL_OR_PAIR_SELECTION_MISSING"}
    else:
        effective = {"vote": "PASS" if (gate1_pass and gate2) else "FAIL", "execution_status": "EXECUTED"}
    return {**effective, "candidate_gate2_vote": "PASS" if gate2 else "FAIL",
            "n_aligned_native": len(shared), "n_simultaneous_strong": int(simultaneous_strong.sum()),
            "n_simultaneous_weak": int(simultaneous_weak.sum()), "count_both_up": int(both_up),
            "n_comparisons": len(shared) * (len(shared) - 1) // 2,
            "native_period_last": str(shared[-1]),
            "gate1_status": "UNVERIFIED" if gate1_pass is None else ("PASS" if gate1_pass else "FAIL"),
            "selection_status": "SELECTED_WITH_EXTERNAL_AUTHORIZATION" if selected else "CANDIDATE_ONLY_NOT_APPROVED"}


def candidate_grid(L: float, U: float, step: float) -> list[float]:
    if not (np.isfinite([L, U, step]).all() and L <= U and step > 0):
        raise ValueError("INVALID_L_U_STEP")
    count = int(np.floor((U - L) / step + 1e-10)) + 1
    if count > 1_000_000:
        raise ValueError("PREDECLARED_BUDGET_REQUIRED")
    return [float(L + i * step) for i in range(count)]


def select_dev_binary_threshold(score, y, *, fnr_max=0.10, fpr_max=0.20,
                                min_positive=1, min_negative=1) -> dict:
    """Exact breakpoints for a FROZEN score: exhaustive candidates, dedup masks.

    Formula on matched DEV rows: FN/P, FP/N; zero is a true zero, not missing.
    Feasible: strict FNR<.10 AND FPR<.20; among feasible min FN, then FP.
    If no feasibility, retain nearest normalized violation (NOT a passing model).
    This is only a decision cutoff search, not native theta-vector/Pools search.
    """
    p = np.asarray(score, float)
    yy = np.asarray(y, int)
    if p.ndim != 1 or yy.shape != p.shape or len(p) == 0:
        raise ValueError("INVALID_MATCHED_SCORE_AND_TARGET")
    if not (np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
            and np.isin(yy, [0, 1]).all()):
        raise ValueError("PROBABILITY_OR_BINARY_TARGET_INVALID")
    n_pos = int((yy == 1).sum())
    n_neg = int((yy == 0).sum())
    if n_pos < min_positive or n_neg < min_negative:
        return {"status": "UNDEFINED_ZERO_OR_INSUFFICIENT_SUPPORT", "n_pos": n_pos,
                "n_neg": n_neg, "n_nominal": 0, "n_executed": 0,
                "n_unique_decision_masks": 0}
    cuts = [np.inf] + sorted(np.unique(p).tolist(), reverse=True)
    seen, rows = set(), []
    for cut in cuts:
        yh = (p >= cut).astype(np.int8)
        mask_hash = yh.tobytes()
        duplicate = mask_hash in seen
        seen.add(mask_hash)
        TP = int(((yh == 1) & (yy == 1)).sum())
        FN = n_pos - TP
        FP = int(((yh == 1) & (yy == 0)).sum())
        TN = n_neg - FP
        fnr = FN / n_pos
        fpr = FP / n_neg
        v = max(0, fnr / fnr_max - 1) + max(0, fpr / fpr_max - 1)
        rows.append({"cut": cut, "TP": TP, "FN": FN, "FP": FP, "TN": TN,
                     "FNR": fnr, "FPR": fpr, "violation": v,
                     "feasible": fnr < fnr_max and fpr < fpr_max,
                     "duplicate_mask": duplicate})
    feasible = [r for r in rows if r["feasible"]]
    best = (min(feasible, key=lambda r: (r["FN"], r["FP"], -r["cut"]))
            if feasible else min(rows, key=lambda r: (r["violation"], r["FN"], r["FP"])))
    idx = rows.index(best)
    neighbors = {"L": None, "U": None, "STEP": None}
    if np.isfinite(best["cut"]):
        vals = np.array(sorted(np.unique(p)), float)
        k = int(np.searchsorted(vals, best["cut"]))
        neighbors["L"] = float(vals[k - 1]) if k else None
        neighbors["U"] = float(vals[k + 1]) if k + 1 < len(vals) else None
        if neighbors["L"] is not None and neighbors["U"] is not None:
            neighbors["STEP"] = float((neighbors["U"] - neighbors["L"]) / 4)
    brier = float(np.mean((p - yy) ** 2))
    clipped = np.clip(p, 1e-12, 1 - 1e-12)
    ll = float(np.mean(-yy * np.log(clipped) - (1 - yy) * np.log(1 - clipped)))
    return {"status": "DEV_FEASIBLE_RESEARCH_ONLY" if feasible else "NO_FEASIBLE_CANDIDATE_TESTED",
            "threshold": float(best["cut"]), "gate_feasible": bool(feasible),
            "n_pos": n_pos, "n_neg": n_neg,
            "n_nominal": len(cuts), "n_executed": len(rows),
            "n_unique_decision_masks": len(seen), "n_duplicate_masks": len(rows) - len(seen),
            "objective": "min_FN_then_FP_if_feasible_else_min_normalized_violation",
            "loss_brier_in_sample": brier, "loss_logloss_in_sample": ll,
            **{k: best[k] for k in ("TP", "FN", "FP", "TN", "FNR", "FPR", "violation")},
            **neighbors}
