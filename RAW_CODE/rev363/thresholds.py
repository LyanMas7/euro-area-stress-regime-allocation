"""REV363 exact finite threshold families (L/U/step, unique masks, Pareto).

The search is always on a frozen score.  Every nominal threshold is executed;
duplicate decision masks are counted, not treated as extra evidence.  A
challenger is not accepted because it wins apparent DEV; it is frozen only
after the DEV gate and before OOS is opened.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import EA7
from .metrics import confusion_binary


@dataclass
class ThresholdResult:
    threshold: float
    TP: int
    FN: int
    FP: int
    TN: int
    FNR: float
    FPR: float
    gate_feasible: bool
    n_nominal: int
    n_unique_masks: int
    n_executed: int
    L: float | None
    U: float | None
    objective: str


def _counts(score, y, cut):
    pred = (score >= cut).astype(int)
    c = confusion_binary(y, pred)
    return c, pred


def exact_threshold_search(score, y, *, fnr_max=0.05, fpr_max=0.15,
                           objective="min_fp"):
    score = np.asarray(score, float)
    y = np.asarray(y, int)
    uniq = np.unique(score)
    cuts = np.r_[np.inf, uniq[::-1]]  # include predict-none
    seen = set()
    rows = []
    for cut in cuts:
        c, pred = _counts(score, y, cut)
        key = pred.tobytes()
        unique = key not in seen
        seen.add(key)
        rows.append({"cut": float(cut), **c, "unique_mask": unique,
                     "gate_feasible": (c["FNR"] is not None and c["FNR"] < fnr_max and
                                       c["FPR"] is not None and c["FPR"] < fpr_max)})
    feasible = [r for r in rows if r["gate_feasible"]]
    if feasible:
        if objective == "min_fp":
            best = sorted(feasible, key=lambda r: (r["FP"], -r["TP"], -r["cut"]))[0]
        elif objective == "max_tp":
            best = sorted(feasible, key=lambda r: (-r["TP"], r["FP"], -r["cut"]))[0]
        else:
            raise ValueError("UNKNOWN_THRESHOLD_OBJECTIVE")
    else:
        # closest normalised violation: FNR first, then FPR
        def violation(r):
            return (max(0.0, (r["FNR"] or 1.0) - fnr_max) / fnr_max +
                    max(0.0, (r["FPR"] or 1.0) - fpr_max) / fpr_max)
        best = sorted(rows, key=lambda r: (violation(r), r["FP"], -r["TP"]))[0]
    idx = rows.index(best)
    lower = None
    for k in range(idx + 1, len(rows)):
        if rows[k]["cut"] < best["cut"] and np.isfinite(rows[k]["cut"]):
            lower = rows[k]["cut"]
            break
    upper = best["cut"] if np.isfinite(best["cut"]) else None
    return ThresholdResult(
        threshold=float(best["cut"]), TP=best["TP"], FN=best["FN"], FP=best["FP"],
        TN=best["TN"], FNR=best["FNR"], FPR=best["FPR"],
        gate_feasible=bool(best["gate_feasible"]), n_nominal=len(cuts),
        n_unique_masks=len(seen), n_executed=len(cuts), L=lower, U=upper,
        objective=objective)


def country_dp(score, y, countries, *, fn_limit=None):
    """Exact per-country scalar-threshold policy under a total FN budget.

    This is the exact Bellman DP used by the REV362 audit, reimplemented for
    the REV363 score.  It is a policy family with seven free cutoffs and must
    carry a complexity caveat.
    """
    score = np.asarray(score, float)
    y = np.asarray(y, int)
    countries = np.asarray(countries)
    if fn_limit is None:
        fn_limit = int(np.floor(0.05 * (y == 1).sum()))
    # per-country non-dominated (fn, fp, cut) options
    opts = {}
    for c in EA7:
        m = countries == c
        s = score[m]
        yy = y[m]
        uniq = np.unique(s)
        cand = []
        for cut in np.r_[np.inf, uniq[::-1]]:
            cc, _ = _counts(s, yy, cut)
            cand.append((cc["FN"], cc["FP"], float(cut)))
        cand.sort()
        pr = []
        best_fp = 10 ** 18
        for fn, fp, cut in cand:
            if fp < best_fp:
                pr.append((fn, fp, cut))
                best_fp = fp
        opts[c] = pr
    dp = [dict() for _ in range(len(EA7) + 1)]
    dp[0][0] = (0.0, [])
    for i, c in enumerate(EA7):
        nxt = {}
        for prev_fn, (prev_fp, path) in dp[i].items():
            for fn, fp, cut in opts[c]:
                t = prev_fn + fn
                if t > fn_limit:
                    continue
                val = prev_fp + fp
                if t not in nxt or val < nxt[t][0]:
                    nxt[t] = (val, path + [(c, cut, fn, fp)])
        dp[i + 1] = nxt
    if not dp[len(EA7)]:
        return {"status": "INFEASIBLE_UNDER_FN_BUDGET", "fn_limit": fn_limit}
    best_fn = min(dp[len(EA7)], key=lambda k: (dp[len(EA7)][k][0], k))
    best_fp, path = dp[len(EA7)][best_fn]
    return {"status": "EXECUTED_APPARENT_DEV", "fn_limit": fn_limit,
            "FN": int(best_fn), "FP": int(best_fp),
            "FNR": float(best_fn / (y == 1).sum()),
            "FPR": float(best_fp / (y == 0).sum()),
            "gate_feasible": bool(best_fn < 0.05 * (y == 1).sum() and
                                  best_fp < 0.15 * (y == 0).sum()),
            "thresholds": {c: (None if not np.isfinite(cut) else cut)
                           for c, cut, fn, fp in path},
            "path": [{"country": c, "threshold": (None if not np.isfinite(cut) else cut),
                      "FN": fn, "FP": fp} for c, cut, fn, fp in path],
            "caveat": "SEVEN_FREE_COUNTRY_CUTOFFS_ADAPTIVE_OVERFIT_RISK"}


def two_dim_search(score, second, y, *, fnr_max=0.05, fpr_max=0.15,
                   max_grid=160):
    """Global 2-D mask score>=a AND second>=b (or <=b), unique-grid dedup.

    The second-feature grid is the union of quantiles and unique values,
    capped deterministically to keep the executed family auditable.  This is a
    bounded research family, not an exhaustion proof.
    """
    score = np.asarray(score, float)
    second = np.asarray(second, float)
    y = np.asarray(y, int)
    qs = np.quantile(second[np.isfinite(second)], np.linspace(.02, .98, 49))
    vgrid = np.unique(np.r_[qs, np.unique(second[np.isfinite(second)])])
    if len(vgrid) > max_grid:
        vgrid = np.unique(np.quantile(vgrid, np.linspace(0, 1, max_grid)))
    su = np.unique(score)
    rows = []
    for a in np.r_[np.inf, su[::-1]]:
        base = score >= a
        for b in vgrid:
            for op in ("ge", "le"):
                m2 = second >= b if op == "ge" else second <= b
                pred = (base & m2).astype(int)
                c = confusion_binary(y, pred)
                rows.append({"score_cut": float(a), "second_cut": float(b),
                             "second_op": op, **c})
    feasible = [r for r in rows if r["FNR"] is not None and r["FNR"] < fnr_max
                and r["FPR"] is not None and r["FPR"] < fpr_max]
    if feasible:
        best = sorted(feasible, key=lambda r: (r["FP"], -r["TP"]))[0]
    else:
        best = sorted(rows, key=lambda r: (max(0, (r["FNR"] or 1) - fnr_max) / fnr_max +
                                           max(0, (r["FPR"] or 1) - fpr_max) / fpr_max,
                                           r["FP"]))[0]
    return {"status": "EXECUTED_BOUNDED_2D", "n_nominal": len(rows),
            "n_unique_masks": len({(r["score_cut"], r["second_cut"], r["second_op"]) for r in rows}),
            "n_executed": len(rows), "gate_feasible": bool(feasible),
            "best": best, "grid_second": len(vgrid),
            "caveat": "BOUNDED_RESEARCH_GRID_NOT_EXHAUSTIVE_PROOF"}
