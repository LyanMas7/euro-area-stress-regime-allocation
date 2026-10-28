"""RQ03 end-to-end two-W0 runner: RQ03-01..RQ03-11 on DEV 2005-2019 (W0_DEV) and OOS 2020-2024 (W0_OOS).

    python -m rev363.rq03_e2e --dataset-root <locked dataset> --rq01-dir <RQ01_NEXT_FINAL> --out <dir> \
        --oos-authorization USER_AUTHORIZED_OOS_20261025_DEC015

Two independent inceptions (master spec J1): each run starts with W0 = 1 held in C1, its own run_id, fresh
benchmark base states, a fresh Main lock state and fresh holdings; nothing (wealth, holdings, base snapshots,
or lock state) is carried from DEV into OOS.  The
OOS risk filter and trailing scenario windows may read DEV history as-of (warm-up), never a later OOS week.
The benchmark trigger is not selected: it is fixed at >= 2 of six capital-weight deviations >= 5pp.  Main
research candidates are selected on DEV only; OOS is reported without tuning.  OOS has been exposed before
(RQ01 OOS issued tape exists), so no OOS result here is an independent sealed-OOS certification.
"""
from __future__ import annotations

import os

# The RQ01 LOGISTIC M11 refit reproduces the issued RQ01 tape exactly only with single-threaded BLAS
# (AI_HANDOFF RQ01_FINAL session log); effective only when this module is imported before numpy.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import rq03_e2e_data as D
from . import rq03_e2e_engine as E
from . import rq03_e2e_risk as RK
from .util import run_id as make_run_id, utcnow

VERSION = "RQ03_E2E_WEIGHT_BAND_V2"
FEES_BPS = (0.0, 5.0, 10.0, 25.0)          # threshold_research.bounded_frontier registered cost grid
NATIVE_DIAGNOSTIC_QUANTILES = (0.0, 0.5, 0.9, 1.0)  # diagnostics only; never benchmark activation
VOTE_K = (2, 3)                             # spec 11.1 (>=2 of 3) and rq03 registered strict gate (3 of 3)
PRIMARY_FEE = 10.0
SHAPE_ESTIMATOR = {"DCC": "DCC_GARCH_ASOF", "EWMA": "EWMA"}


def _hist_windows(R: np.ndarray, window: int = RK.HS_WINDOW) -> tuple[np.ndarray, np.ndarray]:
    n = len(R)
    out = np.full((n, window, 6), np.nan)
    cnt = np.zeros(n, int)
    for t in range(n):
        lo = max(0, t - window)
        blk = R[lo:t]
        if len(blk):
            out[t, -len(blk):] = blk
        cnt[t] = len(blk)
    return out, cnt


def build_inputs(dataset_root, rq01_dir, *, oos_authorization=None, include_oos: bool = True) -> dict:
    t0 = time.time()
    dev_tape, dev_ev = D.period_tape(dataset_root, "DEV")
    tapes = [dev_tape]
    oos_ev = None
    if include_oos:
        oos_tape, oos_ev = D.period_tape(dataset_root, "OOS", oos_authorization=oos_authorization)
        tapes.append(oos_tape)
    full = pd.concat(tapes, ignore_index=True)
    if (full["week_end"].diff().dt.days.dropna() != 7).any():
        raise ValueError("DEV_OOS_CALENDAR_NOT_CONTIGUOUS")
    risky = full[list(D.RISKY)].to_numpy(float)
    fit = RK.fit_dev_garch_dcc(dev_tape[list(D.RISKY)].to_numpy(float))
    dcc = RK.filter_dcc(risky, fit)
    ewma = RK.filter_ewma(risky)
    # prefix invariance: the DEV slice of the continuous filter equals a DEV-only filter (no OOS information)
    dcc_dev_only = RK.filter_dcc(dev_tape[list(D.RISKY)].to_numpy(float), fit)
    n_dev = len(dev_tape)
    prefix_diff = float(np.nanmax(np.abs(dcc["cov"][:n_dev] - dcc_dev_only["cov"])))
    rq01_dir = Path(rq01_dir)
    phase_files = [rq01_dir / "RQ01_DEV_CANONICAL_PHASE.csv"]
    if include_oos:
        phase_files.append(rq01_dir / "RQ01_OOS_CANONICAL_PHASE.csv")
    rq01, rq01_meta = D.rq01_issued_p4(
        dataset_root, issued_oos_tape=(rq01_dir / "RQ01_10_ISSUED_TAPE_OOS.csv") if include_oos else None,
        canonical_phase_files=phase_files, oos_authorization=oos_authorization, include_oos=include_oos)
    monthly = D.ea_monthly_signals(rq01)
    weekly = D.weekly_signals(full, monthly)
    flags = D.asset_bottom_flags(full)
    R6 = full[list(D.ASSETS)].to_numpy(float)
    hist, hist_n = _hist_windows(R6)
    covs = {"DCC": dcc["cov"], "EWMA": ewma["cov"]}
    ready = {"DCC": dcc["ready"], "EWMA": ewma["ready"]}
    shapes = {}
    for cov in ("DCC", "EWMA"):
        for m in E.BENCH:
            shapes[(cov, m)] = E.benchmark_shapes(covs[cov], ready[cov], m, SHAPE_ESTIMATOR[cov])

    def sl(lo, hi, name):
        return E.PeriodInputs(
            period=name, week_start=list(full["week_start"].iloc[lo:hi]), week_end=list(full["week_end"].iloc[lo:hi]),
            R=R6[lo:hi], cov={k: v[lo:hi] for k, v in covs.items()}, ready={k: v[lo:hi] for k, v in ready.items()},
            hist=hist[lo:hi], hist_n=hist_n[lo:hi], regime=weekly["regime"].to_numpy()[lo:hi],
            votes=weekly["votes"].to_numpy(float)[lo:hi], vote_known=weekly["vote_known"].to_numpy(bool)[lo:hi],
            asset_flags=flags[lo:hi], signal_month=list(weekly["signal_month"].iloc[lo:hi]),
            published_at=list(weekly["published_at"].iloc[lo:hi]),
            shapes={k: v[lo:hi] for k, v in shapes.items()})
    n = len(full)
    # t0 of each inception is the first week whose DECISION date (week_start) lies inside the block (master spec
    # J1: t0_OOS >= 2020-01-01).  The straddling week 2019-12-30..2020-01-03 belongs to neither run's trading
    # calendar; it stays in the continuous history and is used only as-of (filter, scenarios, signals).
    dev_lo = int(np.argmax(full["week_start"] >= pd.Timestamp("2005-01-01")))
    offsets = {"DEV": (dev_lo, n_dev)}
    result = {"DEV": sl(*offsets["DEV"], "DEV"), "offsets": offsets,
              "untraded_straddle_weeks": [], "full": full, "weekly": weekly, "monthly": monthly, "rq01": rq01,
              "rq01_meta": rq01_meta, "fit": fit, "dcc": dcc, "ewma": ewma,
              "evidence": {"DEV_tape": dev_ev, "dcc_prefix_invariance_max_abs": prefix_diff,
                           "dcc_min_eigenvalue": dcc["min_eig"], "build_seconds": round(time.time() - t0, 1),
                           "oos_read": bool(include_oos)},
              "shapes_full": shapes, "flags": flags, "n_dev": n_dev, "include_oos": bool(include_oos)}
    if include_oos:
        oos_lo = int(np.argmax(full["week_start"] >= pd.Timestamp("2020-01-01")))
        if oos_lo < n_dev:
            raise ValueError("OOS_T0_BEFORE_DEV_END")
        offsets["OOS"] = (oos_lo, n)
        result["OOS"] = sl(*offsets["OOS"], "OOS")
        result["untraded_straddle_weeks"] = [str(x.date()) for x in full["week_start"].iloc[n_dev:oos_lo]]
        result["evidence"]["OOS_tape"] = oos_ev
    return result


def ledger_frame(st: dict) -> pd.DataFrame:
    n = st["_n"]
    cols = {k: v for k, v in st.items() if isinstance(v, list) and len(v) == n}
    df = pd.DataFrame(cols)
    df.insert(0, "strategy", st["strategy"])
    df.insert(0, "period", st["period"])
    df.insert(0, "run_id", st["run_id"])
    return df


def ledger_hash(df: pd.DataFrame, decimals: int = 10) -> str:
    """Decision/NAV fingerprint rounded to `decimals` (last-ulp float noise and CSV round trips must not change
    it); exact equality is tested separately where byte identity is required (T13)."""
    keep = [c for c in df.columns if c.startswith("w_") or c in
            ("action", "cost", "turnover", "nav_end", "weight_gate_status", "restructure_request",
             "qualifying_count", "base_version_after", "rebase_applied")]
    return hashlib.sha256(df[keep].round(decimals).to_csv(index=False).encode()).hexdigest()


def ledgers_exactly_equal(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    keep = [c for c in a.columns if c.startswith("w_") or c in
            ("action", "cost", "turnover", "nav_end", "nav_pre", "weight_gate_status",
             "restructure_request", "qualifying_count", "base_version_after", "rebase_applied")]
    return bool(a[keep].reset_index(drop=True).equals(b[keep].reset_index(drop=True)))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--rq01-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--oos-authorization", required=True)
    p.add_argument("--mode", choices=("full-sequential", "dev-freeze", "final-oos"), default="full-sequential",
                   help="DEV-only grid/freeze, frozen final OOS, or both in the required order")
    a = p.parse_args(argv)
    from .rq03_e2e_report import run_all, stage_dev_freeze, stage_grid, stage_inputs
    out = Path(a.out)
    if a.mode in ("full-sequential", "dev-freeze"):
        dev = stage_inputs(a.dataset_root, a.rq01_dir, out, include_oos=False)
        grids = {(fam, "DEV"): stage_grid(dev, out, "DEV", fam) for fam in ("BENCHMARKS", "MAIN")}
        manifest = stage_dev_freeze(dev, out, grids)
        if a.mode == "dev-freeze":
            print(json.dumps({"freeze": manifest}, indent=2, default=str))
            return
    res = run_all(a.dataset_root, a.rq01_dir, out, oos_authorization=a.oos_authorization,
                  stages=("inputs", "final", "evidence"))
    print(json.dumps(res["summary"], indent=2, default=str))


if __name__ == "__main__":
    main()
