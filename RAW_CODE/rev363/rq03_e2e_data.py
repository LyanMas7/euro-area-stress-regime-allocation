"""RQ03-01 six-asset weekly tape and the read-only issued RQ01 input adapter for the two-W0 RQ03 run.

Six assets (spec RQ03 risk tolerance section 1; dataset 09_V8_MAIN_SIX_ASSET_SINGLE_SOURCE_CONSUMER.md):
    E1/E2/E3  EURO STOXX Small/Mid/Large NTR weekly: r = P_t / P_prev - 1.  The Sunday `Date` label is the
              OPENING of the following Mon-Fri trading week (week_end = Date + 5 days; identified empirically:
              corr(r, dlog VIX) = -0.63 at +5 days, ~0 at -2 days).
    B1/B2     synthetic constant-maturity 2Y/10Y bond returns from the ECB daily spot curve:
              r_d = y_prev/252 - T/(1+y_prev) dy + T(T+1)/(2(1+y_prev)^2) dy^2, compounded Mon-Fri.
    C1        Euribor-3M accrual proxy: (rate/100) * 7/360 from the latest fixing released before Monday.
DEV 2005-2019 is the LOCKED tape (SHA verified).  OOS 2020-2024 is rebuilt from the raw sources with the
SAME builders (rq03.build_equity_weekly / build_bond_weekly / build_cash_weekly); the builders are first
proven to reproduce the locked DEV tape, otherwise the OOS tape is refused.  B1/B2/C1 remain research proxies
(bond investability and cash funding are BLOCKED owner gates) - nothing here certifies investability.

Clock: a decision for week t is taken at week_start(t) 00:00 (Monday), strictly after week_end(t-1) and the
release of every input used; the return of week t is realised over [week_start(t), week_end(t)].
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from . import rq03 as R3
from .config import EA7
from .data import _exact, find_dataset_root, month_end

ASSETS = ("E1", "E2", "E3", "B1", "B2", "C1")
RISKY = ASSETS[:5]
LOCKED_DEV_TAPE = "V8_RQ03_SIX_ASSET_DEV_WEEKLY.csv"
LOCKED_DEV_SHA = "7924b2ec0b411aa0fee8590dafd9891448b9f2534f81a994ae04228c342cc33e"
PERIODS = {"DEV": ("2005-01-01", "2019-12-31"), "OOS": ("2020-01-01", "2024-12-31")}
REPRO_TOL = {"E1": 1e-9, "E2": 1e-9, "E3": 1e-9, "B1": 1e-12, "B2": 1e-12, "C1": 1e-12}

ASSET_META = {
    "E1": ("EURO_STOXX_Small_NTR_Weekly.csv", "NTR_WEEKLY_INDEX", "P_t/P_prev-1", "OBSERVED_INDEX_NOT_TRADED_FUND"),
    "E2": ("EURO_STOXX_Mid_NTR_Weekly.csv", "NTR_WEEKLY_INDEX", "P_t/P_prev-1", "OBSERVED_INDEX_NOT_TRADED_FUND"),
    "E3": ("EURO_STOXX_Large_NTR_Weekly.csv", "NTR_WEEKLY_INDEX", "P_t/P_prev-1", "OBSERVED_INDEX_NOT_TRADED_FUND"),
    "B1": ("ECB_Euro_Area_2Y_Spot_Rate_Daily.csv", "ECB_SPOT_YIELD_SYNTHETIC_DAILY",
           "y_prev/252-T/(1+y_prev)dy+T(T+1)/(2(1+y_prev)^2)dy^2,T=2,compound Mon-Fri",
           "SYNTHETIC_PROXY_BOND_INVESTABILITY_BLOCKED"),
    "B2": ("ECB_Euro_Area_10Y_Spot_Rate_Daily.csv", "ECB_SPOT_YIELD_SYNTHETIC_DAILY",
           "y_prev/252-T/(1+y_prev)dy+T(T+1)/(2(1+y_prev)^2)dy^2,T=10,compound Mon-Fri",
           "SYNTHETIC_PROXY_BOND_INVESTABILITY_BLOCKED"),
    "C1": ("Euribor_3M_Weekly.xlsx", "BBK_DAILY_EURIBOR_FIXING", "rate_pct/100*7/360 (latest fixing before Monday)",
           "ACCRUAL_PROXY_CASH_FUNDING_BLOCKED"),
}


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_raw_tape(root) -> pd.DataFrame:
    """All common weeks 2005..2024 rebuilt from the raw sources with the locked formulas."""
    e1 = R3.build_equity_weekly(root, "Small").rename(columns={"Small": "E1"})
    e2 = R3.build_equity_weekly(root, "Mid").rename(columns={"Mid": "E2"})
    e3 = R3.build_equity_weekly(root, "Large").rename(columns={"Large": "E3"})
    b1 = R3.build_bond_weekly(root, "2Y", 2.0).rename(columns={"B_2Y": "B1"})
    b2 = R3.build_bond_weekly(root, "10Y", 10.0).rename(columns={"B_10Y": "B2"})
    c1 = R3.build_cash_weekly(root)
    out = e1.merge(e2, on="week_end").merge(e3, on="week_end").merge(b1, on="week_end") \
        .merge(b2, on="week_end").merge(c1, on="week_end")
    out["week_start"] = out["week_end"] - pd.Timedelta(days=4)
    out = out[["week_start", "week_end", *ASSETS]].sort_values("week_end").reset_index(drop=True)
    if (out["week_end"].dt.weekday != 4).any() or out["week_end"].duplicated().any():
        raise ValueError("RAW_TAPE_WEEK_CALENDAR_INVALID")
    return out


def load_locked_dev(root) -> tuple[pd.DataFrame, dict]:
    p = _exact(Path(root), LOCKED_DEV_TAPE)
    sha = _sha(p)
    if sha != LOCKED_DEV_SHA:
        raise ValueError("LOCKED_DEV_TAPE_SHA_MISMATCH:" + sha)
    t = pd.read_csv(p, parse_dates=["week_start", "week_end"]).sort_values("week_end").reset_index(drop=True)
    return t, {"path": str(p), "sha256": sha}


def reproduction_check(raw: pd.DataFrame, dev: pd.DataFrame) -> dict:
    """The OOS builder is admissible only if the same builder reproduces every locked DEV cell."""
    m = dev.merge(raw, on="week_end", how="left", suffixes=("", "_raw"))
    out = {"n_dev_weeks": int(len(dev)), "n_matched": int(m["E1_raw"].notna().sum())}
    ok = out["n_matched"] == len(dev)
    for a in ASSETS:
        d = (m[a] - m[f"{a}_raw"]).abs()
        out[a] = {"max_abs_diff": float(d.max()), "tolerance": REPRO_TOL[a]}
        ok &= bool(d.max() <= REPRO_TOL[a])
    out["week_start_match"] = bool((m["week_start"] == m["week_start_raw"]).all())
    out["status"] = "PASS" if ok and out["week_start_match"] else "FAIL"
    return out


def period_tape(dataset_root, period: str, *, oos_authorization=None) -> tuple[pd.DataFrame, dict]:
    """DEV -> locked tape; OOS -> raw rebuild with the reproduced formulas (explicit OOS authorization)."""
    root = find_dataset_root(dataset_root)
    dev, dev_ev = load_locked_dev(root)
    raw = build_raw_tape(root)
    repro = reproduction_check(raw, dev)
    if repro["status"] != "PASS":
        raise ValueError("SIX_ASSET_BUILDER_DOES_NOT_REPRODUCE_LOCKED_DEV")
    lo, hi = (pd.Timestamp(x) for x in PERIODS[period])
    if period == "DEV":
        tape = dev.copy()
        source = "LOCKED_DEV_TAPE"
    elif period == "OOS":
        from .w01_contract import enforce_execution
        gate = enforce_execution(oos_requested=True, oos_authorization=oos_authorization)
        tape = raw[(raw["week_end"] >= lo) & (raw["week_end"] <= hi)].reset_index(drop=True)
        source = "RAW_REBUILD_WITH_LOCKED_FORMULAS:" + gate
    else:
        raise ValueError("PERIOD_MUST_BE_DEV_OR_OOS")
    if not ((tape["week_end"] >= lo) & (tape["week_end"] <= hi)).all():
        raise ValueError("PERIOD_TAPE_OUTSIDE_BLOCK")
    gaps = tape["week_end"].diff().dt.days.dropna()
    if (gaps != 7).any() or not np.isfinite(tape[list(ASSETS)].to_numpy(float)).all():
        raise ValueError("PERIOD_TAPE_GAP_OR_NONFINITE")
    if (tape[list(ASSETS)].to_numpy(float) <= -1).any():
        raise ValueError("RETURN_AT_OR_BELOW_MINUS_ONE")
    ev = {"period": period, "source": source, "n_weeks": int(len(tape)),
          "first_week_end": str(tape["week_end"].iloc[0].date()), "last_week_end": str(tape["week_end"].iloc[-1].date()),
          "locked_dev": dev_ev, "builder_reproduces_locked_dev": repro,
          "tape_sha256": hashlib.sha256(tape.to_csv(index=False).encode()).hexdigest()}
    return tape, ev


def asset_ledger(tape: pd.DataFrame, root) -> pd.DataFrame:
    """RQ03-01 weekly ledger: returns, index LEVELS (base 1 before the first period week), clocks."""
    out = tape[["week_start", "week_end"]].copy()
    for a in ASSETS:
        out[f"r_{a}"] = tape[a].astype(float)
        out[f"level_{a}"] = (1.0 + tape[a].astype(float)).cumprod()
    out["decision_at"] = out["week_start"].dt.strftime("%Y-%m-%dT00:00:00Z")
    out["return_available_at"] = out["week_end"].dt.strftime("%Y-%m-%dT23:59:59Z")
    return out


def source_register(root) -> pd.DataFrame:
    rows = []
    for a, (name, cls, formula, inv) in ASSET_META.items():
        p = _exact(Path(root), name)
        rows.append({"asset": a, "source_file": str(p.relative_to(root)), "sha256": _sha(p), "source_class": cls,
                     "native_frequency": "WEEKLY" if a[0] in "EC" else "DAILY_COMPOUNDED_TO_WEEKLY",
                     "return_formula": formula, "unit": "DECIMAL_SIMPLE_WEEKLY_RETURN_EUR",
                     "availability": ("WEEK_END_CLOSE" if a != "C1" else "FIXING_RELEASED_BEFORE_WEEK_START"),
                     "investability_status": inv})
    return pd.DataFrame(rows)


def equity_alignment_check(root, shifts=(-9, -2, 5, 12)) -> list:
    """Empirical week-label check: corr(weekly NTR return, weekly dlog VIX close) for candidate Date shifts.
    The correct alignment is the strongly negative one (+5 days: Sunday label = following Mon-Fri week)."""
    vix = pd.read_csv(_exact(Path(root), "VIX_Daily.csv"))
    vix.columns = ["date", "vix"]
    vix["date"] = pd.to_datetime(vix["date"])
    vix["vix"] = pd.to_numeric(vix["vix"], errors="coerce")
    vix = vix.dropna().sort_values("date")
    wk = vix.groupby(vix["date"] - pd.to_timedelta(vix["date"].dt.weekday, unit="D") + pd.Timedelta(days=4))["vix"].last()
    dv = np.log(wk).diff().rename("dlogvix")
    rows = []
    for a, name in (("E1", "Small"), ("E2", "Mid"), ("E3", "Large")):
        e = pd.read_csv(_exact(Path(root), f"EURO_STOXX_{name}_NTR_Weekly.csv"), parse_dates=["Date"])
        for s in shifts:
            j = pd.concat([e.assign(we=e["Date"] + pd.Timedelta(days=s)).set_index("we")["Weekly_Return"], dv],
                          axis=1, join="inner").dropna()
            rows.append({"asset": a, "date_shift_days": s, "n": int(len(j)),
                         "corr_with_dlog_vix": float(j.corr().iloc[0, 1]), "selected": s == 5})
    return rows


# --------------------------------------------------------------------------- issued RQ01 inputs (read-only)
PHASE_FOR_REGIME_FRAME = {"PRE_ONSET": "PRE", "NONE": "N", "DE_ESCALATION": "DEESC", "PEAK": "PEAK",
                          "ENTRY": "ENTRY", "RECOVERY": "RECOVERY", "EXIT": "EXIT"}


def rq01_issued_p4(dataset_root, *, issued_oos_tape=None, canonical_phase_files=(), oos_authorization=None,
                   include_oos: bool = True) -> tuple:
    """Monthly EA7 P4 from the RQ01 DEV-frozen LOGISTIC M11 (the model that issued RQ01_10_ISSUED_TAPE_OOS).

    RQ01 code is CALLED, never modified: panel.build_panel -> corrected M11 block -> estimators.fit_two_family
    on DEV 2005-07..2019-12 only.  DEV probabilities are IN-SAMPLE (fitted on DEV labels); OOS probabilities are
    frozen-DEV forecasts and must reproduce the issued OOS tape, otherwise this adapter refuses to continue.
    """
    from . import estimators as EST
    from . import panel as PNL
    if include_oos:
        from .w01_contract import enforce_execution
        enforce_execution(oos_requested=True, oos_authorization=oos_authorization)
    end = "2024-12" if include_oos else "2019-12"
    prep = PNL.build_panel(dataset_root, row_start="2004-01", row_end=end, data_end=end)
    panel, spec = prep["panel"], prep["spec"]
    cols = PNL.corrected_blocks(spec)["M11"]
    dev = (panel["month"] >= "2005-07") & (panel["month"] <= "2019-12")
    sub = panel[(panel["month"] >= "2005-07") & (panel["month"] <= end)].reset_index(drop=True)
    mdl = EST.fit_two_family(PNL.matrix_for(panel[dev], cols), cols, PNL.y4_from(panel[dev]),
                             family="LOGISTIC", name="M11")
    p4 = mdl.p4(PNL.matrix_for(sub, cols))
    out = sub[["country", "month", "label", "label_status", "phase", "dd_CLIFS", "z_CISS"]].copy()
    for k, c in enumerate(("N", "D", "G", "BOTH")):
        out[f"p_rq01_{c}"] = p4[:, k]
    out["sample"] = np.where(out["month"] <= "2019-12", "DEV_IN_SAMPLE", "OOS_FROZEN_DEV")
    if include_oos:
        issued = pd.read_csv(issued_oos_tape)
        chk = out.merge(issued[["country", "issue"] + [f"p_LOGISTIC_M11_{c}" for c in ("N", "D", "G", "BOTH")]],
                        left_on=["country", "month"], right_on=["country", "issue"], how="inner")
        diff = max(float((chk[f"p_rq01_{c}"] - chk[f"p_LOGISTIC_M11_{c}"]).abs().max())
                   for c in ("N", "D", "G", "BOTH"))
        repro = {"n_oos_rows_compared": int(len(chk)), "max_abs_p4_diff_vs_issued_oos_tape": diff,
                 "issued_oos_tape_sha256": _sha(issued_oos_tape),
                 "status": "PASS" if len(chk) == 420 and diff <= 1e-8 else "FAIL"}
        if repro["status"] != "PASS":
            raise ValueError("RQ01_FROZEN_M11_DOES_NOT_REPRODUCE_ISSUED_OOS_TAPE:" + str(repro))
    else:
        repro = {"n_oos_rows_compared": 0, "max_abs_p4_diff_vs_issued_oos_tape": None,
                 "issued_oos_tape_sha256": None, "status": "NOT_APPLICABLE_DEV_ONLY_OOS_NOT_READ"}
    # canonical six-phase where the issued canonical tape has a row, legacy per-row fallback (FI)
    out["phase_legacy"] = out["phase"]
    out["phase_source"] = "LEGACY"
    for f in canonical_phase_files:
        can = pd.read_csv(f, usecols=["country", "issue_at", "state_after"])
        can["month"] = can["issue_at"].astype(str).str[:7]
        lut = dict(zip(zip(can["country"], can["month"]), can["state_after"].map(PHASE_FOR_REGIME_FRAME)))
        keys = list(zip(out["country"], out["month"]))
        hit = np.array([k in lut and isinstance(lut[k], str) for k in keys])
        out.loc[hit, "phase"] = [lut[k] for k, h in zip(keys, hit) if h]
        out.loc[hit, "phase_source"] = "CANONICAL"
    meta = {"model": "LOGISTIC_M11_DEV_FROZEN_REV_NEXT", "n_features": len(cols), "n_dev_fit": int(dev.sum()),
            "reproduction": repro, "phase_source_counts": out["phase_source"].value_counts().to_dict(),
            "dev_probability_status": "IN_SAMPLE_DEV_FIT_NOT_PIT_ISSUED", "include_oos": include_oos}
    return out, meta


def ea_monthly_signals(rq01: pd.DataFrame) -> pd.DataFrame:
    """EA7 monthly regime (argmax of the mean P4, U if any country is missing) and the Bottom F1/F2 votes.

    Reuses rq03_next.regime_frame unchanged (three-valued logic: NA is U, never a silent negative vote)."""
    from .rq03_next import regime_frame
    frame = regime_frame(rq01[["country", "month", "phase", "dd_CLIFS", "z_CISS", "p_rq01_N", "p_rq01_D",
                               "p_rq01_G", "p_rq01_BOTH"]], None, None)
    frame = frame.rename(columns={"signal_reference_month": "month"})
    frame["published_at"] = [pd.Timestamp(month_end(int(m[:4]), int(m[5:7]))) + pd.Timedelta(hours=23, minutes=59)
                             for m in frame["month"]]
    return frame


def weekly_signals(tape: pd.DataFrame, monthly: pd.DataFrame, *, max_age_days: int = 62) -> pd.DataFrame:
    """Backward as-of join: week t sees the latest month whose month END is strictly before week_start(t).

    No month is used before it is complete (G05); a stale (> 62 days, rq03_next registered rule) or incomplete
    signal is regime U.  F3 is the EA equity (E1-E3) cooling + rebound from COMPLETED prior weeks only."""
    m = monthly.sort_values("published_at").reset_index(drop=True)
    w = tape[["week_start", "week_end", "E1", "E2", "E3"]].copy().sort_values("week_start")
    w["_clock"] = w["week_start"] - pd.Timedelta(seconds=1)
    j = pd.merge_asof(w, m, left_on="_clock", right_on="published_at", direction="backward")
    j["signal_month"] = j["month"]
    j["signal_age_days"] = (j["week_start"] - j["published_at"]).dt.days
    stale = j["signal_age_days"].isna() | (j["signal_age_days"] > max_age_days)
    ok = j["signal_status"].eq("RESEARCH_PROXY_COMPLETE_EA7") & ~stale
    j["regime"] = j["regime_exante"].where(ok, "U").fillna("U")
    for f in ("f1", "f2"):
        j[f] = pd.to_numeric(j[f], errors="coerce").where(ok)
    eq = j[["E1", "E2", "E3"]].mean(axis=1).shift(1)
    rv13, rv52 = eq.rolling(13).std(), eq.rolling(52).std()
    ret4 = (1.0 + eq).rolling(4).apply(np.prod, raw=True) - 1.0
    known = rv13.notna() & rv52.notna() & ret4.notna()
    j["f3"] = ((rv13 < rv52) & (ret4 > 0)).astype(float).where(known)
    j["votes"] = j[["f1", "f2", "f3"]].sum(axis=1, min_count=3)
    j["vote_known"] = j["votes"].notna() & j["regime"].isin(("N", "D", "G", "BOTH"))
    keep = ["week_start", "week_end", "signal_month", "published_at", "signal_age_days", "signal_status", "regime",
            "p_rq01_N", "p_rq01_D", "p_rq01_G", "p_rq01_BOTH", "f1", "f2", "f3", "votes", "vote_known"]
    return j[keep].reset_index(drop=True)


def asset_bottom_flags(tape: pd.DataFrame) -> np.ndarray:
    """Asset-specific equity Bottom flags (rq03_next.bottom_asset_indicators, prior weeks only), period-local."""
    from .rq03_next import bottom_asset_indicators
    _, flags = bottom_asset_indicators(tape[["week_end", "E1", "E2", "E3"]].copy())
    return flags[:, :3]
