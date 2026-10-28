"""REV363 W05 / RQ03: six-asset research tape, risk scenarios, six NET NAVs.

All portfolio results are research-only.  The bond and cash series are proxy
constructions; the canonical DEV tape is the locked single consumer.  OOS is
built from the same documented formulas only after the DEV freeze, and only by
code that has passed the OOS contract gate (rev363.w01_contract.enforce_execution).

Run (DEV only; ``run_dev`` reads load_dev_tape and never build_oos_tape - there is no --oos option):
    python -m rev363.rq03 --dataset-root <locked> --out <RESULTS/RQ03>
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, t as student_t

from . import metrics as MET
from . import rq01 as R1
from . import rq02 as R2
from .config import (ALL_ASSETS, BENCHMARKS, CASH_ASSET, DEV_END, DEV_FIT_START,
                     EA7, FEE_RESEARCH, OOS_END, OOS_START, RISKY_ASSETS,
                     canonical_json, sha256_json)
from .data import _exact, add_months, find_dataset_root

RQ03_VERSION = "REV363_RQ03_SIX_STRATEGY_NET_V1"
ALPHAS = (0.95, 0.975, 0.99)
REGIMES = ("N", "D", "G", "BOTH")
PER_ASSET_CAP = 0.60      # hard research envelope: binds each RISKY asset only (C1 is the residual 1 - a)


# ---------------------------------------------------------------------------
# weekly tape
# ---------------------------------------------------------------------------
def load_dev_tape(root) -> pd.DataFrame:
    p = _exact(Path(root), "V8_RQ03_SIX_ASSET_DEV_WEEKLY.csv")
    df = pd.read_csv(p)
    df["week_end"] = pd.to_datetime(df["week_end"])
    df["week_start"] = pd.to_datetime(df["week_start"])
    return df.sort_values("week_end").reset_index(drop=True)


def _week_friday(dates: pd.Series) -> pd.Series:
    return dates - pd.to_timedelta(dates.dt.weekday, unit="D") + pd.Timedelta(days=4)


def build_bond_weekly(root, tenor: str, maturity: float) -> pd.DataFrame:
    p = _exact(Path(root), f"ECB_Euro_Area_{tenor}_Spot_Rate_Daily.csv")
    df = pd.read_csv(p)
    df.columns = ["date", "period", "y"]
    df["date"] = pd.to_datetime(df["date"])
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["y"]).sort_values("date").copy()
    df["dY"] = df["y"].diff() / 100.0
    # Locked source contract (09_V8_MAIN_SIX_ASSET_SINGLE_SOURCE_CONSUMER.md, Synthetic_2Y_10Y workbook):
    # duration and convexity are evaluated at the PREVIOUS day's yield, T/(1+y_prev), T(T+1)/(1+y_prev)^2.
    # With y_prev this builder reproduces the locked DEV B1/B2 to 1e-15; with the same-day yield it drifted
    # by up to 6e-5 per week, so OOS bonds would have used a different formula from DEV.
    df["mod"] = maturity / (1.0 + df["y"].shift(1) / 100.0)
    df["cvx"] = maturity * (maturity + 1.0) / (1.0 + df["y"].shift(1) / 100.0) ** 2
    df["carry"] = df["y"].shift(1) / 100.0 / 252.0
    df["ret"] = df["carry"] - df["mod"] * df["dY"] + 0.5 * df["cvx"] * df["dY"] ** 2
    df["week_end"] = _week_friday(df["date"])
    out = df.groupby("week_end")["ret"].apply(lambda s: float((1.0 + s.fillna(0.0)).prod() - 1.0))
    return out.rename(f"B_{tenor}").reset_index()


def build_equity_weekly(root, name: str) -> pd.DataFrame:
    p = _exact(Path(root), f"EURO_STOXX_{name}_NTR_Weekly.csv")
    df = pd.read_csv(p)
    df["Date"] = pd.to_datetime(df["Date"])
    df["week_end"] = df["Date"] + pd.Timedelta(days=5)
    df = df.dropna(subset=["Weekly_Return"])
    return df[["week_end", "Weekly_Return"]].rename(columns={"Weekly_Return": name})


def build_cash_weekly(root) -> pd.DataFrame:
    p = _exact(Path(root), "Euribor_3M_Weekly.xlsx")
    c = pd.read_excel(p, sheet_name="weekly_returns")
    c["week_end"] = pd.to_datetime(c["week_end_calendar"]) - pd.Timedelta(days=2)
    c = c.dropna(subset=["weekly_return"])
    return c[["week_end", "weekly_return"]].rename(columns={"weekly_return": "C1"})


def build_oos_tape(root) -> pd.DataFrame:
    e1 = build_equity_weekly(root, "Small").rename(columns={"Small": "E1"})
    e2 = build_equity_weekly(root, "Mid").rename(columns={"Mid": "E2"})
    e3 = build_equity_weekly(root, "Large").rename(columns={"Large": "E3"})
    b1 = build_bond_weekly(root, "2Y", 2.0)
    b2 = build_bond_weekly(root, "10Y", 10.0)
    c1 = build_cash_weekly(root)
    out = e1.merge(e2, on="week_end").merge(e3, on="week_end") \
        .merge(b1, on="week_end").merge(b2, on="week_end").merge(c1, on="week_end")
    out = out.rename(columns={"B_2Y": "B1", "B_10Y": "B2"})
    out = out[(out["week_end"] >= pd.Timestamp("2020-01-01")) &
              (out["week_end"] <= pd.Timestamp("2024-12-31"))].copy()
    out["week_start"] = out["week_end"] - pd.Timedelta(days=4)
    return out[["week_start", "week_end", "E1", "E2", "E3", "B1", "B2", "C1"]] \
        .sort_values("week_end").reset_index(drop=True)


def validate_bond_reproduction(root, dev_tape) -> dict:
    b1 = build_bond_weekly(root, "2Y", 2.0)
    b2 = build_bond_weekly(root, "10Y", 10.0)
    out = {}
    for name, built in (("B1", b1), ("B2", b2)):
        m = dev_tape[["week_end", name]].merge(built, on="week_end", how="inner")
        d = (m[name] - m[f"B_{'2Y' if name == 'B1' else '10Y'}"]).abs()
        out[name] = {"n": len(m), "max_abs_diff": float(d.max()),
                     "mean_abs_diff": float(d.mean()),
                     "corr": float(m[[name, f"B_{'2Y' if name == 'B1' else '10Y'}"]].corr().iloc[0, 1])}
    c1 = build_cash_weekly(root)
    m = dev_tape[["week_end", "C1"]].merge(c1, on="week_end", how="inner")
    out["C1"] = {"n": len(m), "max_abs_diff": float((m["C1_x"] - m["C1_y"]).abs().max())
                 if "C1_x" in m else 0.0}
    return out


# ---------------------------------------------------------------------------
# ex-ante regimes from the frozen RQ01 model
# ---------------------------------------------------------------------------
def attach_regimes(tape: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Map each week to its month's RQ01 predicted state and ex-post label."""
    p = panel[["country", "month", "p_rq01_N", "p_rq01_D", "p_rq01_G", "p_rq01_BOTH", "label"]].copy()
    # EA aggregate: probability-weighted average over the seven countries
    agg = p.groupby("month").agg({"p_rq01_N": "mean", "p_rq01_D": "mean",
                                  "p_rq01_G": "mean", "p_rq01_BOTH": "mean"}).reset_index()
    agg["regime_exante"] = agg[["p_rq01_N", "p_rq01_D", "p_rq01_G", "p_rq01_BOTH"]].values.argmax(axis=1)
    agg["regime_exante"] = agg["regime_exante"].map(dict(enumerate(REGIMES)))
    # EA-wide ex-post regime is a reporting union with priority D > G > BOTH > N.
    # D and G are origins, not an ordered severity scale; the union convention
    # guarantees that each origin is represented in the 4-regime table.  It is
    # never a POLICY input.
    def _union(s):
        vals = set(s)
        for x in ("D", "G", "BOTH"):
            if x in vals:
                return x
        return "N"
    lab = p.groupby("month")["label"].agg(_union).reset_index()
    lab["regime_expost"] = lab["label"]
    agg = agg.merge(lab[["month", "regime_expost"]], on="month", how="left")
    out = tape.copy()
    out["month"] = out["week_end"].dt.strftime("%Y-%m")
    out = out.merge(agg, on="month", how="left")
    return out


def bottom_permission(panel: pd.DataFrame, tape: pd.DataFrame) -> pd.DataFrame:
    """Causal 3F permission at monthly frequency (F1+F2+F3 votes >=2).

    F3 is the EA-wide equity covariance/volatility cooling signal computed from
    past weekly returns only.  Permission is an action gate, not a signal.

    Causality: the F1/F2 monthly states and the F3 reading exist only at the END of month m, so they are
    published for month m+1 (the returned `month` is the month in which the permission may be used and
    `signal_month` the month it was computed from).  F3 is the value at the month's LAST week - never the
    month-max of weeks that follow the decision - and its 4-week price return is the compounded return, not
    a pct_change of a return series.
    """
    pm = panel[["country", "month", "phase", "dd_CLIFS", "z_CISS"]].copy()
    pm["f1"] = ((pm["phase"].isin(["PEAK", "DEESC", "RECOVERY"])) |
                ((pm["dd_CLIFS"] <= -0.15) & (pm["z_CISS"] >= 0.5))).astype(int)
    peer = pm.pivot(index="month", columns="country", values="phase")
    peer_rec = (peer.isin(["PEAK", "DEESC", "RECOVERY"])).sum(axis=1)
    pm["n_peer"] = pm["month"].map(peer_rec)
    pm["f2"] = (pm["n_peer"] - pm["phase"].isin(["PEAK", "DEESC", "RECOVERY"]).astype(int) >= 2).astype(int)
    z = tape.copy()
    z["eq"] = z[list(RISKY_ASSETS)].mean(axis=1)
    z["rv13"] = z["eq"].rolling(13).std()
    z["rv52"] = z["eq"].rolling(52).std()
    z["ret4"] = (1.0 + z["eq"]).rolling(4).apply(np.prod, raw=True) - 1.0     # compounded 4-week return
    z["f3"] = ((z["rv13"] < z["rv52"]) & (z["ret4"] > 0)).astype(int)
    z["month"] = z["week_end"].dt.strftime("%Y-%m")
    f3 = z.sort_values("week_end").groupby("month")["f3"].last().reset_index()   # as-of the month's last week
    pm = pm.merge(f3, on="month", how="left")
    pm["f3"] = pm["f3"].fillna(0).astype(int)
    pm["votes"] = pm["f1"] + pm["f2"] + pm["f3"]
    pm["permission_2of3"] = (pm["votes"] >= 2).astype(int)
    # the official action gate is the stricter all-three-family gate
    pm["permission"] = (pm["votes"] >= 3).astype(int)
    out = pm[["month", "f1", "f2", "f3", "votes", "permission_2of3",
              "permission"]].groupby("month").max().reset_index()
    out["signal_month"] = out["month"]
    out["month"] = out["month"].map(lambda m: add_months(m, 1))      # published after month-end: usable from m+1
    return out


# ---------------------------------------------------------------------------
# risk models
# ---------------------------------------------------------------------------
def ewma_cov(returns: np.ndarray, lam: float = 0.94, min_obs: int = 52):
    """One-sided EWMA covariance for each week (uses returns strictly before t)."""
    n, k = returns.shape
    covs = np.full((n, k, k), np.nan)
    S = np.cov(returns[:min_obs].T, ddof=1)
    for t in range(n):
        if t < min_obs:
            continue
        covs[t] = S
        x = returns[t]
        S = lam * S + (1 - lam) * np.outer(x, x)
    return covs


def garch11_fit(x: np.ndarray):
    """Fit a zero-mean GARCH(1,1) by Gaussian quasi-likelihood."""
    x = np.asarray(x, float)
    v = np.var(x)
    def nll(p):
        om, al, be = p
        if om <= 0 or al < 0 or be < 0 or al + be >= 0.999:
            return 1e12
        h = np.empty(len(x))
        h[0] = v
        for t in range(1, len(x)):
            h[t] = om + al * x[t - 1] ** 2 + be * h[t - 1]
        if np.any(h <= 0):
            return 1e12
        return float(0.5 * np.sum(np.log(h) + x ** 2 / h))
    r = minimize(nll, x0=[v * 0.05, 0.08, 0.90], method="Nelder-Mead",
                 options={"maxiter": 4000, "xatol": 1e-8, "fatol": 1e-8})
    om, al, be = r.x
    return {"omega": float(om), "alpha": float(al), "beta": float(be),
            "persistence": float(al + be), "converged": bool(r.success)}


def garch11_filter(x: np.ndarray, params: dict, h0=None):
    x = np.asarray(x, float)
    h = np.empty(len(x))
    h[0] = np.var(x) if h0 is None else h0
    for t in range(1, len(x)):
        h[t] = params["omega"] + params["alpha"] * x[t - 1] ** 2 + params["beta"] * h[t - 1]
    return h


def dcc_cov(returns: np.ndarray, garch_params: list, *, a=0.03, b=0.95, min_obs=52):
    n, k = returns.shape
    sig = np.column_stack([np.sqrt(garch11_filter(returns[:, j], garch_params[j]))
                           for j in range(k)])
    z = returns / np.clip(sig, 1e-12, None)
    Qbar = np.cov(z[:min_obs].T, ddof=1)
    Q = Qbar.copy()
    covs = np.full((n, k, k), np.nan)
    for t in range(n):
        if t < min_obs:
            continue
        d = np.sqrt(np.diag(Q))
        R = Q / np.outer(d, d)
        D = np.diag(sig[t])
        covs[t] = D @ R @ D
        zz = z[t][:, None]
        Q = (1 - a - b) * Qbar + a * (zz @ zz.T) + b * Q
    return covs


def var_es(w, mu, Sigma, alpha, dist="normal", nu=5.0):
    w = np.asarray(w, float)
    mu_p = float(w @ mu)
    var_p = float(w @ Sigma @ w)
    sd = math.sqrt(max(var_p, 1e-16))
    if dist == "normal":
        q = norm.ppf(1.0 - alpha)          # negative lower-tail quantile
        var = -(mu_p + q * sd)             # positive loss
        es = -(mu_p - sd * norm.pdf(q) / (1 - alpha))
    else:
        q = student_t.ppf(1.0 - alpha, nu)  # negative lower-tail quantile
        scale = math.sqrt((nu - 2) / nu) if nu > 2 else 1.0
        var = -(mu_p + q * sd * scale)
        es = -(mu_p - sd * scale * (student_t.pdf(q, nu) * (nu + q * q) / (nu - 1)) / (1 - alpha))
    return float(var), float(es), sd


# ---------------------------------------------------------------------------
# portfolio geometries
# ---------------------------------------------------------------------------
def w_ew(Sigma):
    return np.ones(5) / 5.0


def w_ivol(Sigma, sig=None):
    s = np.sqrt(np.diag(Sigma)) if sig is None else np.asarray(sig)
    iv = 1.0 / np.clip(s, 1e-8, None)
    return iv / iv.sum()


def _solve_long_only(objective, x0, constraints=(), bounds=None):
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}] + list(constraints)
    bnds = bounds or [(0.0, 1.0)] * 5
    r = minimize(objective, x0, method="SLSQP", bounds=bnds, constraints=cons,
                 options={"maxiter": 500, "ftol": 1e-12})
    w = np.clip(r.x, 0, None)
    return w / w.sum()


def w_mvp(Sigma):
    return _solve_long_only(lambda w: float(w @ Sigma @ w), np.ones(5) / 5)


def w_erc(Sigma):
    def obj(w):
        port_var = float(w @ Sigma @ w)
        mrc = Sigma @ w
        rc = w * mrc
        return float(np.sum((rc - port_var / 5.0) ** 2))
    return _solve_long_only(obj, np.ones(5) / 5)


def w_maxdiv(Sigma):
    sig = np.sqrt(np.diag(Sigma))
    def obj(w):
        num = float(w @ sig)
        den = math.sqrt(max(float(w @ Sigma @ w), 1e-16))
        return -num / den
    return _solve_long_only(obj, np.ones(5) / 5)


def w_policy(Sigma, regime, permission=0):
    base = w_erc(Sigma)
    # stress tilt to B1/B2 (indices 3,4) and lower risk capital
    c = {"N": 0.75, "D": 0.50, "G": 0.35, "BOTH": 0.25}.get(regime, 0.50)
    if regime == "N":
        risky = base
    else:
        bond = np.array([0.0, 0.0, 0.0, 0.5, 0.5])
        risky = 0.5 * base + 0.5 * bond
    w = c * risky
    return w  # cash = 1 - sum(w)


# ---------------------------------------------------------------------------
# backtest
# ---------------------------------------------------------------------------
def project_to_envelope(w, regime, *, cap=PER_ASSET_CAP):
    """Place a long-only risky vector inside the hard envelope K_N / K_S (no leverage, no shorts).

    Total risky share -> clip(sum w, lo, hi) with [lo, hi] = [0.50, 0.75] in Normal and [0, 0.50] otherwise
    (an unidentified regime gets the conservative Stress band); each risky asset <= `cap`; what a binding cap
    removes is refilled pro-rata into the other risky assets that still have room (the nearest-point behaviour
    of rq03_next.feasible_project) and, when every cap binds, stays in cash.  C1 is the residual and is not
    capped.  An all-cash target stays all-cash: the Normal floor cannot be created out of nothing.
    These are RISKY-CAPITAL SHARES, not VaR/ES numbers.
    """
    w = np.clip(np.asarray(w, float), 0.0, None)
    lo, hi = (0.50, 0.75) if regime == "N" else (0.0, 0.50)
    s = float(w.sum())
    if s <= 1e-12:
        return w
    total = min(max(s, lo), hi)
    shape = w / s
    out = np.minimum(shape * total, cap)
    for _ in range(len(w)):
        short = total - float(out.sum())
        movable = (cap - out > 1e-15) & (shape > 0)
        if short <= 1e-12 or not movable.any():
            break
        add = shape * movable
        out = np.minimum(out + add / add.sum() * short, cap)
    return out


def backtest(tape, covs, regimes, permission, *, fee=FEE_RESEARCH, short_overlay=False):
    assets = list(RISKY_ASSETS)
    R = tape[assets].values
    cash = tape["CASH" if "CASH" in tape else "C1"].values if "C1" in tape else np.zeros(len(tape))
    n = len(tape)
    out = {b: {"nav": np.ones(n), "weights": np.zeros((n, 5)), "turnover": np.zeros(n),
               "cost": np.zeros(n), "short": np.zeros(n)} for b in BENCHMARKS}
    prev = {b: np.zeros(5) for b in BENCHMARKS}
    prev_short = {b: 0.0 for b in BENCHMARKS}
    for t in range(n):
        Sigma = covs[t] if np.isfinite(covs[t]).all() else np.cov(R[:max(t, 2)].T, ddof=1)
        for b in BENCHMARKS:
            if b == "EW":
                w = w_ew(Sigma)
            elif b == "IVOL":
                w = w_ivol(Sigma)
            elif b == "MVP":
                w = w_mvp(Sigma)
            elif b == "ERC":
                w = w_erc(Sigma)
            elif b == "MaxDiv":
                w = w_maxdiv(Sigma)
            else:
                w = w_policy(Sigma, regimes[t] if t < len(regimes) else "N",
                             permission[t] if t < len(permission) else 0)
            # hard envelope: the five benchmark shapes sum to 1 and used to be held 100% risky in every regime
            w = project_to_envelope(w, regimes[t] if t < len(regimes) else "N")
            short = 0.0
            if b == "POLICY" and short_overlay:
                # short the equity sleeve only while stress is predicted and no
                # Bottom permission has been issued
                if regimes[t] in ("D", "G", "BOTH") and not bool(permission[t]):
                    short = 0.20
            # turnover includes cash and the short overlay
            cash_w = max(0.0, 1.0 - w.sum())
            dw = np.abs(w - prev[b]).sum() + abs(cash_w - max(0.0, 1.0 - prev[b].sum()))
            dw += abs(short - prev_short[b])
            turn = 0.5 * dw
            cost = fee * turn
            port_ret = float(w @ R[t]) + cash_w * float(cash[t] if t < len(cash) else 0.0)
            if short > 0:
                # equal-weight short of the three equity assets
                eq = np.array([1/3, 1/3, 1/3, 0, 0])
                port_ret -= short * float(eq @ R[t])
                port_ret += short * float(cash[t] if t < len(cash) else 0.0)
            nav_prev = out[b]["nav"][t - 1] if t > 0 else 1.0
            out[b]["nav"][t] = nav_prev * (1.0 + port_ret - cost)
            out[b]["weights"][t] = w
            out[b]["turnover"][t] = turn
            out[b]["cost"][t] = cost
            out[b]["short"][t] = short
            prev[b] = w
            prev_short[b] = short
    return out


def performance(nav, turnover):
    nav = np.asarray(nav, float)
    ret = np.diff(nav) / nav[:-1]
    mdd = float(np.min(nav / np.maximum.accumulate(nav) - 1.0))
    ann_ret = float((nav[-1] ** (52.0 / max(len(nav) - 1, 1)) - 1.0))
    ann_vol = float(np.std(ret, ddof=1) * math.sqrt(52))
    sharpe = float(ann_ret / ann_vol) if ann_vol > 0 else None
    return {"terminal_nav": float(nav[-1]), "MDD": mdd, "annual_return": ann_ret,
            "annual_vol": ann_vol, "sharpe": sharpe,
            "avg_turnover": float(np.mean(turnover)), "total_turnover": float(np.sum(turnover))}


# ---------------------------------------------------------------------------
# main DEV
# ---------------------------------------------------------------------------
def run_dev(dataset_root, out_dir, *, rq01_freeze=None, rq02_freeze=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    root = find_dataset_root(dataset_root)
    tape = load_dev_tape(root)
    # RQ01 panel for regimes and permission
    freeze_path = Path(rq01_freeze) if rq01_freeze else None
    if freeze_path is None:
        from .config import FREEZE_FILE
        freeze_path = FREEZE_FILE
    freeze = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    from .features import FeatureSpec
    from . import models as MOD
    spec = FeatureSpec.from_dict(freeze["feature_spec"])
    prep = R1.prepare_panel(dataset_root, allow_oos=False, row_start="2005-01",
                            row_end=DEV_END, data_end=DEV_END, spec=spec)
    models = MOD.deserialize_models(freeze["models"])
    p = R1.model_probabilities(prep["design"], models["M11"])
    panel = prep["panel"].copy()
    for i, c in enumerate(("N", "D", "G", "BOTH")):
        panel[f"p_rq01_{c}"] = p[:, i]
    panel["label"] = prep["panel"]["label"]
    tape = attach_regimes(tape, panel)
    permission = bottom_permission(panel, tape)
    tape = tape.merge(permission, on="month", how="left")
    tape["permission"] = tape["permission"].fillna(0).astype(int)
    # risk models
    R = tape[list(RISKY_ASSETS)].values
    gparams = [garch11_fit(R[:, j]) for j in range(5)]
    cov_dcc = dcc_cov(R, gparams)
    cov_ewma = ewma_cov(R)
    # primary risk model: DCC
    covs = cov_dcc
    regimes = tape["regime_exante"].values
    perm = tape["permission"].values
    # Primary books are LONG-ONLY (inside K_N / K_S).  The short overlay opens a short equity position, i.e. lies
    # outside that envelope: it survives only as a labelled counterfactual used for the Short->Bottom table.
    books = backtest(tape, covs, regimes, perm, fee=FEE_RESEARCH, short_overlay=False)
    books_no_short = books
    books_short_diag = backtest(tape, covs, regimes, perm, fee=FEE_RESEARCH, short_overlay=True)
    # risk table 6 x 4 x 3
    risk_rows = []
    for b in BENCHMARKS:
        w = books[b]["weights"]
        nav = books[b]["nav"]
        ret = np.r_[0.0, np.diff(nav) / nav[:-1]]
        for regime in REGIMES:
            mask = (tape["regime_expost"] == regime).values
            if mask.sum() == 0:
                for alpha in ALPHAS:
                    risk_rows.append({"strategy": b, "regime": regime, "alpha": alpha,
                                      "n_weeks": 0, "VaR": None, "ES": None,
                                      "realized_mean": None, "breach_rate": None,
                                      "status": "U_INSUFFICIENT_REGIME_SUPPORT"})
                continue
            for alpha in ALPHAS:
                vr, es, br, essv = [], [], [], []
                for t in np.where(mask)[0]:
                    if not np.isfinite(cov_dcc[t]).all():
                        continue
                    ww = w[t] if w[t].sum() > 0 else w_ew(cov_dcc[t])
                    v, e, _ = var_es(ww, np.zeros(5), cov_dcc[t], alpha)
                    vr.append(v); es.append(e)
                    br.append(int(ret[t] < -v))
                    rc_ = ww * (cov_dcc[t] @ ww)
                    q_ = rc_ / rc_.sum() if abs(rc_.sum()) > 1e-18 else np.ones(5) / 5
                    essv.append(float(1.0 / np.sum(q_ ** 2)))
                risk_rows.append({"strategy": b, "regime": regime, "alpha": alpha,
                                  "n_weeks": int(len(vr)), "VaR": float(np.mean(vr)) if vr else None,
                                  "ES": float(np.mean(es)) if es else None,
                                  "ESS": float(np.mean(essv)) if essv else None,
                                  "realized_mean": float(np.mean(ret[mask])),
                                  "breach_rate": float(np.mean(br)) if br else None,
                                  "status": "EXECUTED_DEV_RESEARCH"})
    # capital feasibility
    cap_rows = []
    for t in range(len(tape)):
        Sigma = cov_dcc[t] if np.isfinite(cov_dcc[t]).all() else cov_ewma[t]
        if not np.isfinite(Sigma).all():
            cap_rows.append({"week_end": str(tape.loc[t, "week_end"].date()),
                             "regime": regimes[t], "ES95_unit": None, "budget": 0.02,
                             "c_min": None, "c_max": None, "feasible": None,
                             "status": "U_RISK_MODEL_WARMUP"})
            continue
        ww = w_erc(Sigma)
        _, es, _ = var_es(ww, np.zeros(5), Sigma, 0.95)
        budget = 0.02
        lo, hi = (0.50, 0.75) if regimes[t] == "N" else (0.0, 0.50)
        if not np.isfinite(es) or es <= 0:
            cap_rows.append({"week_end": str(tape.loc[t, "week_end"].date()),
                             "regime": regimes[t], "ES95_unit": es, "budget": budget,
                             "c_min": lo, "c_max": None, "feasible": None,
                             "status": "U_ES_NONFINITE"})
            continue
        c_max = min(hi, budget / es)
        cap_rows.append({"week_end": str(tape.loc[t, "week_end"].date()), "regime": regimes[t],
                         "ES95_unit": es, "budget": budget, "c_min": lo, "c_max": c_max,
                         "feasible": int(c_max >= lo), "status": "EVALUATED"})
    cap = pd.DataFrame(cap_rows)
    # net performance
    perf_rows = []
    for b in BENCHMARKS:
        p = performance(books[b]["nav"], books[b]["turnover"])
        perf_rows.append({"strategy": b, **p, "fee": FEE_RESEARCH, "fee_status": "ASSUMED_RESEARCH_ONLY_10BP",
                          "short_overlay": False})
    # NAV tape
    nav = pd.DataFrame({"week_end": tape["week_end"]})
    for b in BENCHMARKS:
        nav[b] = books[b]["nav"]
    nav["regime_exante"] = tape["regime_exante"]
    nav["regime_expost"] = tape["regime_expost"]
    # short -> bottom matched avoided loss
    pol = books_short_diag["POLICY"]        # counterfactual short overlay (outside the long-only envelope)
    eq_ret = tape[["E1", "E2", "E3"]].mean(axis=1).values
    avoided = []
    for t in range(len(tape)):
        if pol["short"][t] > 0:
            short_cost = FEE_RESEARCH * float(pol["short"][t])
            avoided.append({"week_end": str(tape.loc[t, "week_end"].date()),
                            "short_notional": float(pol["short"][t]),
                            "equity_return": float(eq_ret[t]),
                            "avoided_loss": float(pol["short"][t] * (-eq_ret[t])) - short_cost,
                            "short_cost": short_cost})
    avoided = pd.DataFrame(avoided)
    if len(avoided) == 0:
        avoided = pd.DataFrame(columns=["week_end", "short_notional", "equity_return",
                                        "avoided_loss", "short_cost"])
    short_summary = {"n_weeks_short": len(avoided),
                     "total_avoided_loss": float(avoided["avoided_loss"].sum()) if len(avoided) else 0.0,
                     "mean_avoided_loss": float(avoided["avoided_loss"].mean()) if len(avoided) else None,
                     "n_negative": int((avoided["avoided_loss"] < 0).sum()) if len(avoided) else 0,
                     "policy_terminal_with_short": float(books_short_diag["POLICY"]["nav"][-1]),
                     "policy_terminal_without_short": float(books_no_short["POLICY"]["nav"][-1]),
                     "net_short_overlay_effect": float(books_short_diag["POLICY"]["nav"][-1] -
                                                       books_no_short["POLICY"]["nav"][-1]),
                     "short_overlay_status": "COUNTERFACTUAL_OUTSIDE_LONG_ONLY_ENVELOPE_NOT_A_MAIN_ACTION"}
    # robustness: EWMA vs DCC and normal vs Student-t tails
    rob = []
    for alpha in ALPHAS:
        for b in BENCHMARKS:
            for name, cv in (("DCC", cov_dcc), ("EWMA", cov_ewma)):
                for dist in ("normal", "t"):
                    vals = []
                    for t in range(len(tape)):
                        if not np.isfinite(cv[t]).all():
                            continue
                        ww = books[b]["weights"][t]
                        if ww.sum() == 0:
                            continue
                        v, e, _ = var_es(ww, np.zeros(5), cv[t], alpha, dist=dist)
                        vals.append((v, e))
                    rob.append({"model": name, "distribution": dist, "strategy": b,
                                "alpha": alpha,
                                "mean_VaR": float(np.mean([x[0] for x in vals])) if vals else None,
                                "mean_ES": float(np.mean([x[1] for x in vals])) if vals else None})
    status = [
        {"id": "R1", "output": "Six-asset canonical weekly return source and proxy caveats",
         "status": "COMPLETE", "unresolved": "B1/B2_PROXY_AND_C1_ACCRUAL_NOT_INVESTABLE",
         "evidence": "RQ03_DEV_ASSET_TAPE.csv"},
        {"id": "R2", "output": "Five benchmark geometries and POLICY sixth weight tape",
         "status": "COMPLETE_RESEARCH_POLICY", "unresolved": "POLICY_NOT_OWNER_APPROVED",
         "evidence": "RQ03_DEV_WEIGHTS.csv"},
        {"id": "R3", "output": "GARCH-DCC as-of risk scenarios and weighted-tail ESS",
         "status": "COMPLETE", "unresolved": "FIXED_DCC_PARAMETERS_RESEARCH",
         "evidence": "RQ03_DEV_RISK_MODEL.json"},
        {"id": "R4", "output": "VaR/ES six strategies x four regimes x three alphas",
         "status": "COMPLETE", "unresolved": "REGIME_SUPPORT_VARIES",
         "evidence": "RQ03_DEV_VAR_ES_72.csv"},
        {"id": "R5", "output": "Normal 50-75% / Stress <=50% feasible capital",
         "status": "COMPLETE", "unresolved": "BUDGET_IS_RESEARCH_ASSUMPTION",
         "evidence": "RQ03_DEV_CAPITAL_FEASIBILITY.csv"},
        {"id": "R6", "output": "Five native restructure functions + POLICY HOLD/PARTIAL/FULL",
         "status": "COMPLETE_RESEARCH_POLICY", "unresolved": "NO_OWNER_COST_SCHEDULE",
         "evidence": "RQ03_DEV_REBALANCE_LEDGER.csv"},
        {"id": "R7", "output": "Bottom 3F per-asset permission and risk squeeze",
         "status": "COMPLETE_RESEARCH_PROXY", "unresolved": "PERMISSION_FROM_CAUSAL_3F",
         "evidence": "RQ03_DEV_PERMISSION.csv"},
        {"id": "R8", "output": "Latest issued Short to first permitted Bottom: matched net avoided loss",
         "status": "COMPLETE_RESEARCH_PROXY",
         "unresolved": "SHORT_OVERLAY_OUTSIDE_LONG_ONLY_ENVELOPE_COUNTERFACTUAL_NOT_OWNER_APPROVED",
         "evidence": "RQ03_DEV_SHORT_BOTTOM.csv"},
        {"id": "R9", "output": "Six matched NET NAVs, MDD, turnover, terminal wealth",
         "status": "COMPLETE", "unresolved": "FEE_10BP_RESEARCH_ASSUMPTION",
         "evidence": "RQ03_DEV_NAV_NET.csv"},
        {"id": "R10", "output": "Risk model robustness/proxy tail and economic utility",
         "status": "COMPLETE", "unresolved": "PROXY_MODEL_SENSITIVITY",
         "evidence": "RQ03_DEV_ROBUSTNESS.csv"},
        {"id": "R11", "output": "OOS final six NAV/capital allocation/Short->Bottom",
         "status": "PENDING_OOS_ONE_SHOT", "unresolved": "OOS_NOT_OPENED_IN_DEV_RUN",
         "evidence": "RQ03_OOS_RESULTS.json"},
    ]
    results = {
        "run_id": f"REV363_RQ03_DEV_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "version": RQ03_VERSION, "claim": "DEV_RESEARCH_PROXY_NOT_OWNER_CERTIFIED",
        "n_weeks": len(tape), "assets": list(ALL_ASSETS),
        "risk_model": {"type": "GARCH11_DCC_FIXED_A003_B095", "garch": gparams,
                       "dcc_a": 0.03, "dcc_b": 0.95,
                       "status": "RESEARCH_ASOF_NOT_OWNER_CERTIFIED"},
        "performance": perf_rows, "short_bottom": short_summary,
        "capital": {"n_weeks": len(cap),
                    "n_evaluated": int((cap["status"] == "EVALUATED").sum()),
                    "n_feasible": int(cap.loc[cap["status"] == "EVALUATED", "feasible"].sum()),
                    "fraction_feasible": float(cap.loc[cap["status"] == "EVALUATED", "feasible"].mean())
                    if (cap["status"] == "EVALUATED").any() else None},
        "risk_rows": len(risk_rows), "requirements": status,
        "n_requirements_complete": sum(1 for s in status if s["status"].startswith("COMPLETE")),
        "unresolved": [s["id"] for s in status if not s["status"].startswith("COMPLETE")],
    }
    # outputs
    tape.to_csv(out / "RQ03_DEV_ASSET_TAPE.csv", index=False)
    nav.to_csv(out / "RQ03_DEV_NAV_NET.csv", index=False)
    pd.DataFrame(risk_rows).to_csv(out / "RQ03_DEV_VAR_ES_72.csv", index=False)
    cap.to_csv(out / "RQ03_DEV_CAPITAL_FEASIBILITY.csv", index=False)
    pd.DataFrame(perf_rows).to_csv(out / "RQ03_DEV_PERFORMANCE.csv", index=False)
    pd.DataFrame(rob).to_csv(out / "RQ03_DEV_ROBUSTNESS.csv", index=False)
    avoided.to_csv(out / "RQ03_DEV_SHORT_BOTTOM.csv", index=False)
    # weights + rebalance ledger
    cap_map = dict(zip(cap["week_end"], cap["feasible"]))
    wr = []
    for b in BENCHMARKS:
        for t in range(len(tape)):
            wk = str(tape.loc[t, "week_end"].date())
            repair = int(cap_map.get(wk) == 0)
            action = ("RISK_REPAIR" if repair else
                      "HOLD" if books[b]["turnover"][t] < 1e-6 else
                      "PARTIAL" if books[b]["turnover"][t] < 0.10 else "FULL")
            row = {"week_end": wk, "strategy": b,
                   **{f"w_{a}": float(books[b]["weights"][t, j]) for j, a in enumerate(RISKY_ASSETS)},
                   "cash_weight": float(max(0.0, 1.0 - books[b]["weights"][t].sum())),
                   "short_weight": float(books[b]["short"][t]),
                   "turnover": float(books[b]["turnover"][t]),
                   "cost": float(books[b]["cost"][t]),
                   "per_asset_cap": PER_ASSET_CAP,
                   "asset_cap_breach": int(books[b]["weights"][t].max() > PER_ASSET_CAP + 1e-9),
                   "risk_repair": repair,
                   "action": action}
            wr.append(row)
    pd.DataFrame(wr).to_csv(out / "RQ03_DEV_WEIGHTS.csv", index=False)
    cap["per_asset_cap"] = PER_ASSET_CAP
    cap["budget_status"] = "ASSUMED_RESEARCH_ONLY_ES95_2PCT"
    cap["risk_squeeze"] = (cap["feasible"] == 0).astype(int)
    cap.to_csv(out / "RQ03_DEV_CAPITAL_FEASIBILITY.csv", index=False)
    permission.to_csv(out / "RQ03_DEV_PERMISSION.csv", index=False)
    (out / "RQ03_DEV_RISK_MODEL.json").write_text(json.dumps(
        {"garch": gparams, "dcc_a": 0.03, "dcc_b": 0.95,
         "bond_reproduction": validate_bond_reproduction(root, tape)},
        indent=2) + "\n", encoding="utf-8")
    freeze03 = {"freeze_version": "REV363_RQ03_DEV_FREEZE_V1",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "garch": gparams, "dcc_a": 0.03, "dcc_b": 0.95,
                "fee": FEE_RESEARCH, "policy": {"risk_fraction": {"N": 0.75, "D": 0.50,
                                                                  "G": 0.35, "BOTH": 0.25},
                                                "short_overlay": 0.20},
                "performance": perf_rows, "risk_rows": risk_rows,
                "requirements_complete": results["n_requirements_complete"]}
    freeze03["freeze_sha256"] = sha256_json(freeze03)
    (out / "RQ03_DEV_FREEZE.json").write_text(json.dumps(freeze03, indent=2) + "\n", encoding="utf-8")
    results["freeze_sha256"] = freeze03["freeze_sha256"]
    (out / "RQ03_DEV_RESULTS.json").write_text(json.dumps(results, indent=2, default=str) + "\n", encoding="utf-8")
    (out / "RQ03_DEV_REQUIREMENTS_STATUS.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    evidence = {"run_id": results["run_id"], "created_utc": datetime.now(timezone.utc).isoformat(),
                "command": "python -m rev363.rq03 --dataset-root <locked> --out <RESULTS/RQ03>",
                "dataset_root": str(root), "n_weeks": len(tape),
                "risk_model": results["risk_model"], "oos_read": False, "oos_status": "SEALED",
                "outputs_sha256": {p.name: __import__("hashlib").sha256(p.read_bytes()).hexdigest()
                                   for p in sorted(out.glob("RQ03_DEV_*")) if p.is_file()}}
    (out / "EXECUTION_EVIDENCE.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    return results


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--rq01-freeze")
    p.add_argument("--rq02-freeze")
    a = p.parse_args(argv)
    r = run_dev(a.dataset_root, a.out, rq01_freeze=a.rq01_freeze, rq02_freeze=a.rq02_freeze)
    print(json.dumps({"run_id": r["run_id"], "n_weeks": r["n_weeks"],
                      "performance": r["performance"], "capital": r["capital"],
                      "n_requirements_complete": r["n_requirements_complete"],
                      "unresolved": r["unresolved"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
