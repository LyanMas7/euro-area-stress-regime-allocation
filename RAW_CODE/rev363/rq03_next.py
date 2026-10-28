"""RQ03 corrected execution: one continuous W0(2005) NAV, pre-trade feasibility,
one-sided GARCH/DCC and tail-scenario ESS.

Defects repaired
----------------
P0-02  One ``stateful_backtest`` runs 2005-01 .. 2024-12 for all six portfolios
       with no NAV/cash/weight reset at 2019->2020 or 2021-04->2021-05.
P0-06  Monthly regime/permission signals are joined to weeks with a strict
       backward as-of rule (publication lag), so a week cannot see a month that
       was not yet published.
P0-07  Feasibility projection happens *before* turnover/fee/NAV.  When the
       feasible set is empty, research code labels emergency liquidation.
       Such a trade requires owner authorization and is NOT live-certified.
P0-08  GARCH parameters use FULL DEV vintage; historical DEV results are
       retrospective (not PIT-executable). OOS filtration is one-sided but
       remains post-exposure research. DCC warms on first 52 weeks.
P1-17  ESS is computed on Monte-Carlo tail *scenario* weights, not on asset
       risk contributions.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

from rev363.config import BENCHMARKS, FEE_RESEARCH, RISKY_ASSETS, EA7
from rev363.rq03 import (build_oos_tape, dcc_cov, garch11_fit, garch11_filter,
                         load_dev_tape, w_erc, w_ew, w_ivol, w_maxdiv, w_mvp)
from rev363.labels_next import asof_week_join, STATUS_SCORABLE
from rev363.util import jdump, run_id, utcnow

ALPHAS = (0.95, 0.975, 0.99)
REGIMES = ("N", "D", "G", "BOTH")
ASSETS = ("E1", "E2", "E3", "B1", "B2", "C1")
RISK_BUDGET = 0.02          # ES95 loss budget on risky capital
PER_ASSET_CAP = 0.60
DEV_END = "2019-12"


# ---------------------------------------------------------------------------
# data assembly
# ---------------------------------------------------------------------------
def full_tape(root) -> pd.DataFrame:
    dev = load_dev_tape(root)
    oos = build_oos_tape(root)
    oos["week_start"] = pd.to_datetime(oos["week_start"])
    oos["week_end"] = pd.to_datetime(oos["week_end"])
    cols = ["week_start", "week_end"] + list(ASSETS)
    both = pd.concat([dev[cols], oos[cols]], ignore_index=True)
    both = both.drop_duplicates("week_end").sort_values("week_end").reset_index(drop=True)
    both["month"] = both["week_end"].dt.strftime("%Y-%m")
    return both


def regime_frame(panel: pd.DataFrame, rq01_score: pd.Series, rq01_thr: float) -> pd.DataFrame:
    """Research-only monthly EA7 aggregation; invalid input stays U.

    For a real issued panel, release dates and EA8/IE must be certified by
    their upstream owner. Never impute an absent month/country/probability as N.
    """
    required = ("country", "month", "phase", "dd_CLIFS", "z_CISS",
                "p_rq01_N", "p_rq01_D", "p_rq01_G", "p_rq01_BOTH")
    if panel is None or any(c not in panel.columns for c in required):
        raise ValueError("MISSING_UPSTREAM_RQ01_PANEL_OR_COLUMNS")
    p = panel[list(required)].copy()
    if p[["country", "month"]].duplicated().any():
        raise ValueError("DUPLICATED_UPSTREAM_COUNTRY_MONTH")
    if not p["country"].isin(EA7).all():
        raise ValueError("UNEXPECTED_COUNTRY_UPSTREAM_ROSTER_REQUIRES_OWNER_RECONCILIATION")
    probs = list(required[5:])
    p[probs] = p[probs].apply(pd.to_numeric, errors="coerce")
    row_ok = (np.isfinite(p[probs].to_numpy(dtype=float)).all(axis=1) &
              (p[probs].to_numpy(dtype=float) >= 0).all(axis=1) &
              (p[probs].to_numpy(dtype=float) <= 1).all(axis=1) &
              np.isclose(p[probs].sum(axis=1), 1, atol=1e-6))
    p["_valid_prob"] = row_ok
    by = p.groupby("month", sort=True)
    coverage = by.agg(n_countries=("country", "nunique"),
                      n_valid_probs=("_valid_prob", "sum"))
    for c in probs:
        coverage[c] = by[c].mean()
    complete = (coverage.n_countries.eq(len(EA7)) &
                coverage.n_valid_probs.eq(len(EA7)))
    coverage["signal_status"] = np.where(complete, "RESEARCH_PROXY_COMPLETE_EA7", "U_INCOMPLETE_UPSTREAM")
    coverage["regime_exante"] = "U"
    probability_cols = ["p_rq01_N", "p_rq01_D", "p_rq01_G", "p_rq01_BOTH"]
    if complete.any():
        coverage.loc[complete, "regime_exante"] = np.asarray(REGIMES)[
            coverage.loc[complete, probability_cols].to_numpy().argmax(axis=1)]
    # Three-valued source logic: NA is U, never silently a negative vote.
    # A definite TRUE in an OR can resolve even if another input is U; for
    # a definite FALSE every required operand must be available.
    accepted_phases = {"PRE", "ENTRY", "PEAK", "DEESC", "RECOVERY", "EXIT", "N"}
    event_phases = {"PEAK", "DEESC", "RECOVERY"}
    for field in ("dd_CLIFS", "z_CISS"):
        p[field] = pd.to_numeric(p[field], errors="coerce")
    p["f1"] = np.nan
    p["f2"] = np.nan
    for idx, row in p.iterrows():
        phase = row["phase"]
        phase_ok = isinstance(phase, str) and phase in accepted_phases
        if phase_ok and phase in event_phases:
            p.at[idx, "f1"] = 1.0
        elif phase_ok:
            dd, z = row["dd_CLIFS"], row["z_CISS"]
            if (np.isfinite(dd) and dd > -0.15) or (np.isfinite(z) and z < 0.5):
                p.at[idx, "f1"] = 0.0
            elif np.isfinite(dd) and dd <= -0.15 and np.isfinite(z):
                p.at[idx, "f1"] = 1.0
            # Otherwise U: no operand definitively FALSE, some are missing.
    for month, group in p.groupby("month", sort=False):
        for idx, row in group.iterrows():
            peers = group.loc[group.index != idx, "phase"]
            known = peers.isin(event_phases).sum()
            unknown = (~peers.isin(accepted_phases)).sum()
            if known >= 2:
                p.at[idx, "f2"] = 1.0
            elif known + unknown < 2:
                p.at[idx, "f2"] = 0.0
            # Otherwise U: missing peer phase can swing the threshold.
    def tri_any(series):
        vals = series.to_numpy(dtype=float)
        if np.any(vals == 1):
            return 1.0
        if np.any(~np.isfinite(vals)):
            return np.nan
        return 0.0
    perms = p.groupby("month", sort=True)[["f1", "f2"]].agg(tri_any)
    coverage = coverage.join(perms)
    coverage["f1_unknown_country_count"] = by["f1"].apply(lambda x: int(x.isna().sum()))
    coverage["f2_unknown_country_count"] = by["f2"].apply(lambda x: int(x.isna().sum()))
    coverage.loc[~complete, ["f1", "f2"]] = np.nan
    return coverage.reset_index().rename(columns={"month": "signal_reference_month"})


# ---------------------------------------------------------------------------
# risk model (one-sided, DEV vintage)
# ---------------------------------------------------------------------------
def dev_vintage_garch(dev_returns: np.ndarray) -> list:
    return [garch11_fit(dev_returns[:, j]) for j in range(dev_returns.shape[1])]


def one_sided_covs(all_returns: np.ndarray, params: list, *, a=0.03, b=0.95,
                   warm: int = 52) -> np.ndarray:
    """Causal GARCH-DCC covariance using DEV-fitted parameters."""
    return dcc_cov(all_returns, params, a=a, b=b, min_obs=warm)


def tail_scenario_ess(w, Sigma, *, alpha=0.95, n_scen=20000, seed=363) -> dict:
    """Kish ESS of actual tail scenarios, NOT ESS of five asset contributions.

    For equal-probability Monte Carlo scenarios q_i=1 for each selected tail
    scenario; ESS=(sum q)^2/sum(q^2)=n_tail.  Unequal validated importance
    weights can be supplied to ``weighted_tail_ess`` below.
    """
    if not 0 < alpha < 1 or n_scen < 2:
        raise ValueError("INVALID_TAIL_GRID")
    rng = np.random.default_rng(seed)
    w = np.asarray(w, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    if Sigma.shape != (len(w), len(w)) or not np.isfinite(Sigma).all():
        raise ValueError("INVALID_COVARIANCE")
    eig, V = np.linalg.eigh((Sigma + Sigma.T) / 2)
    if eig.min() < -1e-9:
        raise ValueError("NON_PSD_COVARIANCE")
    transform = V @ np.diag(np.sqrt(np.maximum(eig, 0)))
    simulated = rng.standard_normal((n_scen, len(w))) @ transform.T
    loss = -(simulated @ w)
    n_tail = max(1, int(np.ceil((1 - alpha) * n_scen - 1e-10)))
    tail_indices = np.argpartition(loss, -n_tail)[-n_tail:]
    q = np.ones(len(tail_indices), dtype=float)
    return {"ESS_tail": weighted_tail_ess(q), "n_scen": int(n_scen),
            "n_tail": int(n_tail), "weight_rule": "UNIFORM_SCENARIO_PROBABILITIES",
            "quantile_loss": float(np.min(loss[tail_indices])),
            "mean_tail_loss": float(np.mean(loss[tail_indices]))}


def weighted_tail_ess(q):
    """ESS=(sum_i q_i)^2 / sum_i q_i^2; q over TAIL SCENARIOS."""
    q = np.asarray(q, dtype=float)
    if q.ndim != 1 or not len(q) or not np.isfinite(q).all() or np.any(q < 0):
        raise ValueError("INVALID_TAIL_SCENARIO_WEIGHTS")
    mass = q.sum()
    if mass <= 0:
        raise ValueError("TAIL_SCENARIO_MASS_ZERO")
    ess = float(mass * mass / np.dot(q, q))
    assert 1 - 1e-8 <= ess <= len(q) + 1e-8
    return ess


def var_es(w, Sigma, alpha, dist="normal", nu=5.0):
    sd = math.sqrt(max(float(w @ Sigma @ w), 1e-16))
    if dist == "normal":
        q = norm.ppf(1 - alpha)
        return float(-q * sd), float(sd * norm.pdf(q) / (1 - alpha))
    from scipy.stats import t as st
    q = st.ppf(1 - alpha, nu)
    scale = math.sqrt((nu - 2) / nu) if nu > 2 else 1.0
    return float(-q * sd * scale), float(sd * scale * st.pdf(q, nu) * (nu + q * q) / ((nu - 1) * (1 - alpha)))


# ---------------------------------------------------------------------------
# feasibility projection (P0-07)
# ---------------------------------------------------------------------------
def feasible_project(w_target: np.ndarray, Sigma: np.ndarray, regime: str, *, risk_budget=RISK_BUDGET) -> dict:
    """Project the target onto the admissible set *before* trading.

    Constraints: long-only, sum(w) <= c_max, per-asset cap, ES95 budget.
    Returns the repaired weight and status; never silently reports a trade that
    violates the risk limits.
    """
    k = 5
    # Explicit benchmark-only unknown-regime research domain. Never infer N
    # from U and never loosen the existing Normal maximum risky capital (75%).
    if regime not in (*REGIMES, "U"):
        raise ValueError("UNRECOGNIZED_REGIME_IN_PROJECTION")
    lo, hi = (0.50, 0.75) if regime == "N" else (0.0, 0.50)
    w0 = np.clip(np.asarray(w_target, float), 0.0, None)
    if not np.isfinite(w0).all():
        raise ValueError("NONFINITE_TARGET_WEIGHTS")
    # Never normalize away the actual POLICY cash/risk-capital target.
    # If the requested target is all-cash under a mandated lower bound,
    # the optimizer must expose the capital conflict, not invent 100% risk.

    def es_of(w):
        return var_es(w, Sigma, 0.95)[1]

    def obj(w):
        return float(np.sum((w - w0) ** 2))

    cons = [{"type": "ineq", "fun": lambda w: hi - w.sum()},
            {"type": "ineq", "fun": lambda w: w.sum() - lo},
            {"type": "ineq", "fun": lambda w: risk_budget - es_of(w)}]
    bnds = [(0.0, PER_ASSET_CAP)] * k
    r = minimize(obj, w0, method="SLSQP", bounds=bnds, constraints=cons,
                 options={"maxiter": 400, "ftol": 1e-10})
    w = np.clip(r.x, 0.0, None)
    ok = bool(np.isfinite(w).all() and w.sum() >= lo - 1e-6 and w.sum() <= hi + 1e-6
              and es_of(w) <= risk_budget + 1e-9 and np.all(w <= PER_ASSET_CAP + 1e-9))
    # The bands are RISKY-CAPITAL SHARES (K_N / K_S), the 60% cap binds each risky asset only (C1 is the residual),
    # and the 2% ES95 budget is an ASSUMED_RESEARCH_ONLY number, not an owner cap.
    envelope = "K_N" if regime == "N" else ("K_S" if regime in REGIMES else "K_S_UNKNOWN_REGIME_BENCHMARK_RESEARCH_ONLY")
    if not ok:
        return {"weights": w0 * 0.0, "feasible": False, "status": "INFEASIBLE_HOLD",
                "es95": es_of(w0), "sum": 0.0, "envelope": envelope, "risk_budget_status": "ASSUMED_RESEARCH_ONLY"}
    return {"weights": w, "feasible": True, "status": "FEASIBLE_PROJECTED",
            "es95": es_of(w), "sum": float(w.sum()), "envelope": envelope,
            "risk_budget_status": "ASSUMED_RESEARCH_ONLY"}


def policy_weights(Sigma, regime, permission, bottom_assets=None) -> np.ndarray:
    base = w_erc(Sigma)
    if regime == "N":
        risky = base
    else:
        bond = np.array([0.0, 0.0, 0.0, 0.5, 0.5])
        risky = 0.5 * base + 0.5 * bond
    c = {"N": 0.75, "D": 0.50, "G": 0.35, "BOTH": 0.25}.get(regime, 0.50)
    long_target = c * risky
    # W03 bottom-version challenger: if the three-family permission is issued
    # and specific equity assets show a prior-week drawdown/recovery pattern,
    # move up to 15% of risky capital from bonds into those equities.
    # This is a registered research hypothesis, NOT an owner-approved rule.
    if permission and regime != "N" and bottom_assets is not None:
        elig = np.asarray(bottom_assets[:3], bool)
        if elig.any():
            amount = min(0.15 * c, float(long_target[3:5].sum()))
            bond_base = float(long_target[3:5].sum())
            if bond_base > 0:
                long_target[3:5] *= (bond_base - amount) / bond_base
                long_target[:3] += amount * elig / elig.sum()
    return long_target


def target_weights(b, Sigma, regime, permission, bottom_assets=None) -> np.ndarray:
    if b == "EW":
        return w_ew(Sigma)
    if b == "IVOL":
        return w_ivol(Sigma)
    if b == "MVP":
        return w_mvp(Sigma)
    if b == "ERC":
        return w_erc(Sigma)
    if b == "MaxDiv":
        return w_maxdiv(Sigma)
    return policy_weights(Sigma, regime, permission, bottom_assets)


# ---------------------------------------------------------------------------
# continuous stateful backtest (P0-02, P0-07)
# ---------------------------------------------------------------------------
def stateful_backtest(tape: pd.DataFrame, covs: np.ndarray, regimes, permission, *,
                      fee=FEE_RESEARCH, short_overlay=True, risk_budget=RISK_BUDGET,
                      strategies=BENCHMARKS, bottom_asset_flags=None,
                      permission_known=None, missing_benchmark_regime="HOLD") -> dict:
    """Causal one-week ledger with drift, executed orders, costs and no reset.

    Pre-week order is a function of prior returns, pre-published signals and
    Sigma[t] only.  Same-week return R[t] appears strictly after orders.
    Holdings state is six signed asset/cash weights summing to one, measured
    just before each week's rebalance (post preceding week's market drift).
    Fee is 10bp research assumption times half-L1 self-financed turnover.

    Hard envelope (spec RQ03 risk tolerance 1): long-only, risky share in [0.50, 0.75] (Normal) or
    [0, 0.50] (Stress), each risky asset <= 60%.  The legacy default `short_overlay=True` (kept because
    test_w03_accumulated_r2 pins it) subtracts an equity basket from POLICY's weights in Stress without a Bottom
    permission, which can leave a net SHORT equity position, i.e. lie OUTSIDE that long-only envelope.  Such
    weeks are labelled TRADE_FEASIBLE_SHORT_OVERLAY_RESEARCH and `envelope_ok` records whether the executed
    weights are actually inside K_N/K_S (0 = outside).  Callers that need envelope-conformant books pass
    short_overlay=False.  The 2% ES95 budget and the 10 bp fee are ASSUMED_RESEARCH_ONLY numbers.
    """
    if missing_benchmark_regime not in ("HOLD", "RISK_ONLY_CAP50"):
        raise ValueError("UNSUPPORTED_BENCHMARK_UNKNOWN_REGIME_POLICY")
    from research_adapters.rq03 import risk_tolerance as _RT   # single source of the K_N / K_S envelope
    R = tape[list(RISKY_ASSETS)].to_numpy(dtype=float)
    cash = tape["C1"].to_numpy(dtype=float)
    # Pure input-contract validation, never a signal that turns orders on/off
    # using the unknown actual return of the current trading week.
    if not np.isfinite(R).all() or not np.isfinite(cash).all():
        raise ValueError("MISSING_WEEKLY_MARK_TO_MARKET_RETURN_INPUT_CONTRACT")
    n = len(tape)
    if permission_known is None:
        # Legacy callers have no provenance tape: preserve API while explicitly
        # reserving authoritative gating for run() which supplies the flags.
        permission_known = np.ones(n, dtype=bool)
    else:
        permission_known = np.asarray(permission_known, dtype=bool)
        if permission_known.shape != (n,):
            raise ValueError("PERMISSION_KNOWN_LENGTH_MISMATCH")
    k = len(RISKY_ASSETS)
    short_basket = np.array([1 / 3, 1 / 3, 1 / 3, 0, 0], dtype=float)
    books = {}
    for b in strategies:
        books[b] = {"nav": np.ones(n), "weights": np.zeros((n,k)),
                    "pretrade_weights": np.zeros((n,k)),
                    "net_weights": np.zeros((n,k)),
                    "cash_weight": np.ones(n), "pretrade_cash": np.ones(n),
                    "turnover": np.zeros(n), "cost": np.zeros(n),
                    "short": np.zeros(n), "permission_known": np.zeros(n, int),
                    "action": [""] * n,
                    "status": [""] * n, "feasible": np.zeros(n, int),
                    "risk_es95": np.full(n, np.nan),
                    "gross_return": np.zeros(n), "net_return": np.zeros(n),
                    "traded": np.zeros(n, int),
                    "projected_weights": np.zeros((n,k)),
                    # -1 = not evaluated, 1 = executed weights inside K_N/K_S, 0 = outside (e.g. emergency cash in
                    # Normal, short overlay).  ES on OPENING NAV *including the fee* (C + ES(X)); the budget test
                    # above stays pre-fee because the fee is only known after the order is sized.
                    "envelope_ok": np.full(n, -1, int),
                    "risk_es95_opening_nav": np.full(n, np.nan),
                    "risk_budget_status": "ASSUMED_RESEARCH_ONLY_ES95_2PCT_PRE_FEE_PROJECTION"}
    held = {b: np.r_[np.zeros(k), 1.] for b in BENCHMARKS}
    for t in range(n):
        S = np.asarray(covs[t], dtype=float)
        has_valid_regime = regimes[t] in REGIMES
        risk_ready = bool(np.isfinite(S).all() and
                          np.linalg.eigvalsh((S+S.T)/2).min() >= -1e-9)
        for b in strategies:
            # Only five benchmarks may use this explicit experimental U mode;
            # POLICY must abstain and R3 behavior remains the default.
            benchmark_unknown_research = bool(
                b != "POLICY" and not has_valid_regime and
                missing_benchmark_regime == "RISK_ONLY_CAP50")
            ready = risk_ready and (has_valid_regime or benchmark_unknown_research)
            pre = held[b].copy()  # includes t-1 closing drift, NO t return
            rec = books[b]
            rec["pretrade_weights"][t] = pre[:k]
            rec["pretrade_cash"][t] = pre[-1]
            short = 0.
            if not ready:
                executed = pre.copy()
                status = ("NO_TRADE_MISSING_ISSUED_REGIME" if not has_valid_regime else
                          "NO_TRADE_WARMUP_OR_INVALID_RISK")
                feasible = False
                projected = pre[:k].clip(0, None)
            else:
                this_bottom = (bottom_asset_flags[t] if bottom_asset_flags is not None else None)
                target = target_weights(b, S, regimes[t], permission[t], this_bottom)
                projection_regime = "U" if benchmark_unknown_research else regimes[t]
                # Recompute per call: upstream R4 cache could be reused with
                # different covariance vintages/targets and change decisions.
                projection = feasible_project(target, S, projection_regime,
                                              risk_budget=risk_budget)
                projected = projection["weights"]
                feasible = bool(projection["feasible"])
                if feasible:
                    if b == "POLICY" and short_overlay and regimes[t] in ("D", "G", "BOTH") \
                            and not bool(permission[t]) and bool(permission_known[t]):
                        # Short overlay is NOT owner-authorized, so report research only.
                        # Check final *net* risk and capital before submitting orders.
                        for candidate in (0.20, 0.10, 0.05):
                            net_candidate = projected - candidate * short_basket
                            if var_es(net_candidate, S, 0.95)[1] <= risk_budget + 1e-9:
                                short = candidate
                                break
                    net = projected - short * short_basket
                    executed = np.r_[net, 1. - net.sum()]
                    status = ("TRADE_FEASIBLE_BENCHMARK_U_RISK_ONLY_RESEARCH"
                              if benchmark_unknown_research else "TRADE_FEASIBLE")
                    if short > 0:
                        status = "TRADE_FEASIBLE_SHORT_OVERLAY_RESEARCH"
                else:
                    # An unfeasible target is NEVER secretly marked HOLD after
                    # zeroing weights.  Liquidation is separately identifiable.
                    executed = np.r_[np.zeros(k), 1.]
                    status = "EMERGENCY_LIQUIDATE_RESEARCH_ONLY"
            assert np.isfinite(executed).all() and np.isclose(executed.sum(), 1)
            assert executed[-1] >= -1e-8, "UNFUNDED_CASH_POSITION"
            # Half of complete signed-weight L1; includes balancing cash leg.
            turn = float(0.5 * np.abs(executed - pre).sum())
            cost = float(fee * turn)
            if turn <= 1e-10:
                action = ("ABSTAIN" if status.startswith("EMERGENCY") else
                          "NO_TRADE" if not ready else "HOLD")
            elif status.startswith("EMERGENCY"):
                action = "EMERGENCY_LIQUIDATE"
            else:
                action = "PARTIAL" if turn < 0.10 else "FULL"
            if status.startswith("EMERGENCY"):
                assert action in ("EMERGENCY_LIQUIDATE", "ABSTAIN")
            # Fee paid from cash at execution; weights after fees relative to
            # PRE-fee wealth and net market drift calculated exactly once.
            units = executed.copy()
            units[-1] -= cost
            market = np.r_[1 + R[t], 1 + cash[t]]
            end_units = units * market
            growth = float(end_units.sum())
            if growth <= 0 or not np.isfinite(growth):
                raise ArithmeticError("NONPOSITIVE_NET_NAV")
            nav_prev = rec["nav"][t-1] if t else 1.0
            rec["nav"][t] = nav_prev * growth
            held[b] = end_units / growth
            rec["weights"][t] = projected if feasible else np.zeros(k)
            rec["projected_weights"][t] = projected
            rec["net_weights"][t] = executed[:k]
            rec["cash_weight"][t] = executed[-1]
            rec["turnover"][t] = turn
            rec["cost"][t] = cost
            rec["short"][t] = short
            rec["permission_known"][t] = int(permission_known[t])
            rec["action"][t] = action
            rec["status"][t] = status
            rec["feasible"][t] = int(feasible)
            rec["risk_es95"][t] = var_es(executed[:k], S, 0.95)[1] if ready else np.nan
            if ready:
                _env = _RT.in_envelope(executed, projection_regime,
                                       benchmark_unknown_research=benchmark_unknown_research)
                rec["envelope_ok"][t] = int(_env["status"] == "PASS") if _env["status"] != "U" else -1
                rec["risk_es95_opening_nav"][t] = cost + rec["risk_es95"][t]    # L = C - X'r  (zero-mean Gaussian)
            rec["gross_return"][t] = float(executed @ (market - 1))
            rec["net_return"][t] = growth - 1
            rec["traded"][t] = int(turn > 1e-10)
            if ready and feasible:
                assert np.all(projected >= -1e-7)
                assert np.all(projected <= PER_ASSET_CAP + 1e-7)
                assert rec["risk_es95"][t] <= risk_budget + 1e-8
            if not ready:
                assert turn <= 1e-10, "WARMUP_TRADE_FORBIDDEN"
    return books


def performance(nav, turnover, cash_returns=None) -> dict:
    """Calendar-week net NAV diagnostics, optional mathematically defined Sharpe.

    NAV[0] is wealth *after* the first of n weekly returns, not the initial W0.
    For an excess-return Sharpe, require matched C1 cash weekly returns;
    without them `sharpe` is undefined rather than annual CAGR/vol mislabeled.
    """
    nav = np.asarray(nav, float)
    turn = np.asarray(turnover, float)
    if nav.ndim != 1 or not len(nav) or not np.isfinite(nav).all() or np.any(nav <= 0):
        raise ValueError("INVALID_NET_NAV")
    if len(turn) != len(nav):
        raise ValueError("TURNOVER_NAV_LENGTH_MISMATCH")
    week_ret = np.r_[nav[0] - 1., nav[1:] / nav[:-1] - 1.]
    mdd = float(np.min(nav / np.maximum.accumulate(np.r_[1., nav]) [1:] - 1.))
    ann = float(nav[-1] ** (52.0 / len(nav)) - 1.0)
    vol = float(np.std(week_ret, ddof=1) * math.sqrt(52)) if len(week_ret) > 1 else None
    sharpe = None
    if cash_returns is not None:
        cash_returns = np.asarray(cash_returns, float)
        if cash_returns.shape != week_ret.shape or not np.isfinite(cash_returns).all():
            raise ValueError("CASH_RETURN_CLOCK_MISMATCH")
        excess = week_ret - cash_returns
        ex_vol = float(np.std(excess, ddof=1)) if len(excess) > 1 else None
        sharpe = float(np.mean(excess) / ex_vol * math.sqrt(52)) if ex_vol and ex_vol > 0 else None
    return {"terminal_nav": float(nav[-1]), "MDD": mdd, "annual_return": ann,
            "annual_vol": vol, "sharpe": sharpe,
            "sharpe_status": "NET_EXCESS_OVER_C1" if cash_returns is not None else "UNDEFINED_NO_CASH_TAPE",
            "avg_turnover": float(np.mean(turn)), "total_turnover": float(np.sum(turn))}





def bottom_asset_indicators(tape: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Asset Bottom conditions use strictly completed prior weeks; no current return gate."""
    # Asset-specific Bottom is computed only from completed PRIOR weeks.
    # First construct NAV-index levels from weekly simple returns; this is
    # NOT a new external price input. A prior drawdown plus prior recovery
    # and prior volatility cooling must all be present for asset-specific buy.
    bottom_assets = np.zeros((len(tape), 5), dtype=bool)
    for j, asset in enumerate(("E1", "E2", "E3")):
        path = (1. + tape[asset].astype(float)).cumprod().shift(1)
        dd = path / path.rolling(52, min_periods=52).max() - 1.
        recovery = ((1. + tape[asset].shift(1)).rolling(4).apply(np.prod, raw=True) - 1.) > 0.
        cooling = (tape[asset].shift(1).rolling(13).std() <
                   tape[asset].shift(1).rolling(52).std())
        bottom_assets[:, j] = ((dd <= -0.10) & recovery & cooling).fillna(False).to_numpy()
        tape[f"bottom_{asset}"] = bottom_assets[:, j].astype(int)

    return tape, bottom_assets

def weekly_research_decision_inputs(tape: pd.DataFrame, panel: pd.DataFrame, *, rq01_thr=0.5945) -> pd.DataFrame:
    """Single data-first W03 signal adapter; no external certification to RUN research.

    This is a historical research proxy and keeps temporal as-of filtering.
    Incomplete evidence is `U`, not assumed normal or rejected as a data error.
    Executability and issuer provenance remain distinct certification gates.
    """
    # monthly regime/permission
    rf = regime_frame(panel, None, rq01_thr)
    original_week_end = tape["week_end"].to_numpy(copy=True)
    clock = tape.copy()
    # Create a canonical month field even for the native DEV tape (no month column).
    # It describes the target week, never an input signal vintage.
    clock["month"] = pd.to_datetime(clock["week_end"]).dt.strftime("%Y-%m")
    clock["week_end"] = pd.to_datetime(clock["week_start"]) - pd.Timedelta(nanoseconds=1)
    # Upstream month is a research-effective date, NOT an authenticated issuer stamp.
    # Explicitly track the sourced month and freshness in the weekly tape.
    monthly = rf.rename(columns={"signal_reference_month": "month"})
    tape = asof_week_join(clock, monthly, lag_days=0)
    tape["week_end"] = original_week_end
    tape["signal_reference_month"] = tape["month_y"]
    source_end = pd.to_datetime(tape["signal_reference_month"].astype("string") + "-01",
                                errors="coerce") + pd.offsets.MonthEnd(0)
    tape["signal_age_days"] = (pd.to_datetime(tape["week_start"]) - source_end).dt.days
    too_old = tape["signal_age_days"].isna() | (tape["signal_age_days"] > 62)
    upstream_ok = tape["signal_status"].eq("RESEARCH_PROXY_COMPLETE_EA7") & ~too_old
    tape["signal_status"] = np.where(upstream_ok, "RESEARCH_PROXY_EA7_ASOF_UNVERIFIED_PIT",
                                      "U_MISSING_INCOMPLETE_OR_STALE_SIGNAL")
    tape["month"] = tape["week_end"].dt.strftime("%Y-%m")
    tape["regime_exante"] = tape["regime_exante"].where(upstream_ok, "U").fillna("U")
    tape["f1"] = pd.to_numeric(tape["f1"], errors="coerce").where(upstream_ok, np.nan)
    tape["f2"] = pd.to_numeric(tape["f2"], errors="coerce").where(upstream_ok, np.nan)
    # permission requires a causal F3 from past weekly returns: 3-family gate
    z = tape.copy()
    # Equity-family cooling must use E1/E2/E3 only, not five risky assets.
    z["eq"] = z[["E1", "E2", "E3"]].mean(axis=1)
    z["rv13"] = z["eq"].shift(1).rolling(13).std()
    z["rv52"] = z["eq"].shift(1).rolling(52).std()
    z["ret4"] = ((1. + z["eq"].shift(1)).rolling(4)
                 .apply(np.prod, raw=True) - 1.)
    z["f3"] = ((z["rv13"] < z["rv52"]) & (z["ret4"] > 0)).astype(int)
    f3_known = z["rv13"].notna() & z["rv52"].notna() & z["ret4"].notna()
    tape["f3"] = z["f3"].where(f3_known, np.nan)
    tape["permission_known"] = (tape["f1"].notna() & tape["f2"].notna() &
                                tape["f3"].notna() & tape["regime_exante"].isin(REGIMES))
    tape["votes"] = tape[["f1", "f2", "f3"]].sum(axis=1, min_count=3)
    tape["permission"] = ((tape["votes"] >= 3) & tape["permission_known"]).astype(int)
    tape["permission_status"] = np.where(tape["permission_known"],
                                         np.where(tape["permission"].eq(1),
                                                  "BOTTOM_ALLOWED_RESEARCH_ONLY",
                                                  "BOTTOM_DENIED_SOURCE_RESOLVED"),
                                         "U_FAMILY_SOURCE_UNRESOLVED")

    return tape

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def run(dataset_root, out_dir, *, rq01_panel=None, rq01_thr=0.5945, budget_fast=False) -> dict:
    from rev363.w01_contract import enforce_execution
    enforce_execution(oos_requested=True)  # full_tape() opens 2020-2024: requires scientific owner amendment
    out = out_dir
    from rev363.data import find_dataset_root
    root = find_dataset_root(dataset_root)
    tape = full_tape(root)
    tape = weekly_research_decision_inputs(tape, rq01_panel, rq01_thr=rq01_thr)

    tape, bottom_assets = bottom_asset_indicators(tape)

    # risk model: DEV vintage params, one-sided filter over the full history
    dev_end = pd.Timestamp("2019-12-31")
    dev_R = tape.loc[tape["week_end"] <= dev_end, list(RISKY_ASSETS)].values
    if len(dev_R) <= 60:
        raise ValueError("NO_DEV_VINTAGE_RISK_FIT")
    params = dev_vintage_garch(dev_R)
    allR = tape[list(RISKY_ASSETS)].values
    covs = one_sided_covs(allR, params, a=0.03, b=0.95, warm=52)

    regimes = tape["regime_exante"].values
    perm = tape["permission"].values
    books = stateful_backtest(tape, covs, regimes, perm, short_overlay=True,
                               bottom_asset_flags=bottom_assets,
                               permission_known=tape["permission_known"].to_numpy())
    books_ns = stateful_backtest(tape, covs, regimes, perm, short_overlay=False,
                                  bottom_asset_flags=bottom_assets, strategies=("POLICY",),
                                  permission_known=tape["permission_known"].to_numpy())

    # --- 03-01 asset tape
    tape[["week_start", "week_end"] + list(ASSETS) + ["month", "regime_exante",
                                                      "signal_reference_month", "signal_age_days",
                                                      "signal_status", "permission", "votes",
                                                      "permission_known", "permission_status"]].to_csv(
        out / "RQ03_01_ASSET_TAPE.csv", index=False)

    # --- 03-09 continuous NAV
    nav = pd.DataFrame({"week_end": tape["week_end"], "month": tape["month"]})
    for b in BENCHMARKS:
        nav[b] = books[b]["nav"]
    nav["regime_exante"] = regimes
    nav.to_csv(out / "RQ03_09_SIX_NAV_NET.csv", index=False)

    # --- 03-02 five+1 performance per segment
    segs = {"DEV_2005_2019": (None, pd.Timestamp("2019-12-31")),
            "OOS_FULL_2020_2024": (pd.Timestamp("2020-01-01"), None),
            "OOS_TAIL_2021_05_2024_12": (pd.Timestamp("2021-05-01"), None)}
    perf_rows = []
    for b in BENCHMARKS:
        nv = pd.Series(books[b]["nav"], index=tape["week_end"].values)
        row = {"strategy": b}
        for label, (lo, hi) in segs.items():
            m = np.ones(len(tape), bool)
            if lo is not None:
                m &= (tape["week_end"] >= lo).values
            if hi is not None:
                m &= (tape["week_end"] <= hi).values
            sub = nv[m]
            ix = np.where(m)[0]
            base = (float(nv.iloc[ix[0] - 1]) if len(ix) and ix[0] > 0 else 1.0)
            row[f"W_{label}_start"] = base
            row[f"W_{label}_end"] = float(sub.iloc[-1]) if len(sub) else None
            row[f"ret_{label}"] = (float(sub.iloc[-1] / base - 1.0) if len(sub) else None)
        row.update({f"full_{k}": v for k, v in performance(
            books[b]["nav"], books[b]["turnover"], tape["C1"].to_numpy(dtype=float)).items()})
        row["total_fees"] = float(np.sum(books[b]["cost"]))
        row["n_hold"] = int(sum(1 for a in books[b]["action"] if a == "HOLD"))
        row["n_partial"] = int(sum(1 for a in books[b]["action"] if a == "PARTIAL"))
        row["n_full"] = int(sum(1 for a in books[b]["action"] if a == "FULL"))
        row["n_infeasible_hold"] = 0
        row["n_emergency_liquidate"] = int(sum(1 for a in books[b]["action"] if a == "EMERGENCY_LIQUIDATE"))
        row["avg_risky"] = float(np.mean([w.sum() for w in books[b]["weights"]]))
        row["avg_cash"] = float(np.mean(books[b]["cash_weight"]))
        perf_rows.append(row)
    perf = pd.DataFrame(perf_rows)
    perf.to_csv(out / "RQ03_02_PERFORMANCE_CONTINUOUS.csv", index=False)

    # --- 03-03 GARCH-DCC / ESS
    ess_rows = []
    for t in range(len(tape)):
        S = covs[t]
        if not np.isfinite(S).all():
            continue
        w = books["POLICY"]["net_weights"][t]
        if w.sum() <= 0:
            continue
        e = tail_scenario_ess(w, S, alpha=0.95, n_scen=5000)
        ess_rows.append({"week_end": str(tape.loc[t, "week_end"].date()), "strategy": "POLICY",
                         **e})
    pd.DataFrame(ess_rows).to_csv(out / "RQ03_03_ESS_TAIL.csv", index=False)

    # --- 03-04 VaR/ES per strategy x regime x alpha
    rows = []
    for b in BENCHMARKS:
        w = books[b]["net_weights"]
        nav_b = books[b]["nav"]
        ret = books[b]["net_return"]
        for regime in REGIMES:
            for alpha in ALPHAS:
                vr, es, br = [], [], []
                for t in np.where(regimes == regime)[0]:
                    if not np.isfinite(covs[t]).all():
                        continue
                    ww = w[t]
                    v, e = var_es(ww, covs[t], alpha)
                    vr.append(v); es.append(e); br.append(int(ret[t] < -v))
                rows.append({"strategy": b, "regime": regime, "alpha": alpha,
                             "n_weeks": len(vr), "VaR": float(np.mean(vr)) if vr else None,
                             "ES": float(np.mean(es)) if es else None,
                             "breach_rate": float(np.mean(br)) if br else None,
                             "status": "EXECUTED" if vr else "U_INSUFFICIENT_REGIME_SUPPORT"})
    pd.DataFrame(rows).to_csv(out / "RQ03_04_VAR_ES_6.csv", index=False)

    # --- 03-05 strategy-specific actual feasibility, NOT ERC-only proxy.
    cap_rows = []
    for b in BENCHMARKS:
        bk = books[b]
        for t in range(len(tape)):
            lo, hi = (0.50, 0.75) if regimes[t] == "N" else (0.0, 0.50)   # K_N / K_S risky-capital bands
            eligible = bool(np.isfinite(covs[t]).all())
            exec_net = bk["net_weights"][t]
            long_projected = bk["projected_weights"][t]
            cap_rows.append({"week_end": str(tape.loc[t,"week_end"].date()),
                "strategy": b, "regime": regimes[t],
                "capital_floor": lo, "capital_ceiling": hi,
                "projected_risky_capital": float(long_projected.sum()),
                "net_risky_capital": float(exec_net.sum()),
                "pretrade_ES95_net_risky": float(bk["risk_es95"][t]) if eligible else None,
                "cost": float(bk["cost"][t]),
                "turnover": float(bk["turnover"][t]),
                "feasible": int(bk["feasible"][t]) if eligible else None,
                "status": bk["status"][t]})
    cap = pd.DataFrame(cap_rows)
    cap.to_csv(out / "RQ03_05_CAPITAL_FEASIBILITY.csv", index=False)

    # --- 03-06 action ledger (weights, turnover, fee) for six strategies
    wr = []
    for b in BENCHMARKS:
        for t in range(len(tape)):
            wr.append({"week_end": str(tape.loc[t, "week_end"].date()), "strategy": b,
                       **{f"w_{a}": float(books[b]["weights"][t, j])
                          for j, a in enumerate(RISKY_ASSETS)},
                       "cash_weight": float(books[b]["cash_weight"][t]),
                       "short_weight": float(books[b]["short"][t]),
                       "turnover": float(books[b]["turnover"][t]),
                       "cost": float(books[b]["cost"][t]),
                       "pretrade_cash": float(books[b]["pretrade_cash"][t]),
                       "gross_return": float(books[b]["gross_return"][t]),
                       "net_return": float(books[b]["net_return"][t]),
                       "pretrade_es95": float(books[b]["risk_es95"][t]),
                       "order_status": books[b]["status"][t],
                       "signal_status": tape.loc[t,"signal_status"],
                       "signal_reference_month": tape.loc[t,"signal_reference_month"],
                       "signal_age_days": float(tape.loc[t,"signal_age_days"]),
                       "bottom_permission": int(tape.loc[t,"permission"]),
                       "permission_known": int(tape.loc[t,"permission_known"]),
                       "permission_status": tape.loc[t,"permission_status"],
                       "risk_vintage": ("DEV_FULL_BLOCK_FIT_EX_POST_NOT_HISTORIC_PIT"
                                        if tape.loc[t,"week_end"] <= pd.Timestamp("2019-12-31")
                                        else "FROZEN_DEV_END_2019_FILTERED_ONE_SIDED"),
                       "bottom_asset_count": int(bottom_assets[t,:3].sum()),
                       **{f"net_{a}": float(books[b]["net_weights"][t,j]) for j,a in enumerate(RISKY_ASSETS)},
                       **{f"pre_{a}": float(books[b]["pretrade_weights"][t,j]) for j,a in enumerate(RISKY_ASSETS)},
                       "feasible": int(books[b]["feasible"][t]),
                       "action": books[b]["action"][t]})
    pd.DataFrame(wr).to_csv(out / "RQ03_06_ACTION_LEDGER.csv", index=False)

    # PRE-TRADE VaR/ES risk forecast uses signed executed holdings, known fee
    # and the lagged cash proxy; not raw net NAV vs risky-only VaR.
    from rev363.w03_evaluation import var_tape, aggregate_risk
    weekly_risk = var_tape(tape, pd.DataFrame(wr), params)
    weekly_risk.to_csv(out / "RQ03_04_VAR_ES_ISSUED_WEEKLY.csv", index=False)
    aggregate_risk(weekly_risk).to_csv(out / "RQ03_04_VAR_ES_AGGREGATE_CORRECTED.csv", index=False)
    # Preserve legacy name but replace its formerly mismatched risk accounting.
    aggregate_risk(weekly_risk).to_csv(out / "RQ03_04_VAR_ES_6.csv", index=False)

    # --- 03-07 bottom permission tape
    tape[["week_end", "month", "regime_exante", "f1", "f2", "f3", "votes", "permission",
          "permission_known", "permission_status",
          "bottom_E1", "bottom_E2", "bottom_E3"]].to_csv(
        out / "RQ03_07_BOTTOM_PERMISSION.csv", index=False)

    # --- 03-08 short -> bottom avoided loss
    eq_ret = tape[["E1", "E2", "E3"]].mean(axis=1).values
    sb = []
    for t in range(len(tape)):
        sh = books["POLICY"]["short"][t]
        if sh > 0:
            sb.append({"week_end": str(tape.loc[t, "week_end"].date()),
                       "short_notional": float(sh), "equity_return": float(eq_ret[t]),
                       "short_week_pnl_before_fees": float(sh * (-eq_ret[t] + tape.loc[t, "C1"])),
                       "matched_terminal_delta_net": float(books["POLICY"]["nav"][-1]
                                                           - books_ns["POLICY"]["nav"][-1])})
    sb = pd.DataFrame(sb)
    sb.to_csv(out / "RQ03_08_SHORT_TO_BOTTOM.csv", index=False)

    # --- 03-10 robustness (DCC vs EWMA, normal vs t)
    from rev363.rq03 import ewma_cov
    cov_ewma = ewma_cov(allR)
    rob = []
    for alpha in ALPHAS:
        for b in ("POLICY", "EW"):
            for name, cv in (("DCC", covs), ("EWMA", cov_ewma)):
                for dist in ("normal", "t"):
                    vals = []
                    for t in range(len(tape)):
                        if not np.isfinite(cv[t]).all():
                            continue
                        ww = books[b]["weights"][t]
                        if ww.sum() <= 0:
                            continue
                        v, e = var_es(ww, cv[t], alpha, dist=dist)
                        vals.append((v, e))
                    rob.append({"model": name, "distribution": dist, "strategy": b, "alpha": alpha,
                                "mean_VaR": float(np.mean([x[0] for x in vals])) if vals else None,
                                "mean_ES": float(np.mean([x[1] for x in vals])) if vals else None})
    pd.DataFrame(rob).to_csv(out / "RQ03_10_ROBUSTNESS.csv", index=False)

    # --- 03-11 OOS tail table (2021-05..2024-12) + full OOS
    tail_rows = []
    for b in BENCHMARKS:
        nv = pd.Series(books[b]["nav"], index=tape["week_end"].values)
        m_tail = (tape["week_end"] >= pd.Timestamp("2021-05-01")).values
        m_full = (tape["week_end"] >= pd.Timestamp("2020-01-01")).values
        m_dev = (tape["week_end"] <= pd.Timestamp("2019-12-31")).values
        sub_t = nv[m_tail]; sub_f = nv[m_full]
        tail_rows.append({
            "strategy": b,
            "W0_2005": 1.0,
            "W_DEV_end": float(nv[m_dev].iloc[-1]),
            "W_2021_04_end": float(nv[nv.index <= pd.Timestamp("2021-04-30")].iloc[-1]),
            "W_2024_end": float(nv.iloc[-1]),
            "return_tail_2021_05_2024_12": float(sub_t.iloc[-1] / nv.iloc[np.where(m_tail)[0][0]-1] - 1.0),
            "full_OOS_return_2020_2024": float(sub_f.iloc[-1] / nv.iloc[np.where(m_full)[0][0]-1] - 1.0),
            "full_continuous_terminal_wealth": float(nv.iloc[-1]),
            "MDD_tail": float(np.min(sub_t / np.maximum.accumulate(sub_t) - 1.0)),
            "MDD_full_OOS": float(np.min(sub_f / np.maximum.accumulate(sub_f) - 1.0)),
            "n_trades": int((books[b]["turnover"] > 1e-9).sum()),
            "turnover": float(np.sum(books[b]["turnover"])),
            "total_fees": float(np.sum(books[b]["cost"])),
            "avg_risky": float(np.mean([w.sum() for w in books[b]["weights"]])),
            "avg_cash": float(np.mean(books[b]["cash_weight"])),
            "n_infeasible_hold": 0,
            "n_emergency_liquidate": int(sum(a == "EMERGENCY_LIQUIDATE" for a in books[b]["action"])),
        })
    tail = pd.DataFrame(tail_rows)
    tail.to_csv(out / "RQ03_11_OOS_TAIL_TABLE.csv", index=False)

    pol_nav = pd.Series(books["POLICY"]["nav"])
    wk = tape["week_end"].values
    continue_check = {
        "W_2019_last": float(pol_nav[(wk <= np.datetime64("2019-12-31"))].iloc[-1]),
        "W_2020_first": float(pol_nav[(wk >= np.datetime64("2020-01-01"))].iloc[0]),
        "n_weeks_total": len(tape),
        "nav_reset_performed": False,
    }
    results = {
        "run_id": run_id("REV_NEXT_RQ03"), "created_utc": utcnow(),
        "n_weeks": len(tape), "assets": list(ASSETS),
        "risk_model": {"type": "GARCH11_DCC_DEV_VINTAGE_ONE_SIDED", "garch_params": params,
                       "DEV_risk_fit_lookahead": "FULL_DEV_PARAMETER_FIT_EX_POST_NOT_EXECUTABLE_DEV",
                       "OOS_risk_fit_origin": "FROZEN_DEV_END_2019_POST_EXPOSURE_EXPLORATORY",
                       "dcc_a": 0.03, "dcc_b": 0.95,
                       "status": "RESEARCH_ASOF_NOT_OWNER_CERTIFIED"},
        "continuity": continue_check,
        "performance": perf_rows,
        "tail_table": tail_rows,
        "capital": {"n_evaluated_strategy_weeks": int(cap["feasible"].notna().sum()),
                    "n_feasible_strategy_weeks": int((cap["feasible"] == 1).sum()),
                    "n_blocked_strategy_weeks": int((cap["feasible"] == 0).sum()),
                    "n_warmup_strategy_weeks": int(cap["feasible"].isna().sum())},
        "short_bottom": {"n_weeks_short": len(sb),
                         "net_short_terminal_delta_matched": float(books["POLICY"]["nav"][-1] - books_ns["POLICY"]["nav"][-1]),
                         "policy_terminal_with_short": float(books["POLICY"]["nav"][-1]),
                         "policy_terminal_without_short": float(books_ns["POLICY"]["nav"][-1])},
    }
    jdump(results, out / "RQ03_RESULTS.json")
    return results
