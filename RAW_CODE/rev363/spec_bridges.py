"""Bridges from the integrated REV363 pipeline to the canonical spec engines.

These adapters do not duplicate the mathematics: they validate inputs, call the
retained spec modules, and return immutable, provenance-labelled records.

* ``build_spec_s5``           -> research_adapters.rq02_s5_buyhold_var_es
* ``build_spec_exit_b``       -> rq02_source.rq02_rev300_exit_math.branch_B
* ``main_lock_week``          -> research_adapters.rq03.main_lock
"""
from __future__ import annotations

import numpy as np
import pandas as pd

S5_ASSETS = ("E1", "E2", "E3")


def build_spec_s5(prices: pd.DataFrame, *, h6, onset, short_votes: pd.DataFrame,
                  method: str = "EWMA_NORMAL") -> dict:
    """Compute the three distinct S5 endpoints on native weekly index levels.

    ``prices`` must contain positive E1/E2/E3 index levels indexed by native
    week-end. ``short_votes`` must contain ``issue`` and binary ``vote``.
    Retrospective research output only; never an H6 feature.
    """
    from research_adapters import rq02_s5_buyhold_var_es as S5

    if method != "EWMA_NORMAL":
        raise ValueError("S5_METHOD_NOT_IMPLEMENTED:" + str(method))
    frame = prices[list(S5_ASSETS)].astype(float).sort_index()
    if not frame.index.is_monotonic_increasing or frame.isna().any().any() or (frame <= 0).any().any():
        raise ValueError("S5_POSITIVE_SORTED_INDEX_LEVELS_REQUIRED")
    h6 = pd.Timestamp(h6)
    onset = pd.Timestamp(onset)
    if h6 not in frame.index:
        raise ValueError("S5_H6_NOT_A_NATIVE_WEEK")
    if not h6 < onset:
        raise ValueError("S5_ONSET_MUST_FOLLOW_H6")
    required = {"issue", "vote"}
    if not required.issubset(short_votes.columns):
        raise ValueError("S5_SHORT_VOTES_SCHEMA_REQUIRED")

    returns = frame.pct_change().dropna()
    history = returns.loc[:h6]
    if len(history) < 53:
        raise ValueError("S5_NEEDS_52_PRIOR_WEEKLY_RETURNS")
    cov = history.tail(52).cov().to_numpy()
    weights = S5.initial_weights(cov)

    risk_frames = []
    for portfolio, w0 in weights.items():
        nav, drift, _shares = S5.buy_hold(frame, h6, w0)
        risk = S5.risk_table_ewma_normal(returns, drift.iloc[1:], cov)
        risk["portfolio"] = portfolio
        risk_frames.append(risk)
    all_risk = pd.concat(risk_frames, ignore_index=True)
    all_risk = all_risk[(all_risk["date"] > h6) & (all_risk["date"] <= onset)]
    if all_risk.empty:
        raise ValueError("S5_NO_RISK_OBSERVATION_IN_WINDOW")
    result = S5.s5_endpoints(all_risk, short_votes, h6=h6, onset=onset)
    result.update({
        "method": method,
        "n_risk_rows": int(len(all_risk)),
        "portfolio_weights": {k: v.tolist() for k, v in weights.items()},
        "role": "RETROSPECTIVE_NOT_A_H6_FEATURE",
    })
    return result


def build_spec_exit_b(state_events: list[dict], *, issue, country: str, episode_id: str,
                      onset: dict, rq01_view: dict, config_by_horizon: dict) -> dict:
    """Call the spec state-anchor/seven-EWI EXIT-B engine for all horizons.

    A missing/invalid state tape yields the engine's ``U`` status, never a
    fabricated FAIL. No EXIT-A input is read.
    """
    from rq02_source.rq02_rev280_dual_exit import core as DC
    from rq02_source.rq02_rev300_exit_math import engine as EN

    controller = DC.parallel_exit_controller(
        issue=issue, country=country, episode_id=episode_id, onset=onset,
        rq01_view=rq01_view, state_events=state_events)
    branches = {}
    for horizon in EN.EXIT_HORIZONS:
        if horizon not in config_by_horizon:
            raise ValueError(f"EXIT_B_CONFIG_MISSING_H{horizon}")
        branches[horizon] = EN.branch_B(
            issue=issue, horizon=horizon, controller=controller,
            config=config_by_horizon[horizon])
    return {"controller_status": controller.get("status"), "controller": controller,
            "branches": branches, "role": "RESEARCH_RULE_VOTE_UNFITTED"}


def main_lock_week(*, packet: dict, state, decision_at, weights_pre, returns_next, nav: float,
                   regime: str = "N", risk_measures: dict | None = None,
                   risk_caps: dict | None = None) -> dict:
    """Apply the RQ03 Main-only lock to one week.

    Locked weeks return the exact passive step and a risk monitor; unlocked or
    blocked weeks return the gate and no trade instruction.
    """
    from research_adapters.rq03 import main_lock as ML

    events = ML.lock_events_from_packet(packet)
    gate = ML.gate_strategy("POLICY", state, events, decision_at)
    if str(gate.get("status", "")).startswith("BLOCKED"):
        return {"gate": gate, "locked": None, "step": None, "monitor": None}
    locked = bool(gate.get("locked"))
    if not locked:
        return {"gate": gate, "locked": False, "step": None, "monitor": None}
    step = ML.passive_locked_step(weights_pre, returns_next, nav)
    monitor = ML.locked_risk_monitor(
        step["weights_next"], regime,
        risk_measures=risk_measures, risk_caps=risk_caps)
    return {"gate": gate, "locked": True, "step": step, "monitor": monitor}
