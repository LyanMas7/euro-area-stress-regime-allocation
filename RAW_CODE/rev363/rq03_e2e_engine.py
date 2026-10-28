"""RQ03-02 / 05 / 06 / 07 / 09: five native-geometry benchmarks + MAIN on one weekly ledger per inception.

Accounting (spec RQ03 risk tolerance section 1; master spec J1):  pre-trade weights p (6, drifted, sum 1);
a candidate post-trade weight vector w (6, sum 1) costs the self-financing fee C solving
C = sum_i k_i |(1-C) w_i - p_i| (research_adapters.rq03.trading.self_financing_cost; k = fee on the five
risky assets, 0 on C1).  X = (1-C) w, NAV_{t+1} = NAV_t (1-C)(1 + w'r_t), weights then drift.  The fee is
charged exactly once; a week with no fill has C = 0 and turnover = 0.

Hard envelope (veto before utility, all six strategies, RT09): K_N = {0.50 <= a <= 0.75}, K_S = {a <= 0.50},
every RISKY asset <= 60% (C1 is the residual), and the ES cap of the RiskConfig on the post-fee opening-NAV
loss.  The ES95 = 2% cap, the 10 bp fee and the regime risky-share targets are ASSUMED_RESEARCH_ONLY
numbers already registered in the code base; nothing is owner approved.

Benchmarks (EW/IVOL/MVP/ERC/MaxDiv): target w* = (a v*_b, 1-a), v*_b from research_adapters.rq03.benchmarks
(exact KKT/Newton solvers), a = the largest share <= the regime ceiling whose post-fee ES meets the cap.  The
sole discretionary activation is the authoritative six-capital-weight rule: compare the DRIFTED current
weights against that benchmark's immutable confirmed base and request restructuring iff at least two assets
deviate by >= 5 percentage points.  Native structural deviation is retained only as a diagnostic/target
quantity and cannot activate a trade.  A hard-limit breach remains an independent RISK_REPAIR path and never
proves that the weight gate fired.  The five benchmark runners never read Main lock events (R23 exemption).

MAIN: research_adapters.rq03.main_lock.gate_strategy first (NO_TRADE_LOCKED -> passive drift, risk monitor
only).  Unlocked: Common target q_t = rq03_next.policy_weights (registered POLICY target), lambda_G in
{0, .25, .5, .75, 1} (trading.candidate_actions grid) chosen by max mean log(1 - C + X'r_j) over the trailing
104 weeks inside the feasible set (spec 5.1); Bottom overlay (spec 5.2) only in BOTTOM/RECOVERY phases with a
current 3F permission, b in {0, B/2, B} with the registered research capacity B = 15% of risky capital moved
from bonds to the permitted equities, fees C_total = C_G + (1-C_G) C_B and the global shield re-checked on the
final weights (spec 5.3).  The owner B_max formula (m_s, LB/UB probabilities, sigma_ref) has no owner inputs
and is reported BLOCKED; the 15% overlay is a labelled research challenger.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np

from research_adapters.rq03 import benchmarks as BM
from research_adapters.rq03 import main_lock as ML
from research_adapters.rq03 import trading as TR

from . import rq03_benchmark_gate as BG
from . import rq03_e2e_risk as RK
from .rq03 import project_to_envelope
from .rq03_next import policy_weights

BENCH = BG.BENCHMARKS
STRATEGIES = (*BENCH, "MAIN")
STRESS = ("D", "G", "BOTH")
LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)
BOTTOM_SHARE = 0.15            # rq03_next.policy_weights registered Bottom research capacity (15% of risky capital)
ES_CAP = 0.02                  # registered ES95 research budget (rq03_next.RISK_BUDGET)
ASSET_CAP = 0.60
TOL = 1e-9


@dataclass(frozen=True)
class RiskConfig:
    name: str
    cov: str            # 'DCC' | 'EWMA'
    measure: str        # 'GAUSS' | 'T5' | 'HES104'
    alpha: float = 0.95
    es_cap: float = ES_CAP


RISK_CONFIGS = (RiskConfig("DCC_GAUSS", "DCC", "GAUSS"), RiskConfig("DCC_T5", "DCC", "T5"),
                RiskConfig("EWMA_GAUSS", "EWMA", "GAUSS"), RiskConfig("EWMA_T5", "EWMA", "T5"),
                RiskConfig("HS104_HES", "DCC", "HES104"))


@dataclass
class PeriodInputs:
    """Everything one inception run may see; arrays are aligned on the period weeks (index 0 = inception)."""
    period: str
    week_start: list
    week_end: list
    R: np.ndarray                 # n x 6 realised returns of week t
    cov: dict                     # {'DCC': n x5x5, 'EWMA': n x5x5}
    ready: dict                   # {'DCC': n bool, 'EWMA': n bool}
    hist: np.ndarray              # n x 104 x 6 trailing completed weekly returns (j < t); NaN-padded
    hist_n: np.ndarray            # number of valid trailing weeks
    regime: np.ndarray            # issued EA regime per week: N/D/G/BOTH/U
    votes: np.ndarray             # 3F votes (NaN if unknown)
    vote_known: np.ndarray
    asset_flags: np.ndarray       # n x 3 asset-specific equity Bottom flags (prior weeks only)
    signal_month: list
    published_at: list
    shapes: dict = field(default_factory=dict)   # {(cov, method): n x 5}


# --------------------------------------------------------------------------- shared helpers
def fee_vector(bps: float) -> np.ndarray:
    return np.array([bps / 1e4] * 5 + [0.0])


def trade_cost(pre, post, k) -> tuple[float, float]:
    """Self-financing fee C = sum_i k_i |(1-C) w_i - p_i| (the fixed point solved by trading.self_financing_cost).

    g(C) is a contraction with modulus <= max(k) * sum(w) < 1, so the iteration from C0 = k'|w - p| converges
    geometrically (modulus <= 0.0025 at 25 bp); parity with the brentq solver is a regression test."""
    p, w = np.asarray(pre, float), np.asarray(post, float)
    if np.abs(w - p).max() <= 1e-13:
        return 0.0, 0.0
    if not k.any():
        return 0.0, float(np.abs(w - p).sum())
    c = float(k @ np.abs(w - p))
    for _ in range(60):
        nxt = float(k @ np.abs((1.0 - c) * w - p))
        if abs(nxt - c) <= 1e-17:
            c = nxt
            break
        c = nxt
    return c, float(np.abs((1.0 - c) * w - p).sum())


def envelope_ok(w6, regime: str) -> tuple[bool, list]:
    a = float(w6[:5].sum())
    br = []
    if regime == "N":
        if a < 0.50 - TOL:
            br.append("RISKY_SHARE_BELOW_NORMAL_FLOOR")
        if a > 0.75 + TOL:
            br.append("RISKY_SHARE_ABOVE_NORMAL_CEILING")
    elif regime in STRESS or regime == "S":
        if a > 0.50 + TOL:
            br.append("RISKY_SHARE_ABOVE_STRESS_CEILING")
    if np.any(w6[:5] > ASSET_CAP + TOL):
        br.append("RISKY_ASSET_CAP_60")
    if w6.min() < -TOL:
        br.append("NEGATIVE_HOLDING")
    return not br, br


def es_value(w6, cost, t, P: PeriodInputs, cfg: RiskConfig) -> float:
    c1 = float(P.R[t, 5])                          # known C1 accrual of week t (fixing released before Monday)
    if cfg.measure == "HES104":
        n = int(P.hist_n[t])
        if n < 52:
            return float("nan")
        loss = RK.scenario_losses(w6, cost, P.hist[t, -n:])
        return RK.hist_es(loss, cfg.alpha)[1]
    return RK.parametric_var_es(w6, cost, P.cov[cfg.cov][t], c1, cfg.alpha, cfg.measure)[1]


def risk_ready(t, P: PeriodInputs, cfg: RiskConfig) -> bool:
    if not P.ready[cfg.cov][t]:
        return False
    return cfg.measure != "HES104" or P.hist_n[t] >= 52


def feasible(w6, cost, regime, t, P, cfg) -> tuple[bool, list, float]:
    ok, br = envelope_ok(w6, regime)
    es = es_value(w6, cost, t, P, cfg)
    if not np.isfinite(es):
        return False, br + ["ES_UNDEFINED"], es
    if es > cfg.es_cap + 1e-12:
        br = br + ["ES_CAP"]
    return not br, br, es


def drift(w_post, r) -> np.ndarray:
    g = w_post * (1.0 + r)
    return g / g.sum()


def band(regime: str) -> tuple[float, float]:
    return (0.50, 0.75) if regime == "N" else (0.0, 0.50)


def scaled_target(shape5, a, regime) -> np.ndarray:
    risky = project_to_envelope(a * np.asarray(shape5, float), regime if regime in ("N", *STRESS) else "S")
    return np.r_[risky, 1.0 - risky.sum()]


def max_feasible_target(pre, shape5, regime, t, P, cfg, k) -> tuple[np.ndarray | None, float, float]:
    """Largest risky share a in the regime band (shape kept, 60% cap redistributed) meeting every hard limit."""
    lo, hi = band(regime)
    w = scaled_target(shape5, hi, regime)
    c, _ = trade_cost(pre, w, k)
    ok, _, _ = feasible(w, c, regime, t, P, cfg)
    if ok:
        return w, hi, c
    w = scaled_target(shape5, lo, regime) if lo > 0 else np.r_[np.zeros(5), 1.0]
    c, _ = trade_cost(pre, w, k)
    if not feasible(w, c, regime, t, P, cfg)[0]:
        return None, float("nan"), float("nan")
    a_lo, a_hi = lo, hi
    for _ in range(30):
        mid = 0.5 * (a_lo + a_hi)
        w = scaled_target(shape5, mid, regime)
        c, _ = trade_cost(pre, w, k)
        if feasible(w, c, regime, t, P, cfg)[0]:
            a_lo = mid
        else:
            a_hi = mid
    w = scaled_target(shape5, a_lo, regime) if a_lo > 0 else np.r_[np.zeros(5), 1.0]
    c, _ = trade_cost(pre, w, k)
    return w, a_lo, c


def minimal_repair(pre, regime, t, P, cfg, k) -> tuple[np.ndarray | None, float]:
    """Smallest risk repair of a HOLD that breaks a hard limit: keep the DRIFTED risky shape, move the risky
    share to the nearest feasible level (clip into the band, then lower it until the ES cap holds), redistribute
    a breached 60% asset cap.  None if even the band floor breaks the ES cap (infeasible -> BLOCKED)."""
    a_pre = float(pre[:5].sum())
    if a_pre <= 1e-12:
        return None, float("nan")
    shape = pre[:5] / a_pre
    lo, hi = band(regime)
    a0 = min(max(a_pre, lo), hi)

    def ok(a):
        w = scaled_target(shape, a, regime) if a > 0 else np.r_[np.zeros(5), 1.0]
        c, _ = trade_cost(pre, w, k)
        return feasible(w, c, regime, t, P, cfg)[0], w
    good, w = ok(a0)
    if good:
        return w, a0
    if not ok(lo)[0]:
        return None, float("nan")
    a_lo, a_hi = lo, a0
    for _ in range(30):
        mid = 0.5 * (a_lo + a_hi)
        if ok(mid)[0]:
            a_lo = mid
        else:
            a_hi = mid
    return ok(a_lo)[1], a_lo


def benchmark_shapes(cov_series: np.ndarray, ready: np.ndarray, method: str, estimator: str) -> np.ndarray:
    out = np.full((len(cov_series), 5), np.nan)
    for t in range(len(cov_series)):
        if not ready[t]:
            continue
        s = BM.benchmark_shape(method, cov_series[t], sigma_estimator=estimator)
        if s["status"] == "PASS":
            out[t] = s["shape"]
    return out


# --------------------------------------------------------------------------- ledger record
def _rec(store, t, **kw):
    for k, v in kw.items():
        store.setdefault(k, [None] * store["_n"])[t] = v


def _new_store(n, strategy, period, run_id):
    return {"_n": n, "strategy": strategy, "period": period, "run_id": run_id}


def _finish_week(st, t, P, pre, w_post, cost, turnover, nav, action, status, regime, cfg, extra):
    r = P.R[t]
    growth = (1.0 - cost) * (1.0 + float(w_post @ r))
    if not (growth > 0 and math.isfinite(growth)):
        raise ArithmeticError("NONPOSITIVE_NET_NAV")
    ok, br, es = feasible(w_post, cost, regime if regime != "U" else "S", t, P, cfg) if risk_ready(t, P, cfg) \
        else (None, ["RISK_NOT_READY"], float("nan"))
    _rec(st, t, week_start=P.week_start[t], week_end=P.week_end[t], regime=regime, action=action, status=status,
         nav_pre=nav, cost=cost, turnover=turnover, gross_return=float(w_post @ r), net_return=growth - 1.0,
         nav_end=nav * growth, risky_share_post=float(w_post[:5].sum()), post_hard_ok=ok,
         post_breaches=";".join(br), es_post=es, **{f"pre_{a}": float(pre[i]) for i, a in enumerate(RK_ASSETS)},
         **{f"w_{a}": float(w_post[i]) for i, a in enumerate(RK_ASSETS)}, **extra)
    return nav * growth, drift(w_post, r)


RK_ASSETS = ("E1", "E2", "E3", "B1", "B2", "C1")


# --------------------------------------------------------------------------- benchmarks
def benchmark_restructure_action(pre, target, regime, t, P, cfg, k) -> dict | None:
    """Largest feasible FULL/PARTIAL move toward a valid benchmark target.

    The candidate set comes from the retained trading utility.  Expected loss, wealth
    and utility are intentionally absent: feasibility only constrains execution after
    the fixed weight-band request has already been issued.
    """
    packet = TR.candidate_actions(pre, target, lambdas=(0.25, 0.50, 0.75), cash_fracs=())
    candidates = [c for c in packet["candidates"] if c["action"] in ("FULL", "PARTIAL")]
    candidates.sort(key=lambda c: float(c["param"]), reverse=True)
    for cand in candidates:
        w = np.asarray(cand["weights"], dtype=float)
        cost, turnover = trade_cost(pre, w, k)
        ok, breaches, es = feasible(w, cost, regime, t, P, cfg)
        if ok:
            return {"action": cand["action"], "lambda": float(cand["param"]), "weights": w,
                    "cost": cost, "turnover": turnover, "es": es, "breaches": breaches}
    return None


def _empty_base_update(reason: str) -> dict:
    return {"updated": False, "rebase_applied": False, "update_kind": "NONE", "reason": reason,
            "before_snapshot_id": None, "after_snapshot_id": None}


def run_benchmark(P: PeriodInputs, method: str, fee_bps: float, cfg: RiskConfig, *,
                  w0: float = 1.0, run_id: str = "", unknown_policy: str = "HOLD") -> dict:
    if method not in BENCH:
        raise ValueError("UNKNOWN_RQ03_BENCHMARK")
    n = len(P.R)
    k = fee_vector(fee_bps)
    shapes = P.shapes[(cfg.cov, method)]
    pre = np.r_[np.zeros(5), 1.0]                 # inception: W0 held in C1 before the first fill
    nav = float(w0)
    st = _new_store(n, method, P.period, run_id)
    base_state = BG.BenchmarkBaseState(method)
    base_events = []
    for t in range(n):
        regime = P.regime[t]
        eff = regime if regime in ("N", *STRESS) else ("S" if unknown_policy == "RISK_ONLY_CAP50" else "U")
        base_before = base_state.snapshot
        gate = BG.assess_weight_band(method, base_before, pre, current_confirmed=True)
        target = None
        a_t = float("nan")
        extra = {"tau": None, "deviation": None, "native_geometry_status": "NOT_EVALUATED",
                 "native_geometry_reason": None, "native_geometry_deviation": None,
                 "native_geometry_active_trigger": False, "lock_phase": "NOT_APPLICABLE_BENCHMARK",
                 "benchmark_action_lambda": None, "trade_fill_type": "NO_FILL",
                 "hold_hard_ok": None, "hold_breaches": None, "target_risky_share": None}
        if eff == "U" or not risk_ready(t, P, cfg) or not np.isfinite(shapes[t]).all():
            if eff == "U":
                reason = "NO_TRADE_UNKNOWN_REGIME"
            elif not risk_ready(t, P, cfg):
                reason = "NO_TRADE_RISK_WARMUP"
            else:
                reason = "NO_TRADE_BENCHMARK_TARGET_UNDEFINED"
            base_update = _empty_base_update("NO_COMPLETED_TRADE")
            extra.update(BG.ledger_fields(gate, base_before, base_state.snapshot, target_weights=None,
                                          execution_confirmed=False, execution_kind="NONE",
                                          base_update=base_update))
            nav, pre = _finish_week(st, t, P, pre, pre, 0.0, 0.0, nav, "NO_TRADE", reason, regime, cfg, extra)
            continue
        hold_ok, hold_br, _ = feasible(pre, 0.0, eff, t, P, cfg)
        a_pre = float(pre[:5].sum())
        target, a_t, c_t = max_feasible_target(pre, shapes[t], eff, t, P, cfg, k)
        if a_pre > 1e-12:
            native = BM.structural_deviation(method, pre[:5] / a_pre, P.cov[cfg.cov][t], target=shapes[t])
            extra.update({"deviation": native.get("deviation"), "native_geometry_status": native.get("status"),
                          "native_geometry_reason": native.get("reason"),
                          "native_geometry_deviation": native.get("deviation")})

        execution_kind = "NONE"
        action_lambda = None
        if base_before is None:
            if target is None:
                action, status, w = "BLOCKED", "NO_FEASIBLE_INITIAL_ALLOCATION", pre
            else:
                action, status, w = "FULL", "INITIAL_ALLOCATION_RESEARCH_MANDATE", target
                execution_kind, action_lambda = "INITIAL_ALLOCATION", 1.0
        else:
            if not hold_ok:
                candidate = benchmark_restructure_action(pre, target, eff, t, P, cfg, k) \
                    if gate["triggered"] and target is not None else None
                if candidate is not None:
                    action, w = candidate["action"], candidate["weights"]
                    action_lambda = candidate["lambda"]
                    status = "WEIGHT_BAND_REQUEST_AND_HARD_RISK_BREACH:" + ";".join(hold_br)
                    execution_kind = "WEIGHT_BAND_RESTRUCTURE"
                else:
                    rep, _ = minimal_repair(pre, eff, t, P, cfg, k)
                    if rep is None:
                        action, status, w = "BLOCKED", "RISK_BREACH_NO_FEASIBLE_REPAIR:" + ";".join(hold_br), pre
                    else:
                        action, status, w = "RISK_REPAIR", "MINIMAL_REPAIR_RESEARCH_ONLY:" + ";".join(hold_br), rep
                        execution_kind = "HARD_RISK_REPAIR"
            elif gate["status"].startswith("FAIL_CLOSED"):
                action, status, w = "HOLD", "BENCHMARK_GATE_FAIL_CLOSED:" + gate["status"], pre
            elif not gate["triggered"]:
                action, status, w = "HOLD", "WITHIN_FIXED_SIX_WEIGHT_BAND", pre
            else:
                candidate = benchmark_restructure_action(pre, target, eff, t, P, cfg, k) if target is not None else None
                if candidate is None:
                    action, status, w = "HOLD", "AUTO_RESTRUCTURE_REQUEST_BLOCKED_NO_FEASIBLE_TARGET", pre
                else:
                    action, w = candidate["action"], candidate["weights"]
                    action_lambda = candidate["lambda"]
                    status = "AUTHORITATIVE_WEIGHT_BAND_TRIGGER"
                    execution_kind = "WEIGHT_BAND_RESTRUCTURE"
        cost, turn = trade_cost(pre, w, k)
        if action in ("HOLD", "BLOCKED", "NO_TRADE"):
            cost, turn, w = 0.0, 0.0, pre
        elif action in ("FULL", "PARTIAL") and turn <= BG.FILL_ATOL:
            action, status, cost, turn, w = "HOLD", "AUTO_RESTRUCTURE_REQUEST_TARGET_EQUALS_CURRENT_NO_FILL", 0.0, 0.0, pre
            execution_kind, action_lambda = "NONE", None

        execution_confirmed = action in ("FULL", "PARTIAL", "RISK_REPAIR") and turn > BG.FILL_ATOL
        base_update = _empty_base_update(
            "HARD_RISK_REPAIR_NOT_REBASE_ELIGIBLE" if execution_kind == "HARD_RISK_REPAIR" else "NO_COMPLETED_TRADE")
        if execution_kind in ("INITIAL_ALLOCATION", "WEIGHT_BAND_RESTRUCTURE"):
            execution_id = f"{run_id or P.period}:{method}:{P.week_start[t]}:{action}"
            provenance = json.dumps({"rule_id": BG.TRIGGER_RULE_ID, "gate_status": gate["status"],
                                     "risk_config": cfg.name, "fee_bps": fee_bps,
                                     "target_geometry": BM.GEOMETRY[method], "action": action,
                                     "lambda": action_lambda}, sort_keys=True)
            base_update = base_state.record_completed_fill(
                w, effective_at=P.week_start[t], nav_after_cost=nav * (1.0 - cost), turnover=turn,
                execution_id=execution_id, execution_kind=execution_kind,
                trade_confirmed=execution_confirmed, completed=execution_confirmed,
                gate_status=gate["status"], provenance=provenance)
            if base_update["updated"]:
                base_events.append({**base_state.snapshot.record(), "snapshot_id": base_state.snapshot.snapshot_id,
                                    "week_index": t, "action": action, "gate_status": gate["status"]})
        extra.update({"hold_hard_ok": hold_ok, "hold_breaches": ";".join(hold_br),
                      "target_risky_share": a_t, "benchmark_action_lambda": action_lambda,
                      "trade_fill_type": "CONFIRMED_SIMULATED_FILL_NOT_BROKER" if execution_confirmed else "NO_FILL"})
        extra.update(BG.ledger_fields(gate, base_before, base_state.snapshot, target_weights=target,
                                      execution_confirmed=execution_confirmed, execution_kind=execution_kind,
                                      base_update=base_update))
        nav, pre = _finish_week(st, t, P, pre, w, cost, turn, nav, action, status, regime, cfg, extra)
    st["_base_snapshot_events"] = base_events
    st["_final_base_snapshot"] = base_state.snapshot.record() if base_state.snapshot is not None else None
    st["_benchmark_trigger_rule"] = BG.TRIGGER_RULE_ID
    return st


# --------------------------------------------------------------------------- MAIN lock events
def lock_events(P: PeriodInputs, k_votes: int) -> list:
    """Issued, as-of Main lock events built from the issued EA regime and the issued 3F vote.

    SHORT_CONFIRMED: the first published month whose issued EA regime is Stress (D/G/BOTH) outside an open
    episode; additional Stress months inside a locked episode are further SHORT_CONFIRMED of the SAME episode
    (anchor update, T03).  BOTTOM_CONFIRMED: the first later week whose issued 3F vote (F1/F2 from the
    published month, F3 from completed weeks) is >= k_votes.  EXIT_CONFIRMED: issued EA regime back to N after
    the Bottom; EPISODE_ENDED one week later.  An EXIT before a Bottom never releases the lock (R23)."""
    ev, phase, ep, n_ep = [], "IDLE", None, 0
    last_short_month, lock_week = None, None
    for t in range(len(P.R)):
        reg, month, pub = P.regime[t], P.signal_month[t], P.published_at[t]
        dec = P.week_start[t]
        if phase in ("IDLE", "POST") and reg in STRESS:
            n_ep += 1
            ep = f"{P.period}_EP{n_ep:02d}_{month}"
            ev.append(_event(f"{ep}_S_{month}", "SHORT_CONFIRMED", ep, pub, pub, pub, 0))
            phase, last_short_month, lock_week = "LOCKED", month, t
        elif phase == "LOCKED":
            if reg in STRESS and month != last_short_month:
                ev.append(_event(f"{ep}_S_{month}", "SHORT_CONFIRMED", ep, pub, pub, pub, 0))
                last_short_month = month
            elif t > lock_week and P.vote_known[t] and P.votes[t] >= k_votes:
                at = dec - np.timedelta64(1, "s")
                ev.append(_event(f"{ep}_B_{t}", "BOTTOM_CONFIRMED", ep, at, at, at, 1))
                phase = "BOTTOM"
        elif phase == "BOTTOM" and reg == "N":
            at = dec - np.timedelta64(1, "s")
            ev.append(_event(f"{ep}_X_{t}", "EXIT_CONFIRMED", ep, at, at, at, 2))
            phase = "EXIT"
        elif phase == "EXIT":
            at = dec - np.timedelta64(1, "s")
            ev.append(_event(f"{ep}_E_{t}", "EPISODE_ENDED", ep, at, at, at, 3))
            phase = "POST"
    return ev


def _iso(x) -> str:
    return str(np.datetime64(x, "s")) + "Z"


def _event(eid, typ, ep, issued, confirmed, available, seq):
    return {"event_id": eid, "type": typ, "episode_id": ep, "issued_at": _iso(issued),
            "confirmed_at": _iso(confirmed), "available_at": _iso(available), "issuer_seq": seq}


# --------------------------------------------------------------------------- MAIN
def _utility(w6, cost, t, P) -> float:
    n = int(P.hist_n[t])
    if n < 52:
        return float("nan")
    g = (1.0 - cost) * (1.0 + P.hist[t, -n:] @ w6)
    if np.any(g <= 0):
        return -np.inf
    return float(np.mean(np.log(g)))


def run_main(P: PeriodInputs, fee_bps: float, cfg: RiskConfig, *, events: list | None, overlay: bool,
             k_votes: int, w0: float = 1.0, run_id: str = "") -> dict:
    n = len(P.R)
    k = fee_vector(fee_bps)
    pre = np.r_[np.zeros(5), 1.0]
    nav = float(w0)
    st = _new_store(n, "MAIN", P.period, run_id)
    state = ML.MainLockState()
    seen: list = []
    events = list(events or [])
    for t in range(n):
        regime = P.regime[t]
        dec = _iso(P.week_start[t])
        while events and events[0]["available_at"] <= dec:
            seen.append(events.pop(0))
        gate = ML.gate_strategy("MAIN", state, seen, dec)
        extra = {"tau": None, "deviation": None, "lock_phase": gate.get("phase", gate.get("status")),
                 "episode_id": gate.get("episode_id"), "lambda_G": None, "bottom_b": None,
                 "C_common": None, "C_bottom": None, "utility": None}
        if gate["status"] != "OK":
            nav, pre = _finish_week(st, t, P, pre, pre, 0.0, 0.0, nav, "BLOCKED_INVALID_INPUT",
                                    "LOCK_GATE:" + gate["status"], regime, cfg, extra)
            continue
        if gate["locked"]:
            ready = risk_ready(t, P, cfg)
            mon = ML.locked_risk_monitor(pre, regime if regime != "U" else "U") if regime != "U" else \
                {"status": "NOT_EVALUATED"}
            es = es_value(pre, 0.0, t, P, cfg) if ready else float("nan")
            breach = mon["status"] == "LOCKED_RISK_BREACH" or (np.isfinite(es) and es > cfg.es_cap + 1e-12)
            extra["lock_monitor"] = ("LOCKED_RISK_BREACH:ESCALATE_TO_OWNER" if breach else mon["status"])
            nav, pre = _finish_week(st, t, P, pre, pre, 0.0, 0.0, nav, ML.NO_TRADE_LOCKED,
                                    "FROZEN_SHORT_TO_BOTTOM", regime, cfg, extra)
            continue
        if regime not in ("N", *STRESS) or not risk_ready(t, P, cfg) or P.hist_n[t] < 52:
            nav, pre = _finish_week(st, t, P, pre, pre, 0.0, 0.0, nav, "BLOCKED_INVALID_INPUT",
                                    "U_REGIME_OR_RISK_WARMUP", regime, cfg, extra)
            continue
        q_r = policy_weights(P.cov[cfg.cov][t], regime, 0)
        q = np.r_[q_r, 1.0 - q_r.sum()]
        a_q = float(q[:5].sum())
        shape = q[:5] / a_q
        q_feas, _, _ = max_feasible_target(pre, shape, regime, t, P, cfg, k)
        hold_ok, hold_br, _ = feasible(pre, 0.0, regime, t, P, cfg)
        cands = []
        if q_feas is not None:
            for lam in LAMBDAS:
                w = (1 - lam) * pre + lam * q_feas
                c, turn = trade_cost(pre, w, k)
                ok, br, es = feasible(w, c, regime, t, P, cfg)
                if ok:
                    cands.append((lam, w, c, turn, _utility(w, c, t, P)))
        elif hold_ok:
            cands.append((0.0, pre, 0.0, 0.0, _utility(pre, 0.0, t, P)))
        if not hold_ok:
            rep, _ = minimal_repair(pre, regime, t, P, cfg, k)
            if rep is not None:
                c, turn = trade_cost(pre, rep, k)
                cands.append((-1.0, rep, c, turn, _utility(rep, c, t, P)))     # lambda -1 = minimal repair
        if not cands:
            nav, pre = _finish_week(st, t, P, pre, pre, 0.0, 0.0, nav, "BLOCKED_RISK_BREACH",
                                    "NO_FEASIBLE_COMMON_ACTION:" + ";".join(hold_br), regime, cfg,
                                    {**extra, "hold_hard_ok": hold_ok})
            continue
        best = max(cands, key=lambda x: (x[4], -abs(x[0])))
        lam, wG, cG, turnG, uG = best
        if lam == 0.0:
            action = "HOLD_BY_UTILITY"
        elif not hold_ok:
            action = "RISK_REPAIR"
        else:
            action = "FULL" if lam == 1.0 else "PARTIAL"
        status = "COMMON_LOG_UTILITY" + ("" if hold_ok else ":HOLD_INFEASIBLE:" + ";".join(hold_br))
        w_fin, c_tot, turn_tot, b_star, cB = wG, cG, turnG, 0.0, 0.0
        overlay_status = "OFF"
        if overlay:
            elig = gate["F_bottom"] == "ELIGIBLE"
            perm = bool(P.vote_known[t] and P.votes[t] >= k_votes)
            flags = np.asarray(P.asset_flags[t], bool)
            if not elig:
                overlay_status = "NOT_ELIGIBLE_PHASE"
            elif not perm:
                overlay_status = "ELIGIBLE_NO_CURRENT_PERMISSION_HOLD"
            elif not flags.any():
                overlay_status = "ELIGIBLE_NO_ASSET_SPECIFIC_FLAG_HOLD"
            else:
                bonds = wG[3:5].sum()
                B = min(BOTTOM_SHARE * float(wG[:5].sum()), float(bonds))
                best_b = (0.0, wG, cG, turnG, uG, 0.0)
                for b in (0.5 * B, B):
                    if b <= 1e-12:
                        continue
                    e = np.r_[flags / flags.sum(), 0, 0, 0] * b
                    v = np.r_[0, 0, 0, wG[3:5] / bonds, 0] * b
                    w_try = wG + e - v
                    cb, tb = trade_cost(wG, w_try, k)
                    ct = ML.total_two_layer_cost(cG, cb)["C_total"]
                    w_post = w_try                         # post-trade weights; both fees on opening NAV
                    ok, br, es = feasible(w_post, ct, regime, t, P, cfg)       # global shield
                    u = _utility(w_post, ct, t, P)
                    if ok and u >= best_b[4]:
                        best_b = (b, w_post, ct, turnG + tb, u, cb)
                b_star, w_fin, c_tot, turn_tot, uG, cB = best_b
                overlay_status = "BOTTOM_OVERLAY_APPLIED" if b_star > 0 else "BOTTOM_OVERLAY_HOLD_SHIELD_OR_UTILITY"
        extra.update({"lambda_G": lam, "bottom_b": b_star, "C_common": cG, "C_bottom": cB, "utility": uG,
                      "overlay_status": overlay_status, "hold_hard_ok": hold_ok,
                      "hold_breaches": ";".join(hold_br)})
        if c_tot == 0.0 and np.abs(w_fin - pre).max() <= 1e-13:
            w_fin, turn_tot = pre, 0.0
        nav, pre = _finish_week(st, t, P, pre, w_fin, c_tot, turn_tot, nav, action, status, regime, cfg, extra)
    return st
