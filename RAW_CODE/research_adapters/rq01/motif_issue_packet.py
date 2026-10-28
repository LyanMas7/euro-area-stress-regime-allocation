"""Motif-aware RQ01 issue packet: exact 10x9 Joint + six Pool policies.

Additive compatibility layer over the retained `pairs.py`, `states.py`,
`pools.py` and `pair_selection.py`.  It does not change any protected body.

Key semantics:
* paired Strong/Weak are counted on the SAME matched terminal bases of both sources;
* Gate2 = Strong OR (Weak AND Count), so a Strong pair is NOT ANDed with Count;
* Sideway motifs use a native path-stability proof and a band threshold;
* selected configurations are the only configurations consumed by the Pool;
* GLOBAL NO-U: an unresolved component is NOT_EVALUATED (no value, explicit reason), never 0/FAIL and
  never a third vote; motif letters U/D/S are direction codes (U = UP).
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

from ..vote_contract import is_nonvote, not_evaluated
from .common import SOURCES
from .pairs import (MOTIFS, PAIRS, LagPolicy, TerminalOwner, joint_count,
                    path_stability, strict_gate1_domain)
from .pair_selection import (JointConfig, evaluate_pair_gates, joint_id)
from .pools import OPERATORS, pool_decision
from .selected_pair_consumer_rev360 import pair_consumer_gate
from .states import StateOwner

POLICY_ID = "RQ01_MOTIF_ISSUE_PACKET_RESEARCH_V1"
GATE1_MODE = "MOTIF_AWARE"
SELECTED_MOTIF = "UU"
SELECTED_POOL = "COMPLEMENT_AND_JOINT"
OWNER = "RQ01_MOTIF_RESEARCH_OWNER_V1"
LAG_OWNER = "RQ01_MOTIF_RESEARCH_LAG_V1"
RELEASE_RULE = "RESEARCH_ASOF_TAPE_NOT_OFFICIAL_ISSUER_PIT"
UNIT_SIGN = "RISK_ORIENTED"


def research_state_and_terminal(source: str, values, prof: dict | None):
    """Research owners from prior-delta profiles; never an owner approval."""
    import numpy as np
    hi = float(prof["hi"]) if prof else 1.0
    lo = float(prof["lo"]) if prof else 0.5
    hi = max(hi, 1e-9)
    lo = max(0.0, min(lo, hi))
    vals = [float(v) for v in values] or [0.0]
    eq = float(np.median(vals))
    side = max(0.0, min(lo, hi * 0.5))
    state = StateOwner(source, OWNER, equilibrium=eq, n_min=3,
                       slope_max=hi, flat_slope=min(hi, 0.05 * hi),
                       range_max=max(3.0 * hi, 1e-9), rmse_max=max(hi, 1e-9),
                       equilibrium_band=max(2.0 * hi, 1e-9),
                       terminal_band=max(2.0 * hi, 1e-9), band_fraction=0.6,
                       up_net=max(hi, 1e-9), down_net=max(hi, 1e-9),
                       up_step=max(hi, 1e-9), down_step=max(hi, 1e-9),
                       side_step=max(min(side, 0.5 * hi), 0.0))
    term = TerminalOwner(source, OWNER, base=1, up_min=max(hi, 1e-9),
                         down_min=max(hi, 1e-9), side_max=side)
    return state, term


def research_pair_thresholds(n: int, pa: dict | None, pb: dict | None) -> dict:
    hi_a = max(float(pa["hi"]) if pa else 1.0, 1e-9)
    lo_a = max(0.0, min(float(pa["lo"]) if pa else 0.5, hi_a))
    hi_b = max(float(pb["hi"]) if pb else 1.0, 1e-9)
    lo_b = max(0.0, min(float(pb["lo"]) if pb else 0.5, hi_b))
    m = max(1, n * (n - 1) // 2)
    return {"high_a": hi_a, "low_a": lo_a, "high_b": hi_b, "low_b": lo_b,
            "gate1_a": lo_a, "gate1_b": lo_b,
            "side_strong_a": lo_a, "side_weak_a": hi_a,
            "side_strong_b": lo_b, "side_weak_b": hi_b,
            "min_bases": max(1, min(2, n - 1)),
            "count_k": max(1, int(math.ceil(0.5 * m)))}


def _tri(v, reason: str = "VOTE_NOT_EVALUATED"):
    """Binary vote -> 'PASS'/'FAIL'; anything else is a NonVote (not evaluated), never a third token."""
    if v is True or v == 1:
        return "PASS"
    if v is False or v == 0:
        return "FAIL"
    return not_evaluated(reason)


def _comp(v):
    """Serialise a component: binary value or None (its reason goes to a separate *_reason field)."""
    return None if is_nonvote(v) else v


def _comp_reason(v):
    return v.reason if is_nonvote(v) else None


def _base_deltas(series: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """Terminal bases (base_id, value_end - value_base) for a matched native series."""
    if len(series) < 2:
        return []
    end = series[-1]
    out = []
    for a in series[:-1]:
        if a is None or end is None or a[0] == end[0]:
            continue
        out.append((a[0], float(end[1]) - float(a[1])))
    return out


def paired_strong_weak_gate2(*, series_a, series_b, motif, high_a, low_a, high_b, low_b,
                             min_bases: int, count_lower: int, count_upper: int, count_k: int,
                             side_strong_a=None, side_weak_a=None, side_strong_b=None,
                             side_weak_b=None, stability_a="PASS", stability_b="PASS") -> dict:
    """Gate2 over paired terminal bases with the IJP conditional Count.

    Up/Down: pass if oriented terminal >= high/low; Sideway: |terminal| <= side band.
    Strong arm ignores Count; Weak arm requires Count.  Missing evidence is U.
    """
    if motif not in MOTIFS:
        raise ValueError("ORDERED_MOTIF_REQUIRED")
    if len(series_a) != len(series_b) or len(series_a) < 2:
        return {"status": "NOT_EVALUATED", "reason": "SAME_MATCHED_GRID_REQUIRED"}
    da, db = _base_deltas(series_a), _base_deltas(series_b)
    n = min(len(da), len(db))
    if n == 0:
        return {"status": "NOT_EVALUATED", "reason": "NO_TERMINAL_BASES"}
    if not (1 <= min_bases <= n):
        return {"status": "NOT_EVALUATED", "reason": "MIN_BASES_INFEASIBLE"}
    letter_a, letter_b = motif
    if letter_a == "S" or letter_b == "S":
        if stability_a != "PASS" or stability_b != "PASS":
            return {"status": "NOT_EVALUATED", "reason": "SIDEWAY_PATH_STABILITY_NOT_PROVEN",
                    "strong": None, "weak": None,
                    "stability_a": _comp(stability_a), "stability_b": _comp(stability_b),
                    "stability_a_reason": _comp_reason(stability_a),
                    "stability_b_reason": _comp_reason(stability_b)}
        sa = side_strong_a if side_strong_a is not None else high_a
        wa = side_weak_a if side_weak_a is not None else low_a
        sb = side_strong_b if side_strong_b is not None else high_b
        wb = side_weak_b if side_weak_b is not None else low_b
        strong = sum(abs(a[1]) <= sa and abs(b[1]) <= sb for a, b in zip(da, db))
        weak = sum(abs(a[1]) <= wa and abs(b[1]) <= wb for a, b in zip(da, db))
    else:
        def d(letter, value):
            return value if letter == "U" else -value
        strong = sum(d(letter_a, a[1]) >= high_a and d(letter_b, b[1]) >= high_b
                     for a, b in zip(da, db))
        weak = sum(d(letter_a, a[1]) >= low_a and d(letter_b, b[1]) >= low_b
                   for a, b in zip(da, db))
    strong_v = int(strong >= min_bases)
    weak_v = int(weak >= min_bases)
    if count_lower >= count_k:
        count_v = "PASS"
    elif count_upper < count_k:
        count_v = "FAIL"
    else:
        count_v = not_evaluated("COUNT_INTERVAL_STRADDLES_K")
    if strong_v:
        gate2, reason = "PASS", None
    elif weak_v:
        gate2 = count_v
        reason = count_v.reason if is_nonvote(count_v) else None
    else:
        gate2, reason = "FAIL", "NEITHER_STRONG_NOR_WEAK"
    out = {"strong": strong_v, "weak": weak_v, "count": _comp(count_v), "count_reason": _comp_reason(count_v),
           "n_bases": n, "strong_bases": int(strong), "weak_bases": int(weak), "reason": reason,
           "min_bases": int(min_bases), "count_k": int(count_k),
           "count_lower": int(count_lower), "count_upper": int(count_upper)}
    if is_nonvote(gate2):
        return {"status": "NOT_EVALUATED", **out}
    return {"status": "PASS", "vote": 1 if gate2 == "PASS" else 0, "gate2": gate2, **out}


def _gate1_all_bases(series_a, series_b, motif, term_a, term_b, *,
                     gate1_a: float, gate1_b: float, stability_a: str, stability_b: str) -> dict:
    da, db = _base_deltas(series_a), _base_deltas(series_b)
    if not da or not db:
        return {"status": "NOT_EVALUATED", "reason": "NO_TERMINAL_BASES"}
    la, lb = motif
    if "S" in motif:
        if stability_a != "PASS" or stability_b != "PASS":
            return {"status": "NOT_EVALUATED", "reason": "SIDEWAY_PATH_STABILITY_NOT_PROVEN"}
        ok = any(abs(a[1]) <= gate1_a and abs(b[1]) <= gate1_b for a, b in zip(da, db))
        return {"status": "PASS" if ok else "FAIL", "mode": "MOTIF_AWARE_SIDEWAY"}
    def d(letter, value):
        return value if letter == "U" else -value
    oka = any(d(la, a[1]) >= gate1_a for a in da)
    okb = any(d(lb, b[1]) >= gate1_b for b in db)
    return {"status": "PASS" if (oka and okb) else "FAIL", "mode": "MOTIF_AWARE_DIRECTIONAL"}


def evaluate_pair_motif(*, pair, series_a, series_b, motif, lag, state_a, state_b,
                        term_a, term_b, gate1_a, gate1_b, high_a, low_a, high_b, low_b,
                        min_bases, count_k, gate1_mode=GATE1_MODE,
                        side_strong_a=None, side_weak_a=None, side_strong_b=None,
                        side_weak_b=None, config: JointConfig | None = None) -> dict:
    """One pair x ordered motif row; all nine motifs return an explicit status."""
    if len(series_a) < 2 or len(series_b) < 2:
        return {"pair": pair, "motif": motif, "status": "NOT_EVALUATED",
                "reason": "FEWER_THAN_TWO_MATCHED_ANCHORS", "gate1_mode": gate1_mode,
                "gate1": None, "gate2": None, "strong": None, "weak": None, "count": None,
                "gate1_reason": "FEWER_THAN_TWO_MATCHED_ANCHORS",
                "gate2_reason": "FEWER_THAN_TWO_MATCHED_ANCHORS",
                "count_lower": 0, "count_upper": 0, "n_opportunities": 0,
                "paired_count": None, "strong_bases": 0, "weak_bases": 0, "n_bases": 0,
                "stability_a": None, "stability_b": None,
                "min_bases": min_bases, "count_k": count_k,
                "config_id": joint_id(config) if config is not None else None,
                "selected": motif == SELECTED_MOTIF, "lag": (lag.kind, lag.lag, lag.delta)}
    cnt = joint_count(series_a, series_b, motif, lag, state_a, state_b)
    stability_a = path_stability([v for _, v in series_a], state_a)
    stability_b = path_stability([v for _, v in series_b], state_b)
    if gate1_mode == "STRICT":
        dom = strict_gate1_domain(motif, term_a, term_b, 1.0, 1.0, gate1_a)
        if not dom["feasible"]:
            return {"pair": pair, "motif": motif, "status": "BLOCKED",
                    "reason": dom["reason"], "gate1_mode": gate1_mode,
                    "n_opportunities": cnt["n_opportunities"],
                    "count_lower": cnt["count_lower"], "count_upper": cnt["count_upper"]}
    g1 = _gate1_all_bases(series_a, series_b, motif, term_a, term_b,
                          gate1_a=gate1_a, gate1_b=gate1_b,
                          stability_a=stability_a, stability_b=stability_b)
    g2 = paired_strong_weak_gate2(
        series_a=series_a, series_b=series_b, motif=motif,
        high_a=high_a, low_a=low_a, high_b=high_b, low_b=low_b,
        min_bases=min_bases, count_lower=cnt["count_lower"],
        count_upper=cnt["count_upper"], count_k=count_k,
        side_strong_a=side_strong_a, side_weak_a=side_weak_a,
        side_strong_b=side_strong_b, side_weak_b=side_weak_b,
        stability_a=stability_a, stability_b=stability_b)
    g1_gate = g1["status"] if g1["status"] in ("PASS", "FAIL") else None
    g2_gate = g2.get("gate2")                      # 'PASS'/'FAIL' when Gate2 was evaluated, else absent
    if g1_gate == "FAIL" or g2_gate == "FAIL":
        status, vote, reason = "FAIL", 0, None
    elif g1_gate == "PASS" and g2_gate == "PASS":
        status, vote, reason = "PASS", 1, None
    else:
        # a decision the evaluated components cannot prove: no vote, explicit eligibility reason
        status, vote = "NOT_EVALUATED", None
        if g1_gate is None:
            reason = g1.get("reason") or "GATE1_NOT_EVALUATED"
        else:
            reason = g2.get("reason") or "GATE2_NOT_EVALUATED"
    cfg_id = joint_id(config) if config is not None else None
    row = {"pair": pair, "motif": motif, "status": status, "reason": reason,
           "gate1": g1_gate, "gate1_mode": gate1_mode,
           "gate1_reason": g1.get("reason"),
           "gate2": g2_gate, "gate2_reason": g2.get("reason"),
           "strong": g2.get("strong"), "weak": g2.get("weak"),
           "count": g2.get("count"), "count_reason": g2.get("count_reason"),
           "count_lower": cnt["count_lower"],
           "count_upper": cnt["count_upper"], "n_opportunities": cnt["n_opportunities"],
           "paired_count": cnt["native_distinct_count_applied"] if "native_distinct_count_applied" in cnt else None,
           "strong_bases": g2.get("strong_bases", 0), "weak_bases": g2.get("weak_bases", 0),
           "n_bases": g2.get("n_bases", 0), "stability_a": _comp(stability_a), "stability_b": _comp(stability_b),
           "stability_a_reason": _comp_reason(stability_a), "stability_b_reason": _comp_reason(stability_b),
           "min_bases": min_bases, "count_k": count_k,
           "config_id": cfg_id, "selected": motif == SELECTED_MOTIF,
           "lag": (lag.kind, lag.lag, lag.delta)}
    if vote is not None:
        row["vote"] = vote
    return row


def _config_for(pair, motif, lag) -> JointConfig:
    return JointConfig(pair[0], pair[1], motif, lag, UNIT_SIGN, POLICY_ID,
                       RELEASE_RULE, OWNER).canonical()


def build_issue_packet(*, country, issue, pair_series: dict, individual_votes: dict,
                       lag: LagPolicy, state_owners: dict, terminal_owners: dict,
                       thresholds: dict, selected_configs: dict | None = None,
                       research_configs: dict | None = None,
                       selected_pool: str | None = None,
                       pool_fallback: str = "IND_ONLY",
                       gate1_mode=GATE1_MODE) -> dict:
    """Build 10 pairs x 9 motifs + six Pools + selected/consumed parity for one issue.

    `selected_configs` is the registry decision consumed by the Pool.  An empty
    mapping is a valid NO_PAIR_SELECTED outcome: the issued Pool then falls back
    to `pool_fallback` (IND_ONLY) and says so explicitly.  `research_configs`
    feeds the diagnostic feature packet only and is never called selected.
    """
    rows = []
    configs = {}
    for pair in PAIRS:
        a, b = pair
        sa, sb = pair_series[pair]
        th = thresholds[pair]
        sa = [(x[0], float(x[1])) for x in sa]
        sb = [(x[0], float(x[1])) for x in sb]
        for motif in MOTIFS:
            cfg = _config_for(pair, motif, lag)
            configs[(pair, motif)] = cfg
            rows.append(evaluate_pair_motif(
                pair=pair, series_a=sa, series_b=sb, motif=motif, lag=lag,
                state_a=state_owners[a], state_b=state_owners[b],
                term_a=terminal_owners[a], term_b=terminal_owners[b],
                gate1_a=th["gate1_a"], gate1_b=th["gate1_b"],
                high_a=th["high_a"], low_a=th["low_a"],
                high_b=th["high_b"], low_b=th["low_b"],
                min_bases=th["min_bases"], count_k=th["count_k"],
                gate1_mode=gate1_mode, config=cfg,
                side_strong_a=th.get("side_strong_a"), side_weak_a=th.get("side_weak_a"),
                side_strong_b=th.get("side_strong_b"), side_weak_b=th.get("side_weak_b")))

    if selected_configs is None:
        selected = {(p[0], p[1]): configs[(p, SELECTED_MOTIF)] for p in PAIRS}
    else:
        selected = {tuple(p): c for p, c in selected_configs.items()}
    research = (dict(selected) if research_configs is None
                else {tuple(p): c for p, c in research_configs.items()})
    selected_ids = {p: joint_id(c) for p, c in selected.items()}
    research_ids = {p: joint_id(c) for p, c in research.items()}
    sel_lookup = set(selected_ids.values())
    res_lookup = set(research_ids.values())
    for row in rows:
        row["config_id"] = row.get("config_id") or joint_id(
            configs[(tuple(row["pair"]), row["motif"])])
        row["selected"] = row["config_id"] in sel_lookup
        row["research_consumed"] = row["config_id"] in res_lookup

    # The Pool consumes exactly the selected UU/registered configuration of each pair.
    joint_votes = {}
    for pair, cfg in selected.items():
        row = next(r for r in rows if tuple(r["pair"]) == pair and r["motif"] == cfg.motif)
        joint_votes[pair] = _tri(row.get("vote"), "JOINT_CONFIG_NOT_EVALUATED:" + str(row.get("reason")))
    chosen_pairs = tuple(selected)
    pool_masks, pool_status, pool_reason = {}, {}, {}
    ind_tri = {s: _tri(v, "INDIVIDUAL_VOTE_NOT_EVALUATED") for s, v in individual_votes.items()}
    for op in OPERATORS:
        dec = pool_decision(op, ind_tri, joint_votes, chosen_pairs)
        if dec["execution_status"] != "EXECUTED":
            # undetermined pool: no mask value at all (eligibility ledger), never a third vote
            pool_status[op] = "NOT_EVALUATED"
            if not chosen_pairs:
                pool_reason[op] = "NO_SELECTED_JOINT_CONFIG"
            elif op == "IND_ONLY":
                pool_reason[op] = "INDIVIDUAL_VOTES_NOT_ALL_EVALUABLE"
            else:
                pool_reason[op] = "POOL_NOT_DETERMINED_INSUFFICIENT_INDIVIDUAL_OR_JOINT_EVIDENCE"
        else:
            pool_masks[op] = dec["vote"]
            pool_status[op] = "EVALUATED"
            pool_reason[op] = None
    chosen_pool = selected_pool or SELECTED_POOL
    if chosen_pool not in OPERATORS:
        raise ValueError("UNREGISTERED_POOL_OPERATOR")
    fallback_used = not chosen_pairs
    if fallback_used:
        issued_pool = pool_fallback
        if issued_pool not in OPERATORS:
            raise ValueError("UNREGISTERED_FALLBACK_POOL_OPERATOR")
        selected_pool_vote = pool_masks.get(issued_pool)
        pool_status_selected = "FALLBACK_" + issued_pool + "_NO_SELECTED_PAIR"
        selected_pool_reason = "NO_PAIR_SELECTED_BY_G0_G4_AND_PAIR_SET_SEARCH"
    else:
        issued_pool = chosen_pool
        selected_pool_vote = pool_masks.get(chosen_pool)
        pool_status_selected = pool_status[chosen_pool]
        selected_pool_reason = pool_reason[chosen_pool]

    consumed = pair_consumer_gate(list(selected), list(joint_votes),
                                  mode="SELECTED_POLICY")
    config_ids_selected = {f"{a}-{b}": selected_ids[(a, b)] for a, b in selected}
    config_ids_consumed = {f"{a}-{b}": joint_id(selected[(a, b)]) for a, b in joint_votes}
    config_parity = {
        "status": ("PASS_SELECTED_EQUALS_CONSUMED_CONFIG_ID"
                   if config_ids_selected == config_ids_consumed
                   else "BLOCKED_SELECTED_CONSUMER_MISMATCH"),
        "n_selected_configs": len(config_ids_selected),
        "n_consumed_configs": len(config_ids_consumed),
        "symmetric_difference": sorted(
            set(config_ids_selected.items()) ^ set(config_ids_consumed.items())),
        "identity_includes": "pair,leader,follower,motif,EXACT,lag,delta,unit,profile,release,owner",
    }
    ne_reasons = {}
    for row in rows:
        if row["status"] == "NOT_EVALUATED":
            code = row.get("reason") or "NOT_EVALUATED_UNCLASSIFIED"
            ne_reasons[code] = ne_reasons.get(code, 0) + 1
    packet = {"country": country, "issue": issue, "policy_id": POLICY_ID,
              "gate1_mode": gate1_mode,
              "selected_pool_id": issued_pool,
              "registry_pool_operator": chosen_pool,
              "pool_fallback_used": fallback_used,
              "pool_status": pool_status_selected,
              "pool_reason": selected_pool_reason,
              "selected_motif": (selected[next(iter(selected))].motif if selected
                                 else "NONE"),
              "selected_motifs": {f"{a}-{b}": c.motif for (a, b), c in selected.items()},
              "joint_rows": rows, "pool_masks": pool_masks,
              "pool_status_by_operator": pool_status,
              "pool_reason_by_operator": pool_reason,
              "selected_pool_vote": selected_pool_vote,
              "selected_pool_execution_status": ("EXECUTED" if selected_pool_vote is not None
                                                 else "NOT_EVALUATED"),
              "consumer_parity": consumed, "config_parity": config_parity,
              "selected_config_ids": config_ids_selected,
              "consumed_config_ids": config_ids_consumed,
              "research_config_ids": {f"{a}-{b}": research_ids[(a, b)] for a, b in research},
              "n_selected": len(selected), "n_consumed": len(joint_votes),
              "n_research_consumed": len(research),
              "n_motifs_evaluated": sum(r["status"] in ("PASS", "FAIL") for r in rows),
              "n_motifs_not_evaluated": sum(r["status"] == "NOT_EVALUATED" for r in rows),
              "n_motifs_blocked": sum(r["status"] == "BLOCKED" for r in rows),
              "not_evaluated_reason_counts": ne_reasons,
              "not_evaluated_without_reason": sum(1 for r in rows
                                                  if r["status"] == "NOT_EVALUATED" and not r.get("reason"))}
    packet["sha256"] = hashlib.sha256(json.dumps(packet, sort_keys=True, default=str).encode()).hexdigest()
    return packet


def packet_fingerprint(packet: dict) -> str:
    """Stable fingerprint of issue-time decisions, excluding the self hash."""
    body = {k: v for k, v in packet.items() if k != "sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()
