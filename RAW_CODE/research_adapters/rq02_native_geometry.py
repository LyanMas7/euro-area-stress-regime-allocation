"""RQ02 entry/exit two-target, three-operational-phase, math-first core.

This package is a compositional function specification. It does NOT modify or
replace native REV290/RQ01 models. No learned coefficients, source thresholds,
PIT releases, issuer evidence, forecasts, or regulatory approval are asserted.
Only native raw observations are allowed as feature inputs; retrospective labels
are evaluation-only. No OOS is read.

Canonical Individual / Joint / Pool arithmetic is research_adapters/ijp_kernel.py and its RQ02 application
(horizons, stages, pair groups, ENTRY roles, EXIT-A, coverage ledger) is research_adapters/rq02_universal.py.
The first-hit style helpers below (``magnitude_first_gate``, ``source_vote``, ``individual_count``,
``joint_pair``, ``joint_count``) are LEGACY first-hit constructs kept for the transfer registry: they are not the
specified Strong OR (Weak AND Count) / paired-Joint functions.  GLOBAL NO-U: their outcomes are a binary `pass`
or a non-vote status (NOT_EVALUATED / BLOCKED + reason); a non-evaluable input is never a 0 and never 'U'.
ENTRY horizon names count QUARTERS (H15 = T-45 months); EXIT horizons are months.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import date, datetime
from itertools import combinations, product
from math import isfinite, ceil
from typing import Any, Iterable, Mapping, Sequence
import calendar

from .ijp_kernel import EligibilityError
from .vote_contract import NonVote, decide_and, decide_or, is_nonvote, not_evaluated

SOURCES = ("CG", "DSR-P", "CISS", "CLIFS", "DFC", "GFC", "VIX")
EA7 = ("BE", "DE", "ES", "FI", "FR", "IT", "NL")
# ENTRY horizon NAMES count QUARTERS: H15 = 15 quarters = T-45 months (spec IJP E2: T-45/-36/-30/-24/-18).
H_ONSET = (15, 12, 10, 8, 6)
H_ONSET_MONTHS = {15: 45, 12: 36, 10: 30, 8: 24, 6: 18}
# EXIT horizons are already MONTHS (18/12/6/3).
H_EXIT = (18, 12, 6, 3)
OP_PHASES = ("PRE_ONSET", "ENTRY", "EXIT")
ACTIVE_TARGETS = ("ONSET", "EXIT")
VARIANTS = {
    "M00": frozenset(("B",)),
    "M01": frozenset(("B", "A", "I")),
    "M10": frozenset(("B", "A", "J")),
    "M11": frozenset(("B", "A", "I", "J")),
}


def day(value: Any) -> date:
    if isinstance(value, datetime): return value.date()
    if isinstance(value, date): return value
    if not isinstance(value, str): raise TypeError("date required")
    return date.fromisoformat(value)


def within_dev(d: Any) -> bool:
    return date(2005, 1, 1) <= day(d) <= date(2019, 12, 31)


def _check_dev(d: Any) -> date:
    d = day(d)
    if not within_dev(d): raise ValueError("DEV_ONLY_2005_2019__OOS_SEALED")
    return d


def add_months(t: Any, months: int) -> date:
    if not isinstance(months, int): raise TypeError("integer calendar months")
    t = day(t)
    idx = t.year * 12 + t.month - 1 + months
    y, m0 = divmod(idx, 12)
    m = m0 + 1
    last = calendar.monthrange(y, m)[1]
    # The issue calendar is monthly: a month-END date stays a month-END date (Feb 28 + 1 month = Mar 31), any
    # other day keeps its day number clamped to the target month.
    if t.day == calendar.monthrange(t.year, t.month)[1]:
        return date(y, m, last)
    return date(y, m, min(t.day, last))


@dataclass(frozen=True)
class NativePoint:
    source: str
    country: str | None
    native_period: str
    period_end: date
    available_at: date
    value: float
    raw_unit: str
    source_sha256: str
    released_verified: bool
    first_issued_verified: bool

    def __post_init__(self) -> None:
        if self.source not in SOURCES: raise ValueError("UNKNOWN_7EWI")
        if self.country is not None and self.country not in EA7:
            raise ValueError("INVALID_RQ02_COUNTRY")
        if not self.native_period or not self.raw_unit: raise ValueError("NATIVE_UNIT_REQUIRED")
        if not isinstance(self.value, (float, int)) or isinstance(self.value, bool) or not isfinite(self.value):
            raise ValueError("FINITE_RAW_REQUIRED")
        if len(self.source_sha256) != 64: raise ValueError("SOURCE_SHA256_REQUIRED")
        if day(self.period_end) > day(self.available_at): raise ValueError("OBSERVATION_AFTER_AVAILABILITY")
        if self.source in ("CG", "DSR-P", "DFC"):
            import re
            match = re.fullmatch(r"(\d{4})Q([1-4])", self.native_period)
            if match is None: raise ValueError("QUARTERLY_NATIVE_IDENTIFIER_REQUIRED")
            y, q = int(match.group(1)), int(match.group(2))
            m = 3 * q
            if day(self.period_end) != date(y, m, calendar.monthrange(y, m)[1]):
                raise ValueError("NATIVE_QUARTER_PERIOD_MISMATCH")


def latest_released_native(points: Iterable[NativePoint], issue_at: Any,
                           *, source: str, country: str | None) -> tuple[NativePoint, ...]:
    """Return one issue-valid value per DISTINCT native period (never carried months)."""
    issue = _check_dev(issue_at)
    if source not in SOURCES: raise ValueError("UNKNOWN_7EWI")
    valid: dict[str, NativePoint] = {}
    for p in points:
        if p.source != source or p.country != country: continue
        if not (p.released_verified and p.first_issued_verified): continue
        if day(p.available_at) > issue: continue
        # For actual revision tapes, the latest release known *at issue* is selected.
        old = valid.get(p.native_period)
        if old is None or p.available_at > old.available_at:
            valid[p.native_period] = p
    return tuple(sorted(valid.values(), key=lambda p: (p.period_end, p.available_at)))


def native_readiness(points: Sequence[NativePoint], issue_at: Any, *, min_distinct: int = 2) -> dict:
    issue = _check_dev(issue_at)
    if not points: return {"status": "NOT_EVALUATED", "reason": "NO_NATIVE_SOURCE", "n_native": 0}
    distinct = {p.native_period for p in points if p.available_at <= issue}
    if any(not p.released_verified or not p.first_issued_verified for p in points):
        return {"status": "NOT_EVALUATED", "reason": "ISSUE_VINTAGE_UNVERIFIED", "n_native": len(distinct)}
    if any(p.available_at > issue for p in points):
        return {"status": "NOT_EVALUATED", "reason": "FUTURE_SOURCE_AT_ISSUE", "n_native": len(distinct)}
    if len(distinct) < min_distinct:
        return {"status": "NOT_EVALUATED", "reason": "INSUFFICIENT_DISTINCT_NATIVE", "n_native": len(distinct)}
    if len({p.raw_unit for p in points}) > 1:
        raise ValueError("RAW_UNIT_MISMATCH")
    return {"status": "READY", "n_native": len(distinct), "source": points[0].source}


def horizon_issue(target: Any, horizon_months: int) -> date:
    if horizon_months < 0: raise ValueError("NEGATIVE_HORIZON")
    return add_months(day(target), -horizon_months)


def pre_onset_anchor_map(target: Any) -> dict:
    """Retrospective target scheduling only; NEVER use target date as feature at issue."""
    target = day(target)
    # H-names are QUARTERS: the issue of H15 is T-45 months (not T-15).
    issues = {f"H{h}": horizon_issue(target, H_ONSET_MONTHS[h]) for h in H_ONSET}
    return {"phase": "PRE_ONSET", "target": target.isoformat(),
            "issued_anchors": {k: v.isoformat() for k, v in issues.items()},
            "months_before_target": {f"H{h}": H_ONSET_MONTHS[h] for h in H_ONSET},
            "progressive_confirmations": {f"H{h}": i for i, h in enumerate(H_ONSET)},
            # H15 carries no cross-horizon confirmation: NOT_APPLICABLE, never an Individual Long = 0.
            "cross_horizon_confirmation": {f"H{h}": ("NOT_APPLICABLE" if i == 0 else "APPLICABLE")
                                           for i, h in enumerate(H_ONSET)},
            "research_only_calendar_template": True}


def long_prefix_pairs(n_anchors: int) -> dict:
    """Long anchor geometry: n anchors -> n(n-1)/2 Count pairs, n-1 Terminal bases (n = 1..5 = H15..H6)."""
    if n_anchors not in range(1, 6): raise ValueError("1_TO_5_LONG_ANCHORS")
    return {"n_anchors": n_anchors, "n_triangles": n_anchors * (n_anchors - 1) // 2,
            "n_terminals": n_anchors - 1, "n_adjacent_confirmation": n_anchors - 1,
            "n_confirmations": n_anchors - 1,
            "cross_horizon_confirmation": "NOT_APPLICABLE" if n_anchors == 1 else "APPLICABLE",
            "pairs": tuple(combinations(range(n_anchors), 2))}


def short_native_geometry(n_levels: int) -> dict:
    """Short-stage geometry for n genuine native levels.  n < 2 cannot form a Count: NOT_EVALUATED upstream,
    never a FAIL (EligibilityError is a ValueError, so legacy callers keep working)."""
    if n_levels < 2: raise EligibilityError("REQUIRES_TWO_NATIVE_LEVELS")
    return {"n_native_levels": n_levels, "n_adjacent_moves": n_levels - 1,
            "n_triangles": n_levels * (n_levels - 1) // 2,
            "n_terminal_bases": n_levels - 1,
            "pairs": tuple(combinations(range(n_levels), 2))}


def signed_native_geometry(points: Sequence[NativePoint], issue_at: Any) -> dict:
    """All native pairs and terminal-from-each-base; no theta is read or selected."""
    issue = _check_dev(issue_at)
    q = native_readiness(points, issue)
    if q["status"] != "READY": return {"status": "NOT_EVALUATED", "reason": q["reason"]}
    if any(points[i].period_end >= points[i+1].period_end for i in range(len(points)-1)):
        raise ValueError("NATIVE_PERIOD_NOT_STRICTLY_INCREASING")
    if len({p.source for p in points}) != 1 or len({p.country for p in points}) != 1:
        raise ValueError("SOURCE_AND_COUNTRY_MUST_MATCH")
    rows = []
    for a, b in combinations(range(len(points)), 2):
        delta = float(points[b].value) - float(points[a].value)
        rows.append({"base": a, "end": b, "delta_raw": delta,
                     "abs_magnitude_raw": abs(delta), "positive_excursion": max(delta, 0.0),
                     "negative_excursion": max(-delta, 0.0),
                     "is_adjacent": b == a + 1, "is_terminal": b == len(points) - 1,
                     "raw_unit": points[a].raw_unit,
                     "start_native_period": points[a].native_period,
                     "end_native_period": points[b].native_period})
    return {"status": "FEATURE_GEOMETRY_ONLY", "source": points[0].source,
            "country": points[0].country, "native_len": len(points),
            "pair_count": len(rows), "distinct_bases": len(points) - 1,
            "available_at_max": max(p.available_at for p in points).isoformat(),
            "geometry": rows}


def three_state(delta_raw: float, flat_epsilon_raw: float | None):
    """UP / DOWN / FLAT movement; None when the flat epsilon is not registered (not evaluable, never 'U')."""
    if flat_epsilon_raw is None: return None
    if not isfinite(float(delta_raw)) or not isfinite(float(flat_epsilon_raw)) or flat_epsilon_raw < 0:
        raise ValueError("INVALID_RAW_DELTA_OR_EPSILON")
    if abs(delta_raw) <= flat_epsilon_raw: return "FLAT"
    return "UP" if delta_raw > 0 else "DOWN"


def nine_pattern(a: float, b: float, epsilon_a: float | None, epsilon_b: float | None) -> str:
    state_a, state_b = three_state(a, epsilon_a), three_state(b, epsilon_b)
    if state_a is None or state_b is None: return None          # pattern not evaluable
    return f"{state_a}_{state_b}"


def terminal_path(moves: Sequence[float], *, risk_sign: int | None,
                  flat_epsilon_raw: float | None) -> dict:
    """Native adjacent moves, NOT carried periods; terminal=sum moves telescopes."""
    if not moves or risk_sign not in (-1, 1) or flat_epsilon_raw is None:
        return {"status": "NOT_EVALUATED", "reason": "UNIDENTIFIED_SIGN_OR_NATIVE_MOVES"}
    labels = [three_state(x, flat_epsilon_raw) for x in moves]
    if None in labels: return {"status": "NOT_EVALUATED", "reason": "MOVEMENT_STATE_NOT_EVALUABLE"}
    terminal = float(sum(moves))
    gross = float(sum(abs(x) for x in moves))
    risk_up = sum(risk_sign * x > flat_epsilon_raw for x in moves)
    risk_down = sum(risk_sign * x < -flat_epsilon_raw for x in moves)
    return {"status": "FEATURE_GEOMETRY_ONLY", "n_native_edges": len(moves),
            "up_steps": labels.count("UP"), "down_steps": labels.count("DOWN"),
            "flat_steps": labels.count("FLAT"),
            "risk_direction_steps": risk_up, "opposite_steps": risk_down,
            "terminal_magnitude_raw": terminal, "risk_signed_terminal_raw": risk_sign * terminal,
            "gross_magnitude_raw": gross, "coherence": abs(terminal)/gross if gross else None}


def magnitude_first_gate(geometry: Mapping, *, min_magnitude_raw: float | None,
                         risk_sign: int | None, min_base_support: int | None,
                         min_native_persistence: int | None) -> dict:
    """Explicit gate A (magnitude), base support and one-sided persistence."""
    if geometry.get("status") != "FEATURE_GEOMETRY_ONLY":
        return {"status": "NOT_EVALUATED", "reason": "GEOMETRY_UNAVAILABLE"}
    if any(v is None for v in (min_magnitude_raw, risk_sign, min_base_support, min_native_persistence)):
        return {"status": "BLOCKED", "reason": "THRESHOLD_NOT_REGISTERED"}
    if (risk_sign not in (-1, 1) or min_magnitude_raw < 0 or min_base_support < 1
            or min_native_persistence < 1): raise ValueError("INVALID_THRESHOLD_DOMAIN")
    pairs = geometry["geometry"]
    # Each base contributes at most one vote, regardless of overlapping b endpoints.
    qualifying_bases = set()
    hits = {}
    for row in pairs:
        if risk_sign * row["delta_raw"] >= min_magnitude_raw:
            qualifying_bases.add(row["base"])
            hits.setdefault(row["base"], row["end"])
    # Native observation persistence is per-base subsequent native endpoint support;
    # magnitudes are from the SAME base (not repeated deltas of adjacent periods).
    base_persistent = set()
    for a, first_end in hits.items():
        successors = sorted((r for r in pairs if r["base"] == a and r["end"] >= first_end), key=lambda r: r["end"])
        run = 0
        for row in successors:
            if risk_sign * row["delta_raw"] < min_magnitude_raw: break
            run += 1
        if run >= min_native_persistence: base_persistent.add(a)
    return {"status": "EVALUATED_FOR_GIVEN_THETA_ONLY", "pass": len(base_persistent) >= min_base_support,
            "first_hit_base_indices": hits, "qualified_base_count": len(qualifying_bases),
            "persistently_qualified_bases": len(base_persistent),
            "n_eligible_distinct_bases": geometry["distinct_bases"],
            "n_qualifying_pairs": sum(risk_sign * row["delta_raw"] >= min_magnitude_raw for row in pairs),
            "earliest_native_end_index": min(hits.values()) if hits else None}


def source_vote(long_gate: Mapping, short_gate: Mapping,
                *, conjunction: str | None) -> dict:
    """One EWI -> <=1 vote; aggregation operator is never silently invented.

    Partial-input determination (work order 2.4): OR is PASS on one evaluated PASS child, AND is FAIL on one
    evaluated FAIL child; otherwise a missing child gate leaves the vote NOT_EVALUATED (never 0)."""
    if conjunction not in ("AND", "OR"):
        return {"status": "BLOCKED", "reason": "SOURCE_LONG_SHORT_OPERATOR_UNIDENTIFIED"}
    def child(g):
        return (("PASS" if g["pass"] else "FAIL") if g.get("status") == "EVALUATED_FOR_GIVEN_THETA_ONLY"
                else not_evaluated("MISSING_CHILD_GATE"))
    a, b = child(long_gate), child(short_gate)
    d = decide_and(a, b) if conjunction == "AND" else decide_or(a, b)
    if is_nonvote(d):
        return {"status": "NOT_EVALUATED", "reason": "MISSING_CHILD_GATE"}
    return {"status": "EVALUATED_FOR_GIVEN_THETA_ONLY", "pass": d == "PASS",
            "long_pass": None if is_nonvote(a) else a == "PASS",
            "short_pass": None if is_nonvote(b) else b == "PASS", "operator": conjunction}


def individual_count(votes: Mapping[str, Mapping], *, k_min: int | None) -> dict:
    if set(votes) - set(SOURCES): raise ValueError("UNKNOWN_EWI_IN_COUNT")
    eligible = [v.get("pass") for v in votes.values() if v.get("status") == "EVALUATED_FOR_GIVEN_THETA_ONLY"]
    unknown = len(eligible) != len(votes)
    lower = sum(v is True for v in eligible)
    if k_min is None: return {"status": "BLOCKED", "reason": "K_MIN_NOT_REGISTERED", "lower_bound": lower}
    if not 0 <= k_min <= 7: raise ValueError("INVALID_K_MIN")
    if lower >= k_min: decision = True
    elif lower + (len(votes)-len(eligible)) < k_min: decision = False
    else: return {"status": "NOT_EVALUATED", "reason": "DECISION_NOT_DETERMINED_FROM_AVAILABLE_INPUTS",
                  "lower_bound": lower}
    return {"status": "EVALUATED_FOR_GIVEN_THETA_ONLY", "pass": decision,
            "k_real_lower_bound": lower, "n_known_sources": len(eligible), "n_registered_sources": len(votes),
            "not_evaluated_present": unknown}


def all_joint_source_pairs() -> tuple[tuple[str, str], ...]:
    return tuple(combinations(SOURCES, 2))


def joint_pair(first_hits: Mapping[str, Mapping], source_a: str, source_b: str,
               *, sign_mode: tuple[int,int] | None, max_lag_days: int | None,
               require_individual_confirmed: bool | None) -> dict:
    """Two independent source *names*; shared data-driver independence NOT assumed."""
    if source_a == source_b or source_a not in SOURCES or source_b not in SOURCES:
        raise ValueError("TWO_DISTINCT_EWI_REQUIRED")
    if sign_mode not in ((1,1),(-1,-1),(1,-1),(-1,1)) or max_lag_days is None or require_individual_confirmed is None:
        return {"status": "BLOCKED", "reason": "JOINT_POLICY_NOT_IDENTIFIED"}
    if max_lag_days < 0: raise ValueError("NEGATIVE_LAG")
    a,b=first_hits.get(source_a),first_hits.get(source_b)
    if not a or not b or a.get("status") != "EVALUATED_FOR_GIVEN_THETA_ONLY" or b.get("status") != "EVALUATED_FOR_GIVEN_THETA_ONLY":
        return {"status": "NOT_EVALUATED", "reason": "SOURCE_FIRSTHIT_UNVERIFIED"}
    for p in (a,b):
        if p.get("sign") not in (-1,1) or p.get("crossing_at") is None:
            return {"status": "NOT_EVALUATED", "reason": "NATIVE_CROSSING_UNAVAILABLE"}
        if require_individual_confirmed and p.get("individual_confirmed") is not True:
            return {"status": "NOT_EVALUATED", "reason": "INDIVIDUAL_PREREQUISITE_UNMET"}
    lag = (day(b["crossing_at"])-day(a["crossing_at"])).days
    return {"status": "EVALUATED_FOR_GIVEN_THETA_ONLY", "pass":
            a["sign"] == sign_mode[0] and b["sign"] == sign_mode[1] and abs(lag)<=max_lag_days,
            "signed_lag_days":lag,"shared_driver_independence": "NOT_IDENTIFIABLE"}


def joint_count(pairs: Mapping[tuple[str,str], Mapping], *, j_min: int | None) -> dict:
    if any((a == b or a not in SOURCES or b not in SOURCES) for a,b in pairs):
        raise ValueError("UNKNOWN_JOINT_PAIR")
    if j_min is None: return {"status": "BLOCKED", "reason": "J_MIN_NOT_REGISTERED"}
    if j_min < 0 or j_min > 21: raise ValueError("INVALID_J_MIN")
    known = [p["pass"] for p in pairs.values() if p.get("status") == "EVALUATED_FOR_GIVEN_THETA_ONLY"]
    true = sum(v is True for v in known)
    if true >= j_min: decision = True
    elif true + len(pairs) - len(known) < j_min: decision = False
    else: return {"status": "NOT_EVALUATED", "reason": "DECISION_NOT_DETERMINED_FROM_AVAILABLE_INPUTS",
                  "lower_bound": true}
    return {"status": "EVALUATED_FOR_GIVEN_THETA_ONLY", "pass": decision,
            "j_real_lower_bound": true, "n_evaluated_pairs": len(known), "n_total_pairs": len(pairs),
            "not_independent_pair_votes": True}


def model_membership(model_id: str) -> dict:
    if model_id not in VARIANTS: raise ValueError("UNKNOWN_FOUR_M_MODEL")
    return {"model": model_id, "enabled_nodes": sorted(VARIANTS[model_id]),
            "shared_accumulation_once": model_id in ("M01", "M10", "M11"),
            "i_internal_only_for_joint": model_id == "M10",
            "original_operator_required": True}


def _graph_in(x):
    if x is None:
        return not_evaluated("GRAPH_NODE_NOT_EVALUATED")
    if x is True or x is False:
        return "PASS" if x else "FAIL"
    raise ValueError("GRAPH_NODE_MUST_BE_BOOL_OR_NOT_EVALUATED")


def tri_and(*xs):
    """AND over graph nodes (bool, or None = node not evaluated); True/False or a NonVote (no decision)."""
    d = decide_and(*[_graph_in(x) for x in xs])
    return d if is_nonvote(d) else d == "PASS"


def tri_or(*xs):
    """OR over graph nodes (bool, or None = node not evaluated); True/False or a NonVote (no decision)."""
    d = decide_or(*[_graph_in(x) for x in xs])
    return d if is_nonvote(d) else d == "PASS"


def apply_declared_graph(model: str, *, b: bool | None, a: bool | None,
                         i: bool | None, j: bool | None,
                         operator: str | None) -> dict:
    """Declarative Boolean example; never claims match to original proprietary F_r.

    Removing nodes means omission from graph, NOT replacing RAW measurements by zero.
    Permission to use this prototype as historical baseline requires B/F original parity.
    """
    nodes = VARIANTS.get(model)
    if nodes is None: raise ValueError("UNKNOWN_MODEL")
    if model == "M00":
        if b is None: return {"status": "NOT_EVALUATED", "reason": "BASELINE_NODE_NOT_EVALUATED"}
        return {"status": "GRAPH_PROTOTYPE_ONLY", "decision": bool(b)}
    if operator not in ("AND", "OR"):
        return {"status": "BLOCKED", "reason": "ORIGINAL_GRAPH_OPERATOR_NOT_VERIFIED"}
    vals = [dict(B=b,A=a,I=i,J=j)[key] for key in ("B","A","I","J") if key in nodes]
    d = tri_and(*vals) if operator == "AND" else tri_or(*vals)
    if is_nonvote(d):
        return {"status": "NOT_EVALUATED", "reason": d.reason}
    return {"status": "GRAPH_PROTOTYPE_ONLY", "decision": d}


def pre_onset_function(issue_at: Any, *, horizon: int, source_geometries: Mapping[str,Mapping],
                       baseline_issued: Mapping | None) -> dict:
    """Build identified *inputs*, never infer unknown threshold or historical outcome."""
    issue = _check_dev(issue_at)
    if horizon not in H_ONSET: raise ValueError("ONSET_HORIZON_NOT_DEFINED")
    if set(source_geometries)-set(SOURCES): raise ValueError("EXTRA_SOURCE")
    for source, geometry in source_geometries.items():
        if geometry.get("status") != "FEATURE_GEOMETRY_ONLY":
            return {"status":"NOT_EVALUATED", "reason":"SOURCE_FEATURE_NOT_VERIFIED", "source":source}
        if not geometry.get("available_at_max"):
            return {"status":"NOT_EVALUATED", "reason":"SOURCE_ISSUE_CLOCK_UNAVAILABLE", "source":source}
        if day(geometry["available_at_max"]) > issue:
            raise ValueError("FUTURE_SOURCE_GEOMETRY_LEAK")
    if not baseline_issued or not baseline_issued.get("issue_verified") or day(baseline_issued.get("available_at","9999-01-01")) > issue:
        return {"status": "NOT_EVALUATED", "reason": "NATIVE_ONSET_BASELINE_NOT_ISSUED"}
    return {"status": "FEATURE_READY_MODEL_NOT_FITTED", "phase": "PRE_ONSET", "track": "ONSET",
            "issue_at": issue.isoformat(),
            "target_at": add_months(issue, H_ONSET_MONTHS[horizon]).isoformat(),   # H-names are quarters
            "horizon_months": H_ONSET_MONTHS[horizon],
            "sources_represented": sorted(source_geometries),"threshold_decision": None}


def entry_firsthit_function(issue_at: Any, *, onset_signal: Mapping | None,
                            previously_issued_state: str | None) -> dict:
    issue = _check_dev(issue_at)
    # previously_issued_state: 'N' / 'STRESS', or None when no state was issued (never a 'U' class)
    if previously_issued_state == "U":
        raise ValueError("LEGACY_U_ISSUED_STATE_REJECTED_USE_NONE")
    if previously_issued_state not in ("N", "STRESS", None):
        return {"status":"NOT_EVALUATED","reason":"ISSUED_STATE_REQUIRED"}
    if not onset_signal or onset_signal.get("issue_at") is None:
        return {"status":"NOT_EVALUATED","reason":"NO_ISSUED_ONSET_GATE"}
    if day(onset_signal["issue_at"]) > issue:
        raise ValueError("ONSET_SIGNAL_FROM_FUTURE")
    if previously_issued_state is None:
        return {"status":"NOT_EVALUATED","reason":"NO_PREVIOUSLY_ISSUED_STATE"}
    if previously_issued_state != "N":
        return {"status":"NOT_APPLICABLE","reason":"ENTRY_ONLY_WHEN_PREV_ISSUED_N"}
    signal = onset_signal.get("signal")
    if signal not in (True, False):return {"status":"NOT_EVALUATED","reason":"ONSET_DECISION_NOT_ISSUED"}
    return {"status":"EVALUATED_FOR_GIVEN_SIGNAL_ONLY", "entry":signal,
            "issue_at":issue.isoformat()}


def exit_first_n_target(issue_at: Any, *, horizon_months: int,
                        issued_stress_at_issue: bool | None) -> dict:
    issue = _check_dev(issue_at)
    if horizon_months not in H_EXIT:raise ValueError("EXIT_HORIZON_NOT_IN_18_12_6_3")
    if issued_stress_at_issue is not True:
        return {"status":"NOT_EVALUATED","reason":"STRESS_RISK_SET_NOT_ISSUED_VERIFIED"}
    through = add_months(issue, horizon_months)
    return {"status":"TARGET_DEFINITION_ONLY", "issue_at":issue.isoformat(),
            "through":through.isoformat(), "target_month": through.strftime("%Y-%m"),
            # PRIMARY (spec RQ02 EXIT 1.3): the first N falls in EXACTLY the target month.
            "target":"FIRST_N_EXACT_MONTH",
            "primary_definition":"1[month(first ESRB N after issue) == month(issue + h)]",
            # CONTROL only, never mixed into the same loss table:
            "control_target":"FIRST_N_BY_HORIZON",
            "control_definition":"1[issue < first ESRB N <= issue + h]",
            "future_s4_not_as_feature":True, "exit_has_no_s5_var_es_roundback":True}


def exit_parallel_branches(issue_at: Any, *, onset_confirmed: bool,
                           a_forecast: Mapping | None, b_state: Mapping | None) -> dict:
    issue=_check_dev(issue_at)
    if not onset_confirmed:return {"status":"NOT_EVALUATED","reason":"ISSUED_ONSET_PREREQUISITE_MISSING"}
    a = a_forecast if a_forecast and day(a_forecast.get("issue_at","9999-01-01")) <= issue else None
    b = b_state if b_state and day(b_state.get("issue_at","9999-01-01")) <= issue else None
    if a is None: return {"status":"NOT_EVALUATED","reason":"EXIT_A_ALWAYS_EMITS_REQUIRED"}
    return {"status":"FEATURE_AND_ISSUE_HANDOFF_ONLY", "A":a,"B":b,
            "B_does_not_overwrite_A":True,"state_conditioned_B":b is not None}


def integrated_state_step(*, issue_at: Any, prior_issued: str | None,
                          onset: Mapping | None, exit_signal: Mapping | None) -> dict:
    """One issue clock; no retrodating. This is a deterministic OPERATIONAL policy, not fitted and not an RQ01 label.

    An EXIT forecast PASS never confirms that RQ01's current state is N (spec RQ02 EXIT 6): STRESS -> N needs
    an RQ01-issued, release-verified first-N proof carried in ``exit_signal['rq01_first_n_proof']`` (available
    at or before the issue).  Without it the state stays STRESS and the forecast is reported next to it.
    """
    issue=_check_dev(issue_at)
    if prior_issued not in ("N", "STRESS"):
        return {"status":"NOT_EVALUATED","reason":"PRIOR_ISSUED_STATE_NOT_IDENTIFIED"}
    source=onset if prior_issued=="N" else exit_signal
    if not source or source.get("issue_at") is None or source.get("signal") not in (True,False):
        return {"status":"NOT_EVALUATED","reason":"APPLICABLE_TRANSITION_GATE_NOT_ISSUED"}
    if day(source["issue_at"])>issue:raise ValueError("FUTURE_APPLICABLE_GATE")
    other = exit_signal if prior_issued=="N" else onset
    # No future non-applicable signal can be inserted at old issuance.
    if other and other.get("issue_at") and day(other["issue_at"])>issue:
        raise ValueError("BACKDATED_COMPOSITE_OTHER_GATE")
    value=source["signal"]
    common={"issue_at":issue.isoformat(),"previous_state":prior_issued,
            "used_gate":"ONSET" if prior_issued=="N" else "EXIT","not_certified_prediction":True,
            "state_kind":"OPERATIONAL_POLICY_STATE_NOT_RQ01_LABEL","rq01_label_overwritten":False}
    if prior_issued=="N":
        return {"status":"ISSUED_RULE_ONLY","next_state":"STRESS" if value else "N",**common}
    if not value:
        return {"status":"ISSUED_RULE_ONLY","next_state":"STRESS",**common}
    proof=source.get("rq01_first_n_proof")
    if not (isinstance(proof,dict) and proof.get("confirmed_by")=="RQ01_ISSUED"
            and proof.get("release_verified") is True and proof.get("available_at")):
        return {"status":"ISSUED_RULE_ONLY","next_state":"STRESS","exit_forecast_pass":True,
                "reason":"EXIT_FORECAST_DOES_NOT_CONFIRM_RQ01_FIRST_N",**common}
    if day(proof["available_at"])>issue:raise ValueError("FUTURE_FIRST_N_PROOF")
    return {"status":"ISSUED_RULE_ONLY","next_state":"N","exit_forecast_pass":True,
            "first_n_proof_available_at":day(proof["available_at"]).isoformat(),**common}


def matched_binary_metrics(rows: Sequence[Mapping], *, expected_keys: Sequence[tuple] | None = None) -> dict:
    """Actual issued and matured targets only.

    The population is whatever the CALLER's contract registers in ``expected_keys``; no cardinality is
    hard-coded (EA7 x 180 = 1,260 country-month slots are theoretical slots, NOT automatically valid labels or
    issued forecasts, and no figure is inferred from a counter).  Keys that do not match exactly -> NOT_EVALUATED.
    """
    if expected_keys is None:return {"status":"NOT_EVALUATED","reason":"SIGNED_CANONICAL_KEYS_MISSING"}
    expected=[tuple(k) for k in expected_keys]
    if len(expected)!=len(set(expected)):raise ValueError("DUPLICATE_EXPECTED_KEY")
    if not expected:return {"status":"NOT_EVALUATED","reason":"EMPTY_EXPECTED_POPULATION"}
    ids=[tuple(x["key"]) for x in rows]
    if len(ids)!=len(set(ids)):raise ValueError("DUPLICATE_ISSUE_TARGET_KEY")
    if set(ids)!=set(expected):
        return {"status":"NOT_EVALUATED","reason":"CANONICAL_KEY_COVERAGE_MISMATCH",
                "n_matched":len(set(ids)&set(expected)),"n_expected":len(expected),
                "n_missing":len(set(expected)-set(ids)),"n_unexpected":len(set(ids)-set(expected))}
    for row in rows:
        if row.get("issued_verified") is not True or row.get("label_matured") is not True:
            return {"status":"NOT_EVALUATED","reason":"ISSUE_OR_LABEL_MATURITY_NOT_VERIFIED"}
        if day(row["last_issued_at"])>day(row["issue_at"]):
            raise ValueError("RETROACTIVE_ISSUANCE")
        if row.get("gold") not in ("N","STRESS") or row.get("predicted") not in ("N","STRESS"):
            return {"status":"NOT_EVALUATED","reason":"PREDICTION_NOT_ISSUED_OR_GOLD_UNAVAILABLE"}
    p=sum(x["gold"]=="STRESS" for x in rows)
    n=len(rows)-p
    if p==0 or n==0:return {"status":"UNDEFINED","reason":"ZERO_CLASS_DENOMINATOR"}
    fn=sum(x["gold"]=="STRESS" and x["predicted"]=="N" for x in rows)
    fp=sum(x["gold"]=="N" and x["predicted"]=="STRESS" for x in rows)
    return {"status":"DEV_MATCHED_DIAGNOSTIC_ONLY", "n":len(rows),"P":p,"N":n,
            "FN":fn,"FP":fp,"TP":p-fn,"TN":n-fp,
            "n_expected_keys":len(expected),"population_source":"CALLER_EXPECTED_KEYS",
            "FNR":fn/p,"FPR":fp/n,"FNR_under_5pct":fn/p<0.05,"FPR_under_15pct":fp/n<0.15,
            "max_FN_strict":ceil(.05*p)-1,"max_FP_strict":ceil(.15*n)-1,
            "gate_cutoffs_note":"5%/15% are NOT verified against the File-01 RQ02 targets (other W02 modules use 10%/20%)",
            "PIT_CERTIFIED":False,"OOS_TOUCHED":False}


def feature_contract_registry() -> list[dict]:
    out=[]
    for phase in OP_PHASES:
        for source in SOURCES:
            for mode in ("LONG_POS","LONG_NEG","SHORT_POS","SHORT_NEG"):
                out.append({"id":f"RQ02.{phase}.{source}.{mode}", "phase":phase,"source":source,"mode":mode,
                            "threshold_owner":"UNVERIFIED", "raw_unit":"NATIVE_SOURCE_OWNED",
                            "threshold_value":None, "status":"NOT_FITTED"})
    return out
