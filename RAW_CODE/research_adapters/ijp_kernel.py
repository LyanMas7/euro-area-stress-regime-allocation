"""Universal binary Individual -> Joint -> Pool kernel (spec IJP-S5-DESIGN R01, sections B, C, D, G1, H).

One shared mathematical core for RQ01 PRE-ONSET, RQ02 ENTRY and RQ02 EXIT-A.  Votes are
strictly 0/1 for an ELIGIBLE computation.  There is no 'U' vote: when a vote cannot be
legitimately issued the function raises EligibilityError and the caller records
NOT_EVALUATED (+ reason) in its external coverage ledger.  Missing data is never coerced
into 0, into 'N' or into FAIL.

  Count     C_s   = #{a<b : s*(x_b - x_a) > 0}                      M = n(n-1)/2
  Terminal  D_s,a = s*(x_{n-1} - x_a), a = 0..n-2                    B = n-1
  I         = Strong OR (Weak AND Count)
  J         = Gate1 AND TimingValid AND [Strong_J OR (Weak_J AND CountJoint)]
  Pool      six Boolean policies; COMPLEMENT_* are indexed PER PAIR.

This module is the single implementation of triangle / individual / joint / pool; other
adapters must call it instead of duplicating the geometry.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Mapping, Sequence

POLICIES = ('IND_ONLY', 'JOINT_ONLY', 'IND_AND_JOINT', 'IND_OR_JOINT',
            'COMPLEMENT_AND_JOINT', 'COMPLEMENT_OR_JOINT')


class EligibilityError(ValueError):
    """No legitimate numeric vote can be issued for this observation (-> NOT_EVALUATED)."""


@dataclass(frozen=True)
class NativeLevel:
    period_id: str
    period_end: str
    available_at: str
    value: float


def native_asof(rows: Sequence[NativeLevel], issue: str, start: str, end: str) -> list[NativeLevel]:
    """Distinct native periods with period_end in [start, end] and available_at <= issue.

    ISO date strings compare lexicographically.  Duplicate period ids (two vintages of
    one period) are refused: the caller must choose the vintage, never this function.
    A malformed/missing period or release field is ineligible, never a silent zero.
    """
    if not (isinstance(start, str) and isinstance(end, str) and isinstance(issue, str)
            and start and end and issue and start <= end <= issue):
        raise EligibilityError('invalid segment/issue clock')
    clean = []
    for r in rows:
        if not getattr(r, 'period_id', None) or not getattr(r, 'period_end', None) \
                or not getattr(r, 'available_at', None):
            raise EligibilityError('MALFORMED_NATIVE_RECORD')
        try:
            value = float(r.value)
        except (TypeError, ValueError) as exc:
            raise EligibilityError('MALFORMED_NATIVE_VALUE') from exc
        if not math.isfinite(value):
            raise EligibilityError('NONFINITE_NATIVE_VALUE')
        clean.append(r)
    eligible = [r for r in clean if start <= r.period_end <= end and r.available_at <= issue]
    eligible.sort(key=lambda x: (x.period_end, x.period_id))
    ids = [r.period_id for r in eligible]
    if len(ids) != len(set(ids)):
        raise EligibilityError('duplicate native periods; audit vintage choice')
    return eligible


def triangle(levels: Sequence[float]) -> dict:
    """All chronological pairs (Count) plus the terminal column to the last level."""
    if len(levels) < 2:
        raise EligibilityError('requires >= 2 genuine native levels')
    vals = [float(x) for x in levels]
    if not all(math.isfinite(x) for x in vals):
        raise EligibilityError('nonfinite level')
    pairs = [(a, b, vals[b] - vals[a]) for a, b in combinations(range(len(vals)), 2)]
    up = sum(d > 0 for _, _, d in pairs)
    down = sum(d < 0 for _, _, d in pairs)
    flat = sum(d == 0 for _, _, d in pairs)
    n = len(vals)
    if up + down + flat != n * (n - 1) // 2:
        raise AssertionError('TRIANGLE_COUNT_INVARIANT')
    terminal = [vals[-1] - v for v in vals[:-1]]
    return dict(n=n, pairs=pairs, up=up, down=down, flat=flat, terminal=terminal)


def _thresholds(x, n: int, name: str) -> tuple[float, ...]:
    if isinstance(x, (float, int)) and not isinstance(x, bool):
        return (float(x),) * n
    x = tuple(float(z) for z in x)
    if len(x) != n:
        raise EligibilityError(f'{name}: expected {n} thresholds')
    return x


def _strong_weak_count(term: Sequence[float], signed_count: int, sign: int, theta_hi, theta_lo,
                       min_bases: int, count_k: int) -> dict:
    if sign not in (-1, 1):
        raise ValueError('sign must be -1 or +1')
    b = len(term)
    hi = _thresholds(theta_hi, b, 'strong')
    lo = _thresholds(theta_lo, b, 'weak')
    if any(h < 0 or l < 0 or l > h for h, l in zip(hi, lo)):
        raise EligibilityError('must satisfy 0 <= weak <= strong')
    if not (1 <= min_bases <= b):
        raise EligibilityError('infeasible base minimum')
    if not (0 <= count_k <= b * (b + 1) // 2):
        raise EligibilityError('infeasible count threshold')
    oriented = [sign * d for d in term]
    b_hi = sum(d > 0 and d >= h for d, h in zip(oriented, hi))
    b_lo = sum(d > 0 and d >= l for d, l in zip(oriented, lo))
    strong, weak = b_hi >= min_bases, b_lo >= min_bases
    count = signed_count >= count_k
    return dict(vote=int(strong or (weak and count)), strong=int(strong), weak=int(weak),
                count=int(count), B_hi=b_hi, B_lo=b_lo, signed_count=signed_count)


def individual(levels: Sequence[float], sign: int, *, theta_hi, theta_lo, min_bases: int,
               count_k: int) -> dict:
    """I_{j,s} = Strong OR (Weak AND Count), in {0,1}."""
    m = triangle(levels)
    count = m['up'] if sign == 1 else m['down']
    return {**m, **_strong_weak_count(m['terminal'], count, sign, theta_hi, theta_lo,
                                      min_bases, count_k), 'sign': sign}


def joint(matched_levels: Sequence[tuple[float, float]], signs: tuple[int, int], *, high_a, high_b,
          low_a, low_b, gate1_a: float, gate1_b: float, min_bases: int, count_k: int,
          timing_valid: bool) -> dict:
    """Paired native movement Gate1 AND Timing AND [Strong_J OR (Weak_J AND CountJoint)].

    `matched_levels` must already be genuine, distinct, released native anchors.  The paired
    Count is taken over the SAME (a,b) anchor pairs of both sources; it is never a sum or a
    minimum of the two solo Counts, and the result is never I_j AND I_k.
    """
    if len(matched_levels) < 2:
        raise EligibilityError('joint requires >=2 matched native anchors')
    if any(s not in (-1, 1) for s in signs):
        raise ValueError('invalid joint signed mode')
    va = [float(z[0]) for z in matched_levels]
    vb = [float(z[1]) for z in matched_levels]
    a_tri, b_tri = triangle(va), triangle(vb)
    ta = [signs[0] * d for d in a_tri['terminal']]
    tb = [signs[1] * d for d in b_tri['terminal']]
    na = len(ta)
    ha, hb = _thresholds(high_a, na, 'high_a'), _thresholds(high_b, na, 'high_b')
    la, lb = _thresholds(low_a, na, 'low_a'), _thresholds(low_b, na, 'low_b')
    if any(not (0 <= lx <= hx) for hx, lx in [*zip(ha, la), *zip(hb, lb)]):
        raise EligibilityError('invalid paired weak/high thresholds')
    if min_bases < 1 or min_bases > na:
        raise EligibilityError('joint base count out of domain')
    if not (0 <= count_k <= na * (na + 1) // 2):
        raise EligibilityError('infeasible paired count threshold')
    paired_count = sum(signs[0] * da > 0 and signs[1] * db > 0
                       for (_, _, da), (_, _, db) in zip(a_tri['pairs'], b_tri['pairs']))
    both_hi = sum(a > 0 and b > 0 and a >= h1 and b >= h2 for a, b, h1, h2 in zip(ta, tb, ha, hb))
    both_lo = sum(a > 0 and b > 0 and a >= l1 and b >= l2 for a, b, l1, l2 in zip(ta, tb, la, lb))
    gate1 = any(a > 0 and b > 0 and a >= gate1_a and b >= gate1_b for a, b in zip(ta, tb))
    gate2 = both_hi >= min_bases or (both_lo >= min_bases and paired_count >= count_k)
    return dict(vote=int(gate1 and gate2 and bool(timing_valid)), gate1=int(gate1), gate2=int(gate2),
                timing_valid=int(bool(timing_valid)), paired_count=paired_count, paired_hi=both_hi,
                paired_lo=both_lo, n=len(va), signs=signs)


def _k_or(xs):
    if any(x is True for x in xs):
        return True
    return None if any(x is None for x in xs) else False


def _k_and(xs):
    if any(x is False for x in xs):
        return False
    return None if any(x is None for x in xs) else True


def pool(individuals: Mapping[str, int], joint_votes: Mapping[tuple[str, str], int], policy: str,
         *, registry: Sequence[str] | None = None) -> int:
    """One binary decision for already-ELIGIBLE components; nothing is imputed.

    COMPLEMENT_* are PAIR-indexed (spec D2):
        COMPLEMENT_AND_JOINT = OR_pair [ J_pair AND OR_{j not in pair} I_j ]
        COMPLEMENT_OR_JOINT  = OR_pair [ J_pair OR  OR_{j not in pair} I_j ]
    Never one global 'sources outside every pair' set (that would hand a pair the support of a
    source that already belongs to another pair).

    `registry` (optional) lists every source the module owns.  A registry source missing from
    `individuals` is INELIGIBLE (unknown): the decision is derived with Kleene logic and
    EligibilityError is raised only when it cannot be derived from the eligible components.
    Without `registry` an absent source is simply out of scope (reference behaviour).
    """
    if policy not in POLICIES:
        raise ValueError('unregistered policy')
    if not individuals and not joint_votes:
        raise EligibilityError('no evaluable components')
    if any(z not in (0, 1) for z in individuals.values()):
        raise EligibilityError('individual votes must be binary')
    if any(z not in (0, 1) for z in joint_votes.values()):
        raise EligibilityError('joint votes must be binary')
    known = set(registry) if registry is not None else set(individuals)
    if any(a not in known or b not in known for a, b in joint_votes):
        raise EligibilityError('every joint member must be in the source registry')
    unknown_sources = [] if registry is None else [s for s in registry if s not in individuals]
    i_terms = [bool(v) for v in individuals.values()] + [None] * len(unknown_sources)
    i_any = _k_or(i_terms) if i_terms else None
    j_any = _k_or([bool(v) for v in joint_votes.values()]) if joint_votes else (
        False if registry is None else None)
    if policy == 'IND_ONLY':
        out = i_any
    elif policy == 'JOINT_ONLY':
        out = j_any
    elif policy == 'IND_AND_JOINT':
        out = _k_and([i_any, j_any])
    elif policy == 'IND_OR_JOINT':
        out = _k_or([i_any, j_any])
    else:
        terms = []
        for pair, z in joint_votes.items():
            outside = ([bool(q) for s, q in individuals.items() if s not in pair]
                       + [None for s in unknown_sources if s not in pair])
            comp = _k_or(outside) if outside else False
            terms.append(_k_and([bool(z), comp]) if policy == 'COMPLEMENT_AND_JOINT' else _k_or([bool(z), comp]))
        out = _k_or(terms) if terms else False
    if out is None:
        raise EligibilityError('POOL_DECISION_NOT_DERIVABLE_FROM_ELIGIBLE_COMPONENTS')
    return int(out)


def nominal_budget() -> dict:
    """RQ02 Pre-Onset nominal module-stage-sign budget (spec H4-H6): 9,175,534 < 10,000,000."""
    out = {}
    for stage, kl, km, kq in (('S1/H12', 9, 80, 80), ('S2/H10', 27, 80, 27),
                              ('S3/H8', 80, 80, 27), ('S4/H6', 80, 80, 27)):
        il = 2 * 2 * kl
        is_ = 2 * (4 * km + 3 * kq)
        jl = 2 * kl * kl * 12
        js = 2 * 12 * (6 * km * km + 12 * km * kq + 3 * kq * kq)
        po = 2 * 28 * 28 * 3
        out[stage] = (il, is_, jl, js, po, il + is_ + jl + js + po)
    il, is_, jl, js, po = 0, 2 * 7 * 40, 0, 2 * 21 * 40 * 40 * 12, 2 * 28 * 28 * 3
    out['S5/RETROSPECTIVE'] = (il, is_, jl, js, po, il + is_ + jl + js + po)
    if [v[-1] for v in out.values()] != [3233404, 1619278, 1755594, 1755594, 811664]:
        raise AssertionError('STAGE_BUDGET_MISMATCH')
    if sum(v[-1] for v in out.values()) != 9175534:
        raise AssertionError('TOTAL_BUDGET_MISMATCH')
    return out


def rq01_archive_budget() -> int:
    """Archived RQ01 five-source/six-policy family (spec H7): (6^5*5)*(3*2*3*6) = 4,199,040."""
    return (6 ** 5 * 5) * (3 * 2 * 3 * 6)


def geometry(n: int) -> tuple[int, int]:
    """(Count pairs, Terminal bases) for n genuine native levels: (n(n-1)/2, n-1)."""
    if type(n) is not int or n < 2:
        raise EligibilityError('requires >= 2 genuine native levels')
    return n * (n - 1) // 2, n - 1
