"""RQ02 Bottom 3F canonical issue-time functions (spec RQ02 Bottom, parts II-IV and VI).

Three families, one issued signal per (country, asset, issue):
  F1 = A OR B            A = issued phase in {PEAK, DEESC, RECOVERY}; B = ddCLIFS <= -0.15 AND zCISS >= 0.5
  F2 = [#peers (6, current country excluded) in {PEAK, DEESC, RECOVERY} >= 2]
  F3 = [sd13(r) < sd52(r) AND 4-week price return > 0], computed on weekly RETURNS of EACH asset
       separately (E1,E2,E3,B1,B2) from the newest observation available at the issue
  vote V = F1 + F2 + F3, issue iff V >= 2;  LOFO and Shapley describe the 2/3 voting game only.
  zone [L,U]: native weeks strictly AFTER the issue up to issue + 3 calendar months (capped by the target window).

Everything here is forward-only.  The realised trough (argmin of the INDEX LEVEL, never of the weekly
return) and the interval score are scoring-branch functions that are called only after maturity.
"""
from __future__ import annotations

from datetime import date
from itertools import combinations
from math import factorial
from statistics import stdev

from dateutil.relativedelta import relativedelta

EA7 = ('BE', 'DE', 'ES', 'FI', 'FR', 'IT', 'NL')
ASSETS = ('E1', 'E2', 'E3', 'B1', 'B2')
COOLING_PHASES = frozenset(('PEAK', 'DEESC', 'RECOVERY'))
DD_THRESHOLD, Z_THRESHOLD = -0.15, 0.5
MIN_WEEKLY_LEVELS = 53                     # 52 weekly returns need 53 levels


class BottomEligibilityError(ValueError):
    """Input cannot support an honest factor value -> NOT_EVALUATED, never 0."""


def compute_f1_ewi_cooling(phase, dd_clifs, z_ciss, *, phase_eligible: bool, values_eligible: bool,
                           dd_threshold: float = DD_THRESHOLD, z_threshold: float = Z_THRESHOLD) -> dict:
    phase_pass = bool(phase_eligible and phase in COOLING_PHASES)
    numeric_pass = bool(values_eligible and dd_clifs <= dd_threshold and z_ciss >= z_threshold)
    # one passing OR-branch is sufficient even if the other is unknown;
    # F1 = 0 needs BOTH branches observable.
    if not (phase_pass or numeric_pass) and not (phase_eligible and values_eligible):
        raise BottomEligibilityError('F1_INDETERMINATE_INPUT_ELIGIBILITY')
    return {'F1': int(phase_pass or numeric_pass), 'phase_pass': phase_pass, 'numeric_cooling_pass': numeric_pass,
            'thresholds': {'dd': dd_threshold, 'z': z_threshold}}


def compute_f2_peer_recovery(country: str, issued_peer_phases: dict, *, allowed_countries=EA7,
                             min_peers: int = 2) -> dict:
    allowed = set(allowed_countries)
    if country not in allowed:
        raise BottomEligibilityError('COUNTRY_NOT_IN_EA7')
    peers = sorted(allowed - {country})
    if len(peers) != 6 or not all(isinstance(issued_peer_phases.get(k), str) and issued_peer_phases.get(k)
                                  for k in peers):
        raise BottomEligibilityError('F2_INCOMPLETE_EA7_PEER_ASOF')
    count = sum(issued_peer_phases[p] in COOLING_PHASES for p in peers)
    return {'F2': int(count >= min_peers), 'n_peer_recovery': count, 'n_peer_eligible': 6,
            'qualified_peers': [p for p in peers if issued_peer_phases[p] in COOLING_PHASES]}


def compute_f3_asset_cooling(levels_asof) -> dict:
    """Ascending native weekly INDEX LEVELS, every one released by the issue.  Volatility is of returns."""
    levels = [float(x) for x in levels_asof]
    if len(levels) < MIN_WEEKLY_LEVELS or any(p <= 0 or p != p for p in levels):
        raise BottomEligibilityError('F3_NEEDS_53_POSITIVE_WEEKLY_LEVELS')
    returns = [b / a - 1.0 for a, b in zip(levels, levels[1:])]
    sigma13, sigma52 = stdev(returns[-13:]), stdev(returns[-52:])
    ret4 = levels[-1] / levels[-5] - 1.0
    return {'F3': int(sigma13 < sigma52 and ret4 > 0), 'sigma13_weekly': sigma13, 'sigma52_weekly': sigma52,
            'return4_week': ret4}


def levels_from_weekly_returns(returns, base: float = 1.0) -> list:
    """Compound weekly returns into an index level path (needed when a tape stores returns, not levels)."""
    out, level = [base], base
    for r in returns:
        if not r > -1:
            raise BottomEligibilityError('RETURN_AT_OR_BELOW_MINUS_ONE')
        level *= 1.0 + float(r)
        out.append(level)
    return out


def factor_vote(f1: int, f2: int, f3: int, min_votes: int = 2) -> int:
    f = (f1, f2, f3)
    if any(v not in (0, 1) for v in f):
        raise BottomEligibilityError('FACTORS_MUST_BE_BINARY_AND_ELIGIBLE')
    return int(sum(f) >= min_votes)


def factor_contributions(factors: tuple) -> dict:
    """Leave-one-factor-out and Shapley of the 2/3 voting game (logic only, not financial causality)."""
    f = tuple(factors)
    if len(f) != 3 or any(x not in (0, 1) for x in f):
        raise BottomEligibilityError('INVALID_FACTOR_VECTOR')

    def v(active):
        return factor_vote(*(f[i] if i in active else 0 for i in range(3)))

    full = v({0, 1, 2})
    lofo, shapley = [], []
    for i in range(3):
        lofo.append(full - v({0, 1, 2} - {i}))
        others = [j for j in range(3) if j != i]
        phi = 0.0
        for k in range(3):
            for sub in combinations(others, k):
                s = set(sub)
                phi += factorial(k) * factorial(2 - k) / factorial(3) * (v(s | {i}) - v(s))
        shapley.append(phi)
    return {'vote': full, 'lofo': lofo, 'shapley': shapley, 'shapley_sum': sum(shapley)}


def fixed_rule_vote_without(f1: int, f2: int, f3: int, drop: str, min_votes: int = 2) -> int:
    """FIXED-RULE ablation: remove one factor and keep the >= 2 rule, so the two remaining must BOTH pass.
    Re-tuning the rule to 2/2 or 1/2 would be a different (independent re-tuning) experiment."""
    if drop not in ('F1', 'F2', 'F3'):
        raise BottomEligibilityError('UNKNOWN_FACTOR')
    kept = [v for name, v in (('F1', f1), ('F2', f2), ('F3', f3)) if name != drop]
    return factor_vote(kept[0], kept[1], 0, min_votes=min_votes)


def bottom_factor_ablation(loss: dict, *, retuned: bool = False) -> dict:
    """Matched forecast-value ablation  dL_Fj = L(M without Fj) - L(M123)  on IDENTICAL keys.

    `loss` = {'M123': {key: L}, 'M23': {...}, 'M13': {...}, 'M12': {...}} (M23 drops F1, M13 drops F2,
    M12 drops F3).  dL > 0: the full system had the lower loss on the matched sample; dL = 0: no
    difference seen; dL < 0: adding the factor hurt in DEV.  This is NOT the Shapley share of the vote.
    """
    if set(loss) != {'M123', 'M23', 'M13', 'M12'}:
        raise BottomEligibilityError('FOUR_MATCHED_VARIANTS_REQUIRED')
    keys = set(loss['M123'])
    if not keys or any(set(v) != keys for v in loss.values()):
        raise BottomEligibilityError('IDENTICAL_ISSUE_KEYS_REQUIRED')
    for v in loss.values():
        if any(x is None or x != x or x in (float('inf'), float('-inf')) for x in v.values()):
            raise BottomEligibilityError('FINITE_LOSS_REQUIRED_ON_EVERY_MATCHED_KEY')
    n = len(keys)
    total = {m: float(sum(v.values())) for m, v in loss.items()}
    d = {'F1': total['M23'] - total['M123'], 'F2': total['M13'] - total['M123'],
         'F3': total['M12'] - total['M123']}
    return {'status': 'INDEPENDENT_RETUNING_EXPERIMENT' if retuned else 'FIXED_RULE_ABLATION', 'n_keys': n,
            'delta_loss_total': d, 'delta_loss_mean': {k: v / n for k, v in d.items()},
            'reading': {k: ('FULL_BETTER' if v > 0 else 'NO_DIFFERENCE' if v == 0 else 'FACTOR_HURT_IN_DEV')
                        for k, v in d.items()},
            'not_a': 'SHAPLEY_SHARE_OR_CAUSAL_CLAIM'}


def predict_bottom_zone(*, issue: date, factor_vote_2of3: int, native_weeks, target_window_end: date,
                        width_months: int = 3) -> dict:
    """[L,U] over native weeks with  issue < L <= U <= min(issue + width_months, target end)."""
    if factor_vote_2of3 not in (0, 1):
        raise BottomEligibilityError('INVALID_VOTE')
    if factor_vote_2of3 == 0:
        return {'issued_zone': False, 'lower': None, 'upper': None, 'reason': 'NO_ZONE_FACTOR_VOTE_BELOW_2'}
    cap = min(issue + relativedelta(months=width_months), target_window_end)
    future = sorted(d for d in native_weeks if issue < d <= cap)     # a week equal to the issue is not future
    if not future:
        raise BottomEligibilityError('NO_FUTURE_NATIVE_WEEK_IN_REGISTERED_ZONE')
    return {'issued_zone': True, 'lower': future[0].isoformat(), 'upper': future[-1].isoformat(),
            'reason': None}


def realized_trough(native_future_levels) -> dict:
    """Scoring branch only: the week with the LOWEST INDEX LEVEL (earliest on ties), not the worst return."""
    if not native_future_levels or any(price <= 0 for _, price in native_future_levels):
        raise BottomEligibilityError('INVALID_OR_MISSING_NATIVE_LEVELS')
    day, level = min(native_future_levels, key=lambda z: (z[1], z[0]))
    return {'trough_week': day, 'trough_level': level}


def interval_score_weeks(lo: float, hi: float, truth: float, alpha: float) -> float:
    if not (0 < alpha < 1 and lo <= hi):
        raise BottomEligibilityError('INVALID_INTERVAL_ALPHA_OR_BOUNDS')
    return (hi - lo) + (2 / alpha) * max(0, lo - truth) + (2 / alpha) * max(0, truth - hi)


def issue_bottom_signal(country: str, asset: str, issue: date, *, phase, dd_clifs, z_ciss, phase_eligible,
                        values_eligible, issued_peer_phases, asset_levels_asof, native_weeks,
                        target_window_end: date, allowed_countries=EA7) -> dict:
    """One asset, one issue: F1, F2, F3 -> vote -> zone.  Any ineligible factor -> NOT_EVALUATED row."""
    if asset not in ASSETS:
        raise BottomEligibilityError('UNKNOWN_BOTTOM_ASSET')
    try:
        f1 = compute_f1_ewi_cooling(phase, dd_clifs, z_ciss, phase_eligible=phase_eligible,
                                    values_eligible=values_eligible)
        f2 = compute_f2_peer_recovery(country, issued_peer_phases, allowed_countries=allowed_countries)
        f3 = compute_f3_asset_cooling(asset_levels_asof)
    except BottomEligibilityError as exc:
        return {'country': country, 'asset': asset, 'issue': issue.isoformat(), 'status': 'NOT_EVALUATED',
                'reason': str(exc), 'issued_zone': False}
    vote = factor_vote(f1['F1'], f2['F2'], f3['F3'])
    zone = predict_bottom_zone(issue=issue, factor_vote_2of3=vote, native_weeks=native_weeks,
                               target_window_end=target_window_end)
    return {'country': country, 'asset': asset, 'issue': issue.isoformat(), 'status': 'EVALUATED',
            'f1': f1['F1'], 'f2': f2['F2'], 'f3': f3['F3'], 'votes': f1['F1'] + f2['F2'] + f3['F3'],
            'contribution': factor_contributions((f1['F1'], f2['F2'], f3['F3'])), **zone,
            'rule_id': 'BOTTOM_3F_2_OF_3_V1'}
