"""RQ03-15..22, C02, D01, D02: regime shares, freeze/bottom zones and ONE capital controller.

Weights w_{b,s,u} = (a_{s,u} v_{b,u}, 1 - a_{s,u}); a needs an owner record whose UNIT is one of
RISKY_CAPITAL_FRACTION / RISK_BUDGET_FRACTION / ES_TOLERANCE (C05). Only the capital unit
maps to a directly; the other units need a risk solve and stay BLOCKED here.
Zones (issue-observable, nothing dated after the issue is read):
  NORMAL -> NO_RISKY_ZONE from the latest issued Short confirmation -> per-asset
  BOTTOM_REENTRY_ZONE after an issued bottom AND a granted permission -> POST_BOTTOM_STRESS
  -> EXIT only on an issued first-N (EXIT with a non-N issued regime escalates).
Single controller (D01):  cap_{a,u} = clip(base_{regime} - D_u + A_{a,u}, 0, owner_cap_a),
  with A_{a,u} = 0 unless the asset-specific permission is granted (C02: a predicted bottom
  is never a permission). Post-bottom: A decays A0 exp(-age/tau_A) and defense eases
  D0 exp(-age_exit/tau_D): two separate levers, never one ambiguous "more risk" knob.
"""
from __future__ import annotations
import copy
import math
from datetime import date, datetime, timezone

import numpy as np

from .contract import REGIMES, RISKY, STRESS, Blocked, owner_ok, result, simplex, stamp

SHARE_UNITS = ('RISKY_CAPITAL_FRACTION', 'RISK_BUDGET_FRACTION', 'ES_TOLERANCE')
ZONES = ('NORMAL', 'STRESS_PRE_CONFIRMATION', 'NO_RISKY_ZONE', 'BOTTOM_REENTRY_ZONE', 'POST_BOTTOM_STRESS', 'EXIT', 'EXIT_REGIME_CONFLICT')


def _start_of_eligibility(value):
    """Date-only zone starts mean 00:00 UTC; issuance dates mean day-end.

    Unlike publication, a calendar validity interval includes the *entire*
    first day. The permission itself must still be published before any order.
    """
    if isinstance(value, date) and not isinstance(value, datetime):
        return datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    if isinstance(value, str) and len(value) == 10:
        return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)
    return stamp(value)


def risky_share(regime: str, owner: dict | None) -> dict:
    """RQ03-15 (N) / RQ03-16 (D, G, BOTH): a_s from an approved owner record only."""
    if regime not in REGIMES:
        raise ValueError('UNKNOWN_REGIME')
    if not owner_ok(owner):
        return result('BLOCKED', 'C05_RISKY_SHARE_OWNER_AND_UNITS_REQUIRED', regime=regime)
    unit = owner.get('unit')
    if unit not in SHARE_UNITS:
        return result('BLOCKED', 'C05_CAP_UNIT_UNDEFINED', regime=regime, supplied_unit=unit)
    rec = (owner.get('by_regime') or {}).get(regime)
    if rec is None:
        return result('BLOCKED', 'REGIME_SPECIFIC_SHARE_MISSING', regime=regime)
    if unit != 'RISKY_CAPITAL_FRACTION':
        return result('BLOCKED', 'NON_CAPITAL_UNIT_REQUIRES_RISK_SOLVE', regime=regime, unit=unit)
    lo, hi, a = rec.get('lower'), rec.get('upper'), rec.get('value')
    if not all(isinstance(v, (int, float)) for v in (lo, hi, a)) or not 0 <= lo <= a <= hi <= 1:
        return result('FAIL', 'RISKY_SHARE_OUTSIDE_OWNER_BAND', regime=regime, value=a, band=(lo, hi))
    return result('PASS', regime=regime, a=float(a), band=(lo, hi), unit=unit, owner_hash=owner['owner_hash'],
                  time_since_onset_policy=rec.get('time_since_onset_policy', 'NONE_REGISTERED'))


def allocate(shape5, a: float) -> np.ndarray:
    v = simplex(shape5, 5)
    if not 0 <= a <= 1:
        raise ValueError('RISKY_SHARE_DOMAIN')
    return np.r_[a * v, 1 - a]


def project_to_asset_caps(target, caps: dict) -> dict:
    """Project the proposed risky sleeve DOWN into per-asset caps.

    The owner has not licensed cross-asset redistribution: excess flows to
    cash, *never* automatically to another risky asset. This deterministic
    operation is not permission to BUY; issued zone checks follow separately.
    """
    try:
        raw = simplex(target, 6)
        ceiling = np.array([caps[a] for a in RISKY], dtype=float)
        if not np.isfinite(ceiling).all() or (ceiling < 0).any() or (ceiling > 1).any():
            return result('FAIL', 'ASSET_CAPS_OUTSIDE_FRACTION_DOMAIN')
        after = raw.copy()
        after[:5] = np.minimum(raw[:5], ceiling)
        after[5] = 1.0 - float(after[:5].sum())
        if after[5] < -1e-10:
            return result('FAIL', 'CAP_PROJECTION_NEGATIVE_CASH')
        return result('PASS', weights=simplex(after, 6).tolist(),
                      cash_reallocation=float(after[5] - raw[5]),
                      projected_assets=[a for i, a in enumerate(RISKY)
                                        if raw[i] > ceiling[i] + 1e-12])
    except (KeyError, ValueError, TypeError) as exc:
        return result('FAIL', 'ASSET_CAP_PROJECTION_INPUT_INVALID:' + str(exc))


def _issued(record, t, fields):
    if record is None:
        return None
    if 'issued_at' not in record:
        raise Blocked('ISSUED_AT_REQUIRED')
    if stamp(record['issued_at']) > t:
        return None  # not yet issued: invisible, never a retroactive trade
    missing = [f for f in fields if f not in record]
    if missing:
        raise Blocked('ISSUED_RECORD_FIELDS_MISSING:' + ','.join(missing))
    return record


def asset_bottom_zone(a: str, t, bottoms: dict, permissions: dict, latest=None) -> str:
    """Per-asset Bottom zone at issue time `t` (the former zone_state loop body, behaviour unchanged).

    `latest` is the newest confirmed Short issue time (a datetime) or None.  A Bottom older than that
    Short, an ungranted permission, or a missing Bottom is NO_RISKY_ZONE; inside the issued
    [zone_start, zone_end] window it is BOTTOM_REENTRY_ZONE, after it POST_BOTTOM_STRESS.
    Nothing dated after `t` is read.  The caller decides whether the zone may be used at all (R23: only
    the Main model, and only after a same-episode BOTTOM_CONFIRMED).
    """
    t = stamp(t)
    b = _issued((bottoms or {}).get(a), t, ('asset', 'zone_start', 'zone_end'))
    perm = (permissions or {}).get(a)
    granted = isinstance(perm, dict) and perm.get('status') == 'PASS' and perm.get('asset') == a
    # Direct/legacy controller fixtures may not carry an issuance clock;
    # the strict official bridge rejects them. A clock that *is* supplied
    # must never activate a permission prior to publication.
    if granted and 'issued_at' in perm:
        try:
            granted = stamp(perm['issued_at']) <= t
        except (ValueError, TypeError):
            granted = False
    if granted and perm.get('valid_from'):
        granted = _start_of_eligibility(perm['valid_from']) <= t
    if granted and perm.get('valid_until'):
        granted = t <= stamp(perm['valid_until'])
    if b is None or (latest is not None and stamp(b['issued_at']) < latest):
        return 'NO_RISKY_ZONE'
    if not granted:
        return 'NO_RISKY_ZONE'
    if _start_of_eligibility(b['zone_start']) <= t <= stamp(b['zone_end']):
        return 'BOTTOM_REENTRY_ZONE'
    if t > stamp(b['zone_end']):
        return 'POST_BOTTOM_STRESS'
    return 'NO_RISKY_ZONE'


def zone_state(t, *, regime_record, short_confirmations, bottoms: dict, permissions: dict, exit_record=None) -> dict:
    """RQ03-17 / RQ03-18: issue-observable zone per asset; inputs are issued records."""
    t = stamp(t)
    reg = _issued(regime_record, t, ('regime',))
    if reg is None or reg['regime'] not in REGIMES:
        return result('U', 'ISSUED_REGIME_UNAVAILABLE', issue_times=[t])
    ex = _issued(exit_record, t, ('first_n',))
    shorts = [s for s in (short_confirmations or []) if _issued(s, t, ('confirmed',)) and s['confirmed'] is True]
    latest = max((stamp(s['issued_at']) for s in shorts), default=None)
    if ex is not None and ex['first_n'] is True:
        # EXIT must agree with the *current issued* Normal and be strictly
        # newer than the Short confirmation. A simultaneous/newer Short is a
        # contradictory issue: never auto-promote it to NORMAL or overwrite it.
        if reg['regime'] != 'N' or (latest is not None and stamp(ex['issued_at']) <= latest):
            reason = ('EXIT_WITH_STRESS_REGIME_ESCALATE' if reg['regime'] != 'N'
                      else 'EXIT_WITH_SIMULTANEOUS_OR_NEWER_SHORT_ESCALATE')
            return result('U', reason, issue_times=[t], zone='EXIT_REGIME_CONFLICT',
                          assets={a:'EXIT_REGIME_CONFLICT' for a in RISKY})
        # Coherent EXIT is the transition event, not an eternal state.
        # A later independently issued N makes NORMAL legal without erasing
        # the old Short and without promoting an unissued forecast.
        if stamp(reg['issued_at']) > stamp(ex['issued_at']):
            return result('PASS', issue_times=[t], zone='NORMAL',
                          assets={a:'NORMAL' for a in RISKY},
                          exit_released_at=stamp(ex['issued_at']).isoformat())
        return result('PASS', issue_times=[t], zone='EXIT',
                      assets={a:'EXIT' for a in RISKY})
    # A newer issued N cannot by itself cancel an outstanding Short forecast.
    # Only an independently issued, coherent EXIT releases the frozen state.
    if latest is not None and reg['regime'] == 'N':
        return result('U', 'NORMAL_REGIME_WITH_UNRESOLVED_SHORT_REQUIRES_ISSUED_EXIT',
                      issue_times=[t], zone='EXIT_REGIME_CONFLICT',
                      assets={a: 'EXIT_REGIME_CONFLICT' for a in RISKY})
    if latest is None and reg['regime'] == 'N':
        return result('PASS', issue_times=[t], zone='NORMAL', assets={a: 'NORMAL' for a in RISKY})
    if latest is None and reg['regime'] in STRESS:
        return result('PASS', issue_times=[t], zone='STRESS_PRE_CONFIRMATION',
                      assets={a: 'NO_RISKY_ZONE' for a in RISKY},
                      new_risky_purchases='BLOCKED_PENDING_ISSUED_SHORT_AND_ASSET_PERMISSION',
                      note='Conservative pre-confirmation: no new risky buys without short evidence')
    per = {a: asset_bottom_zone(a, t, bottoms, permissions, latest) for a in RISKY}
    zone = 'BOTTOM_REENTRY_ZONE' if 'BOTTOM_REENTRY_ZONE' in per.values() else \
        'POST_BOTTOM_STRESS' if 'POST_BOTTOM_STRESS' in per.values() else 'NO_RISKY_ZONE'
    return result('PASS', issue_times=[t], zone=zone, assets=per,
                  latest_short_issued_at=None if latest is None else latest.isoformat())


def bottom_permission(asset: str, t, *, bottom_forecast: dict | None, risk_check: dict | None,
                      funding_owner: dict | None, eligibility_owner: dict | None, policy_owner: dict | None) -> dict:
    """C02 / RQ03-18: permission is a separate object; the forecast alone never authorises."""
    t = stamp(t)
    pred = _issued(bottom_forecast, t, ('asset', 'zone_start', 'zone_end'))
    gates = {'BOTTOM_FORECAST_ISSUED': pred is not None and pred.get('asset') == asset,
             'RISK_CAPS_SATISFIED': isinstance(risk_check, dict) and risk_check.get('status') == 'PASS'
             and risk_check.get('within_caps') is True,
             'FUNDING_OWNER': owner_ok(funding_owner), 'PORTFOLIO_ELIGIBILITY': owner_ok(eligibility_owner)
             and asset in (eligibility_owner or {}).get('assets', ()),
             'REENTRY_POLICY_OWNER': owner_ok(policy_owner)}
    missing = [g for g, ok in gates.items() if not ok]
    prediction = {'asset': asset, 'issued': pred is not None, 'zone': None if pred is None else
                  (str(pred['zone_start']), str(pred['zone_end']))}
    if missing:
        owner_gap = [g for g in missing if g in ('FUNDING_OWNER', 'PORTFOLIO_ELIGIBILITY', 'REENTRY_POLICY_OWNER')]
        st = 'BLOCKED' if owner_gap else 'FAIL'
        return result(st, 'PERMISSION_GATES_NOT_MET:' + ','.join(missing), issue_times=[t], asset=asset,
                      prediction=prediction, gates=gates, permission=False)
    return result('PASS', issue_times=[t], asset=asset, prediction=prediction, gates=gates, permission=True)


def bottom_committee(votes: dict, input_sets: dict, rule_owner: dict | None, *, vote_history=None) -> dict:
    """RQ03-19: EWI / country / risk-covariance committee with dependence accounting."""
    families = ('EWI', 'COUNTRY', 'RISK_COV')
    if set(votes) != set(families) or set(input_sets) != set(families):
        raise ValueError('THREE_FAMILY_VOTES_AND_INPUT_SETS_REQUIRED')
    shared = {f: sorted(set(input_sets[f]) & set().union(*(set(input_sets[g]) for g in families if g != f)))
              for f in families}
    # Connected components of the input-sharing graph. A bridge family must
    # merge *both* prior groups, not only the first one it touches.
    todo = set(families)
    groups = []
    while todo:
        group = {next(f for f in families if f in todo)}
        todo -= group
        while True:
            linked = {f for f in todo if any(set(input_sets[f]) & set(input_sets[g]) for g in group)}
            if not linked:
                break
            group.update(linked)
            todo -= linked
        groups.append([f for f in families if f in group])
    n_eff = len(groups)
    corr = None
    if vote_history is not None:
        h = np.asarray(vote_history, dtype=float)
        c = np.nan_to_num(np.corrcoef(h, rowvar=False), nan=0.0)
        np.fill_diagonal(c, 1.0)
        lam = np.clip(np.linalg.eigvalsh(c), 0, None)
        corr = {'participation_ratio': float(lam.sum() ** 2 / np.square(lam).sum()), 'corr': c.tolist()}
    yes = [f for f in families if votes[f] is True]
    # Correlated source families count as one independent confirmation only
    # when every member of that dependence group agrees. Otherwise treating
    # one YES as an entire group YES manufactures evidence for bottom entry.
    indep_yes = sum(1 for g in groups if all(votes[f] is True for f in g))
    base = dict(votes=votes, shared_inputs=shared, dependence_groups=groups, n_effective_votes=n_eff,
                independent_yes=indep_yes, naive_yes=len(yes), history_dependence=corr)
    if any(v is None for v in votes.values()):
        return result('U', 'COMMITTEE_VOTE_UNKNOWN', **base)
    if not owner_ok(rule_owner) or type(rule_owner.get('k_independent')) is not int:
        return result('BLOCKED', 'COMMITTEE_RULE_OWNER_REQUIRED', **base)
    if not 1 <= rule_owner['k_independent'] <= n_eff:
        return result('BLOCKED', 'COMMITTEE_K_OUTSIDE_IDENTIFIED_INDEPENDENT_VOTE_SUPPORT', **base)
    return result('PASS', decision=indep_yes >= rule_owner['k_independent'], **base)


def bottom_conditional_update(forecast: dict, bottom: dict, owner: dict | None) -> dict:
    """RQ03-20: new record with new ID; the issued forecast tape is never rewritten."""
    original = copy.deepcopy(forecast)
    if not owner_ok(owner) or owner.get('mode') not in ('PREDICTION_UPDATE', 'ACTION_THRESHOLD_ONLY'):
        return result('BLOCKED', 'BOTTOM_UPDATE_OWNER_AND_MODE_REQUIRED', original_id=forecast.get('forecast_id'))
    if stamp(bottom['issued_at']) < stamp(forecast['issued_at']):
        return result('FAIL', 'BOTTOM_ISSUED_BEFORE_FORECAST_NOT_AN_UPDATE')
    new = {'forecast_id': f"{forecast['forecast_id']}|BOTTOM_UPD|{owner['owner_hash'][:12]}",
           'parent_id': forecast['forecast_id'], 'issued_at': str(bottom['issued_at']), 'mode': owner['mode']}
    if owner['mode'] == 'PREDICTION_UPDATE':
        cal = owner.get('calibrator')
        if not callable(cal):
            return result('BLOCKED', 'CALIBRATOR_OWNER_FUNCTION_REQUIRED')
        p_s, p_n = cal(forecast['P_s'], forecast['P_n'], bottom)
        if not (0 <= p_s <= 1 and 0 <= p_n <= 1):
            return result('FAIL', 'CALIBRATED_PROBABILITY_DOMAIN')
        new.update(P_s=float(p_s), P_n=float(p_n))
    else:
        new.update(P_s=forecast['P_s'], P_n=forecast['P_n'], action_threshold=owner.get('action_threshold'))
    assert forecast == original, 'ISSUED_FORECAST_MUTATED'
    return result('PASS', new_record=new, original_unchanged=True)


def decay(level0: float, age_weeks: float, tau_weeks: float) -> float:
    if level0 < 0 or age_weeks < 0 or tau_weeks <= 0:
        raise ValueError('DECAY_DOMAIN')
    return float(level0 * math.exp(-age_weeks / tau_weeks))


def capital_controller(*, regime: str, zone: dict, base_cap: dict | None, defense: dict | None,
                       allowance: dict | None, owner_asset_caps: dict | None, controllers: tuple = ('SINGLE',)) -> dict:
    """D01 / RQ03-21 / RQ03-22: one cap per risky asset from two levers."""
    if len(controllers) != 1:
        return result('FAIL', 'COMPETING_POLICY_OWNERS', controllers=list(controllers))
    if not (owner_ok(base_cap) and owner_ok(defense) and owner_ok(owner_asset_caps)):
        return result('BLOCKED', 'CAPITAL_CONTROLLER_OWNERS_MISSING',
                      missing=[k for k, v in (('base_cap', base_cap), ('defense', defense),
                                             ('owner_asset_caps', owner_asset_caps)) if not owner_ok(v)])
    if zone.get('status') != 'PASS':
        return result('U', 'ZONE_UNAVAILABLE')
    try:
        base = float(base_cap['by_regime'][regime])
        d = float(defense['D_u'])
        upper = {a: float(owner_asset_caps['by_asset'][a]) for a in RISKY}
    except (ValueError, TypeError, KeyError) as exc:
        return result('BLOCKED', 'CAP_OWNER_VALUES_MISSING_OR_INVALID:' + str(exc))
    if not np.isfinite([base, d, *upper.values()]).all() or \
       base < 0 or base > 1 or d < 0 or d > 1 or \
       any(u < 0 or u > 1 for u in upper.values()):
        return result('BLOCKED', 'BASE_DEFENSE_ASSET_CAPS_OUTSIDE_FRACTION_DOMAIN')
    caps, used_a = {}, {}
    for a in RISKY:
        z = zone['assets'][a]
        a_val = 0.0
        if z in ('BOTTOM_REENTRY_ZONE', 'POST_BOTTOM_STRESS'):
            if not owner_ok(allowance):
                return result('BLOCKED', 'BOTTOM_ALLOWANCE_SIGNED_OWNER_REQUIRED', asset=a)
            if not isinstance(allowance.get('by_asset'), dict):
                return result('BLOCKED', 'BOTTOM_ALLOWANCE_OWNER_ASSET_MAP_MISSING')
            if not isinstance(allowance['by_asset'].get(a), dict) or \
               'A0' not in allowance['by_asset'][a]:
                return result('BLOCKED', 'BOTTOM_ALLOWANCE_ASSET_VALUE_REQUIRED', asset=a)
            rec = allowance['by_asset'][a]
            try:
                a_val = decay(float(rec.get('A0', 0.0)),
                              float(rec.get('age_weeks', 0.0)),
                              float(rec.get('tau_weeks', 1.0))) \
                    if z == 'POST_BOTTOM_STRESS' else float(rec.get('A0', 0.0))
            except (ValueError, TypeError):
                return result('BLOCKED', 'BOTTOM_ALLOWANCE_DECAY_DOMAIN_INVALID', asset=a)
        if zone['zone'] in ('NO_RISKY_ZONE', 'STRESS_PRE_CONFIRMATION') or z == 'NO_RISKY_ZONE':
            caps[a] = 0.0  # freeze: no risky purchases; existing holdings are handled by actions
            used_a[a] = 0.0
            continue
        used_a[a] = a_val
        if not np.isfinite(a_val) or a_val < 0 or a_val > 1:
            return result('BLOCKED', 'BOTTOM_ALLOWANCE_OUTSIDE_FRACTION_DOMAIN', asset=a)
        caps[a] = float(np.clip(base - d + a_val, 0.0, upper[a]))
    return result('PASS', caps=caps, base=base, defense=d, allowance=used_a, zone=zone['zone'],
                  levers={'D_u': 'DEFENSIVE_PRESSURE_REDUCES_CAP', 'A_a_u': 'BOTTOM_ALLOWANCE_RAISES_ASSET_CAP'},
                  formula='clip(base - D_u + A_a_u, 0, owner_cap_a)')


def defense_after_exit(d0: float, weeks_since_exit: float, tau_weeks: float) -> dict:
    """RQ03-22: defense easing (risk TOLERANCE) is reported separately from exposure direction."""
    d = decay(d0, weeks_since_exit, tau_weeks)
    return result('PASS', D_u=d, risk_tolerance_easing=d0 - d, exposure_direction='NOT_IMPLIED_BY_EASING')


def resolve_freeze_breach(*, frozen: bool, risk_breach: bool, emergency_override: dict | None) -> dict:
    """D02: freeze, emergency override and risk breach are distinct; no hidden rebalance."""
    if frozen and risk_breach:
        if owner_ok(emergency_override):
            return result('PASS', action='EMERGENCY_RISK_REPAIR', owner_hash=emergency_override['owner_hash'])
        return result('BLOCKED', 'FREEZE_AND_RISK_BREACH_CONFLICT_ESCALATE', action='ESCALATE_INFEASIBLE')
    if frozen:
        return result('PASS', action='HOLD_FROZEN')
    if risk_breach:
        return result('PASS', action='RISK_REPAIR')
    return result('PASS', action='NO_CONSTRAINT_ACTION')


def stress_shares_complete(owner: dict | None) -> dict:
    out = {s: risky_share(s, owner) for s in STRESS}
    bad = [s for s, r in out.items() if r['status'] != 'PASS']
    return result('PASS' if not bad else 'BLOCKED', None if not bad else 'STRESS_SHARES_INCOMPLETE:' + ','.join(bad),
                  shares={s: r.get('a') for s, r in out.items()})
