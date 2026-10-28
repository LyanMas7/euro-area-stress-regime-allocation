"""W05 conditional order admissibility: issued permissions, owner hurdle and caps.

This is an executable *conditional* decision policy, not an assertion that owner
records are genuine or that historical signals had been published. The official
path must receive authentic upstream issuer packets and authenticated owners.

Economic rule: trade only if feasible AND (approved forced risk repair OR
(benchmark-specific structural breach AND positive cost-adjusted utility gain
above the owner hurdle)). Asset-specific bottom permission is separate from
forecasting the trough; neither a forecast nor an aggregate zone authorises
purchases in a different asset.
"""
from __future__ import annotations
import numpy as np
from .contract import MAIN_IDS, METHODS, RISKY, owner_ok, result, stamp, simplex
from . import benchmarks as B, regime_policy as P

EPS = 1e-10


def issued_permission_gate(packet: dict, decision_at, policy_owner: dict | None, *,
                           risk_caps_owner: dict | None = None) -> dict:
    """Verify bottom and permit are as-of, asset-matched and linked.

    No implicit permissions and no unissued permission records. The gate checks
    *record structure and time*, not cryptographic authenticity.
    """
    bottom = packet.get('bottoms') or {}
    permits = packet.get('permissions') or {}
    if not isinstance(bottom, dict) or not isinstance(permits, dict):
        return result('BLOCKED', 'BOTTOM_AND_PERMISSIONS_MUST_BE_ASSET_MAPS')
    if (bottom or permits) and not owner_ok(policy_owner):
        return result('BLOCKED', 'BOTTOM_PERMISSION_POLICY_OWNER_MISSING')
    if permits:
        eligible = (policy_owner or {}).get('eligible_assets')
        if not isinstance(eligible, list) or not eligible or any(a not in RISKY for a in eligible):
            return result('BLOCKED', 'BOTTOM_PORTFOLIO_ELIGIBILITY_OWNER_ASSET_MAP_REQUIRED')
        if any(asset not in eligible for asset in permits):
            return result('BLOCKED', 'BOTTOM_PERMISSION_ASSET_NOT_OWNER_ELIGIBLE')
    t = stamp(decision_at)
    for asset, rec in bottom.items():
        if asset not in RISKY or not isinstance(rec, dict):
            return result('BLOCKED', 'BOTTOM_FORECAST_ASSET_INVALID', asset=asset)
        required = ('asset', 'issued_at', 'zone_start', 'zone_end', 'source_sha', 'publisher_owner_hash')
        # A proposed three-factor bottom is not issuer-released simply because
        # somebody supplied a SHA string and an old clock. Only released and
        # published bottoms may participate in trading permission.
        if rec.get('status') != 'RELEASE_VERIFIED' or rec.get('published') is not True:
            return result('BLOCKED', 'BOTTOM_FORECAST_RELEASE_UNVERIFIED', asset=asset)
        if any(not rec.get(k) for k in required) or rec['asset'] != asset:
            return result('BLOCKED', 'BOTTOM_ISSUER_PROVENANCE_MISSING', asset=asset)
        try:
            if stamp(rec['issued_at']) > t or stamp(rec['zone_start']) > stamp(rec['zone_end']):
                return result('BLOCKED', 'BOTTOM_FUTURE_OR_INVERTED_ZONE', asset=asset)
        except (ValueError, TypeError):
            return result('BLOCKED', 'BOTTOM_CLOCK_UNVERIFIED', asset=asset)
    for asset, rec in permits.items():
        if asset not in RISKY or asset not in bottom or not isinstance(rec, dict):
            return result('BLOCKED', 'PERMISSION_WITHOUT_MATCHED_ISSUED_BOTTOM', asset=asset)
        keys = ('asset', 'issued_at', 'bottom_issued_at', 'bottom_source_sha',
                'publisher_owner_hash', 'committee_id', 'risk_check_id', 'funding_owner_hash')
        if any(not rec.get(k) for k in keys) or rec.get('status') != 'PASS':
            return result('BLOCKED', 'BOTTOM_PERMISSION_ISSUANCE_INCOMPLETE', asset=asset)
        b = bottom[asset]
        try:
            lineage_match = stamp(rec['bottom_issued_at']) == stamp(b['issued_at'])
        except (ValueError, TypeError):
            return result('BLOCKED', 'PERMISSION_BOTTOM_CLOCK_INVALID', asset=asset)
        if rec['asset'] != asset or rec['bottom_source_sha'] != b['source_sha'] or not lineage_match:
            return result('BLOCKED', 'PERMISSION_BOTTOM_LINEAGE_MISMATCH', asset=asset)
        try:
            if not stamp(b['issued_at']) <= stamp(rec['issued_at']) <= t:
                return result('BLOCKED', 'PERMISSION_NOT_ISSUED_ASOF', asset=asset)
        except (ValueError, TypeError):
            return result('BLOCKED', 'PERMISSION_CLOCK_UNVERIFIED', asset=asset)
        try:
            valid_from = P._start_of_eligibility(rec['valid_from']) if rec.get('valid_from') else None
            valid_until = stamp(rec['valid_until']) if rec.get('valid_until') else None
            if valid_from is not None and valid_until is not None and valid_from > valid_until:
                return result('BLOCKED', 'BOTTOM_PERMISSION_VALIDITY_INTERVAL_INVERTED', asset=asset)
        except (ValueError, TypeError, OverflowError):
            return result('BLOCKED', 'BOTTOM_PERMISSION_VALIDITY_CLOCK_INVALID', asset=asset)
        # Bottom 3F must be an independently checkable three-family decision,
        # not a free-text committee ID. Signed/issued structural evidence is
        # necessary, but even these hashes are NOT external authentication.
        rule = (policy_owner or {}).get('committee_rule')
        if not owner_ok(rule):
            return result('BLOCKED', 'BOTTOM_COMMITTEE_RULE_OWNER_REQUIRED', asset=asset)
        ev = rec.get('committee_evidence')
        if not isinstance(ev, dict) or ev.get('committee_id') != rec['committee_id'] or \
           ev.get('bottom_source_sha') != b['source_sha'] or \
           ev.get('rule_owner_hash') != rule['owner_hash'] or \
           ev.get('status') != 'RELEASE_VERIFIED' or ev.get('published') is not True:
            return result('BLOCKED', 'BOTTOM_THREE_FACTOR_ISSUED_EVIDENCE_MISSING', asset=asset)
        votes, inputs = ev.get('votes'), ev.get('input_sets')
        families = {'EWI', 'COUNTRY', 'RISK_COV'}
        if not isinstance(votes, dict) or set(votes) != families or \
           any(type(votes[f]) is not bool for f in families) or \
           not isinstance(inputs, dict) or set(inputs) != families or \
           any(not isinstance(inputs[f], list) or not inputs[f] or
               any(not isinstance(x, str) or not x for x in inputs[f]) for f in families):
            return result('BLOCKED', 'BOTTOM_THREE_FACTOR_VOTE_OR_SOURCE_SUPPORT_INVALID', asset=asset)
        try:
            committee_time = stamp(ev['issued_at'])
            if not stamp(b['issued_at']) <= committee_time <= stamp(rec['issued_at']):
                return result('BLOCKED', 'BOTTOM_THREE_FACTOR_COMMITTEE_CLOCK_INVALID', asset=asset)
        except (ValueError, TypeError, KeyError, OverflowError):
            return result('BLOCKED', 'BOTTOM_THREE_FACTOR_COMMITTEE_CLOCK_INVALID', asset=asset)
        computed = P.bottom_committee(votes, inputs, rule)
        if computed['status'] != 'PASS' or not computed['decision']:
            return result('BLOCKED', 'BOTTOM_THREE_FACTOR_INDEPENDENCE_OR_VOTE_GATE_FAILED',
                          asset=asset, committee_status=computed['status'],
                          n_effective=computed.get('n_effective_votes'))
        risk = rec.get('risk_check_evidence')
        # A PASS flag from an unrelated risk scenario is NOT confirmation that
        # the exact scenario/covariance entering this week's optimizer passed.
        if not isinstance(risk, dict) or not packet.get('scenario_source_sha') or \
           not packet.get('covariance_source_sha') or \
           risk.get('scenario_source_sha') != packet['scenario_source_sha'] or \
           risk.get('covariance_source_sha') != packet['covariance_source_sha']:
            return result('BLOCKED', 'BOTTOM_RISK_CHECK_ISSUED_SOURCE_LINEAGE_MISMATCH', asset=asset)
        # The issuer's within-caps boolean is meaningless unless it refers
        # to the SAME signed risk-cap schedule used by the current optimizer.
        # The bare standalone validator can inspect structure without owners;
        # the official W05 bridge always passes the active owner here.
        if risk_caps_owner is not None and (not owner_ok(risk_caps_owner) or
                risk.get('risk_caps_owner_hash') != risk_caps_owner['owner_hash']):
            return result('BLOCKED', 'BOTTOM_TAIL_RISK_CAP_OWNER_LINEAGE_MISMATCH', asset=asset)
        if risk.get('risk_check_id') != rec['risk_check_id'] or \
           risk.get('asset') != asset or risk.get('status') != 'RELEASE_VERIFIED' or \
           risk.get('published') is not True or risk.get('within_caps') is not True or \
           not risk.get('source_sha') or not risk.get('publisher_owner_hash'):
            return result('BLOCKED', 'BOTTOM_ISSUED_TAIL_RISK_EVIDENCE_MISSING', asset=asset)
        try:
            if not stamp(b['issued_at']) <= stamp(risk['issued_at']) <= stamp(rec['issued_at']):
                return result('BLOCKED', 'BOTTOM_RISK_CHECK_CLOCK_INVALID', asset=asset)
        except (ValueError, TypeError, KeyError, OverflowError):
            return result('BLOCKED', 'BOTTOM_RISK_CHECK_CLOCK_INVALID', asset=asset)
    return result('PASS', verified_assets=sorted(permits),
                  claim='STRUCTURAL_ASOF_PROOF_NOT_CRYPTOGRAPHIC_AUTHENTICATION')


def structural_trigger(pre, method, cov, tolerance_owner, *, target=None, exposure_owner=None) -> dict:
    """Benchmark-native structural criterion; zero-risk sleeve is not a shape.

    Cash-to-risk activation needs its own owner-specified capital-fraction band;
    benchmark weight geometry cannot determine an allocation hurdle at zero risk.
    """
    if method in MAIN_IDS:
        # Spec RQ03 risk tolerance 5/10.6: MAIN has a risk-tolerance VECTOR Theta_M, not a benchmark tau.
        return result('BLOCKED', 'MAIN_HAS_NO_BENCHMARK_STRUCTURAL_TOLERANCE', method=method)
    if method not in METHODS:
        return result('BLOCKED', 'UNKNOWN_BENCHMARK_GEOMETRY', method=str(method))
    p = simplex(pre, 6)
    risk_now = float(p[:5].sum())
    capital_gap = None if target is None else abs(float(simplex(target, 6)[:5].sum()) - risk_now)
    signed_band = (owner_ok(exposure_owner) and exposure_owner.get('unit') == 'RISKY_CAPITAL_FRACTION'
                   and type(exposure_owner.get('value')) in (int, float)
                   and np.isfinite(exposure_owner['value'])
                   and 0 <= exposure_owner['value'] <= 1)
    capital_triggered = (capital_gap is not None and signed_band and
                         capital_gap > float(exposure_owner['value']) + EPS)
    if risk_now > EPS:
        s = B.structural_deviation(method, p[:5]/risk_now, cov)
        ck = B.tolerance_check(method, s, tolerance_owner)
        if ck['status'] != 'PASS':
            return ck
        return result('PASS', triggered=bool(not ck['within'] or capital_triggered),
                      shape_triggered=not ck['within'], capital_triggered=bool(capital_triggered),
                      capital_gap=capital_gap, capital_tolerance_owner_valid=bool(signed_band),
                      native_geometry=s, reason_code='BENCHMARK_SHAPE_OR_OWNER_CAPITAL_BAND')
    if capital_triggered:
        return result('PASS', triggered=True, capital_triggered=True, capital_gap=capital_gap,
                      native_geometry=None, reason_code='OWNER_EXPOSURE_ACTIVATION')
    return result('PASS', triggered=False, native_geometry=None, capital_gap=capital_gap,
                  capital_tolerance_owner_valid=bool(signed_band),
                  reason_code='ZERO_RISKY_SLEEVE_WITHOUT_APPROVED_TRIGGER')


def allowable_weights(pre, post, asset_zones: dict, caps: dict, *, emergency=False) -> bool:
    """Uniform asset cap and frozen-purchase check for EW/IVOL/MVP/ERC/MaxDiv/POLICY."""
    p, w = simplex(pre, 6), simplex(post, 6)
    for i, a in enumerate(RISKY):
        zone = asset_zones.get(a, 'NO_RISKY_ZONE')
        permitted_buy = zone in ('NORMAL', 'BOTTOM_REENTRY_ZONE', 'POST_BOTTOM_STRESS', 'EXIT')
        if w[i] > p[i] + EPS and not permitted_buy:
            return False
        # A waiver can authorise a protective trade, never a holding above
        # the owner-signed asset cap. Cash is the residual funding asset.
        if w[i] > float(caps[a]) + EPS:
            return False
    return True


def risk_repair_status(pre, caps, zone, emergency_owner) -> dict:
    p = simplex(pre, 6)
    approved = bool(owner_ok(emergency_owner))
    breached = [a for i, a in enumerate(RISKY) if p[i] > float(caps[a]) + EPS]
    frozen = [a for a in breached if zone['assets'].get(a) == 'NO_RISKY_ZONE']
    if not breached:
        # `emergency_approved` lets choose_economically repair a VaR/ES breach of HOLD itself, which
        # an asset-cap check alone cannot see.
        return result('PASS', forced=False, breached=[], frozen=[], emergency_approved=approved)
    if frozen and not approved:
        return result('BLOCKED', 'FREEZE_AND_RISK_BREACH_CONFLICT_ESCALATE',
                      breached=breached, frozen=frozen)
    if not approved:
        return result('BLOCKED', 'RISK_REPAIR_OWNER_REQUIRED', breached=breached, frozen=frozen)
    return result('PASS', forced=True, breached=breached, frozen=frozen, emergency_approved=True)


def choose_economically(myopic: dict, *, structural: dict, repair: dict,
                        hurdle_owner: dict | None) -> dict:
    """Prevent profitable-but-unauthorised discretionary orders and forced trade.

    Hurdle is expressed in expected *utility* units, not gross percent return.
    A forced repair requires an approved override and a feasible NON-HOLD action.
    """
    if myopic.get('status') != 'PASS':
        return myopic
    rows = myopic['rows']
    hold = next((r for r in rows if r['action'] == 'HOLD'), None)
    feasible = [r for r in rows if r['feasible']]
    if repair.get('forced'):
        options = [r for r in feasible if r['action'] != 'HOLD']
        if not options:
            return result('BLOCKED', 'APPROVED_RISK_REPAIR_NOT_FEASIBLE')
        best = max(options, key=lambda r: r['expected_utility'])
        return result('PASS', best=best, rows=rows, decision_reason='APPROVED_FORCED_RISK_REPAIR')
    if hold is None or not hold['feasible']:
        # HOLD itself breaches a hard VaR/ES cap.  D <= tau_trigger can never mask that (spec 3.4 / 4.4):
        # only an owner-approved forced repair may trade; otherwise escalate.  The old reason code is kept.
        if repair.get('emergency_approved'):
            options = [r for r in feasible if r['action'] != 'HOLD']
            if not options:
                return result('BLOCKED', 'APPROVED_RISK_REPAIR_NOT_FEASIBLE', label='RISK_REPAIR_REQUIRED')
            best = max(options, key=lambda r: r['expected_utility'])
            return result('PASS', best=best, rows=rows, label='RISK_REPAIR_REQUIRED',
                          decision_reason='APPROVED_FORCED_RISK_REPAIR_HOLD_BREACHES_CAPS')
        return result('BLOCKED', 'HOLD_INFEASIBLE_NO_APPROVED_REPAIR', label='RISK_REPAIR_REQUIRED')
    if not structural.get('triggered'):
        return result('PASS', best=hold, rows=rows, decision_reason='HOLD_WITHIN_STRUCTURAL_GATE')
    if not owner_ok(hurdle_owner) or hurdle_owner.get('unit') != 'EXPECTED_UTILITY_DELTA' or \
       not isinstance(hurdle_owner.get('value'), (int, float)) or \
       not np.isfinite(hurdle_owner['value']) or hurdle_owner['value'] < 0:
        return result('PASS', best=hold, rows=rows, decision_reason='HOLD_UNAPPROVED_DISCRETIONARY_HURDLE')
    best = max(feasible, key=lambda r: r['expected_utility'])
    gain = best['expected_utility'] - hold['expected_utility']
    if best['action'] == 'HOLD' or gain <= float(hurdle_owner['value']) + 1e-14:
        return result('PASS', best=hold, rows=rows, decision_reason='HOLD_NET_VALUE_BELOW_HURDLE', gain=gain)
    return result('PASS', best=best, rows=rows, decision_reason='APPROVED_NET_UTILITY_UPLIFT',
                  gain=gain, hurdle=float(hurdle_owner['value']))
