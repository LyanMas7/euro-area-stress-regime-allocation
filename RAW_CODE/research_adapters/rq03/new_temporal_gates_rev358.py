"""Additive risk-vintage chronology and unit gates; no self-certification."""
from __future__ import annotations

import numpy as np

from .contract import ASSETS, ALPHAS, METHODS, REGIMES, stamp
from .new_workflow_rev357 import (build_risk_frame_asof_v1,
    authenticate_permission_against_external_registry)

VERSION = 'W05_REV358_TEMPORAL_GATES_V1'


def build_risk_frame_asof_v2(record, *, decision_at, registry=None, verify_signature=None):
    """Chronology: fitted/observed <= available <= issued <= decision.

    A predecision fit is still invalid if the immutable scenario record was
    issued before that fit existed. Horizon scenario provenance belongs to the
    same authenticated record and cannot silently outlive the decision clock.
    """
    base = build_risk_frame_asof_v1(record, decision_at=decision_at, registry=registry,
                                   verify_signature=verify_signature)
    if base['status'] != 'PASS':
        return base
    names = ('observed_through', 'fitted_at', 'available_at', 'issued_at')
    available,issued,decision=[stamp(record['available_at']),stamp(record['issued_at']),stamp(decision_at)]
    if any(stamp(record[n]) > available for n in names[:2]) or available>issued or issued>decision:
        return {'status': 'BLOCKED', 'reason': 'RISK_VINTAGE_CHRONOLOGY',
                'required_order': 'fitted_at AND observed_through <= available_at <= issued_at <= decision_at'}
    for frame in record.get('additional_horizons', []):
        if frame.get('path_policy') != 'BUY_AND_HOLD_INITIAL_FEE_ONCE':
            return {'status': 'BLOCKED', 'reason': 'JOINT_PATH_ACTION_POLICY_NOT_IDENTIFIED'}
        if any(stamp(frame.get(name, record[name])) > stamp(record['issued_at']) for name in names[:3]):
            return {'status': 'BLOCKED', 'reason': 'MULTIHORIZON_SOURCE_AFTER_ISSUANCE'}
        paths = np.asarray(frame['joint_paths'], float)
        if paths.ndim != 3 or paths.shape[2] != 6 or paths.shape[1] <= 1 or not np.isfinite(paths).all() or np.any(paths <= -1):
            return {'status': 'BLOCKED', 'reason': 'JOINT_PATH_DOMAIN'}
    return {**base, 'version': VERSION, 'clock_order_verified': list(names), 'fit_is_issuable': True,
            'external_registry_authentication_only': True}


def validate_issued_contract_v1(config, packets, *, registry=None, verify_signature=None):
    """Fail closed on units, positivity and all required risk horizons."""
    if not packets:
        return {'status': 'BLOCKED', 'reason': 'ISSUED_PACKETS_REQUIRED'}
    auth = authenticate_permission_against_external_registry(config, registry, verify_signature,
                                                              decision_at=packets[0]['issue_at'])
    if auth['status'] != 'PASS':
        return auth
    expected = {'capital_unit': 'FRACTION_OF_NAV', 'risk_unit': 'DECIMAL_LOSS_FRACTION_OF_NAV',
                'cost_unit': 'PROPORTIONAL_TRADE_VALUE', 'return_unit': 'DECIMAL_SIMPLE_RETURN',
                'account_currency': 'EUR'}
    if any(config.get(k) != v for k, v in expected.items()):
        return {'status': 'BLOCKED', 'reason': 'EXPLICIT_ECONOMIC_UNITS_REQUIRED', 'required_units': expected}
    if set(config.get('strategies',{}))!=set((*METHODS,'POLICY')):
        return {'status':'BLOCKED','reason':'SIX_STRATEGY_OWNER_RECORDS_REQUIRED'}
    from .risk_tolerance import geometry_threshold_errors
    warnings = []
    for strategy, owner in config.get('strategies', {}).items():
        if 'geometry_threshold' in owner:
            # each tau must carry its own geometry/unit; Main borrowing a benchmark tau is surfaced as a warning
            errs, warns = geometry_threshold_errors(strategy, owner['geometry_threshold'],
                                                    reference_benchmark=owner.get('reference_benchmark'))
            if errs:
                return {'status': 'BLOCKED', 'reason': 'TOLERANCE_RECORD_INVALID:' + errs[0],
                        'strategy_id': strategy, 'errors': errs}
            warnings += warns
        for key in ('utility_hurdle', 'capital_threshold'):
            value = owner.get(key)
            if not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
                return {'status': 'BLOCKED', 'reason': 'INVALID_OWNER_THRESHOLD:' + key, 'strategy_id': strategy}
        if set(owner.get('risk_by_regime',{}))!=set(REGIMES):
            return {'status':'BLOCKED','reason':'FOUR_REGIME_RISK_OWNER_RECORDS_REQUIRED','strategy_id':strategy}
        for regime, risk in owner.get('risk_by_regime', {}).items():
            if risk.get('risk_basis') != 'NET_AFTER_COST' or risk.get('unit') != 'DECIMAL_LOSS_FRACTION_OF_NAV':
                return {'status': 'BLOCKED', 'reason': 'RISK_OWNER_UNIT_OR_BASIS', 'strategy_id': strategy, 'regime': regime}
            if set(risk.get('limits', {})) != set(map(str, ALPHAS)):
                return {'status': 'BLOCKED', 'reason': 'ALL_ALPHA_RISK_OWNER_REQUIRED'}
            for limits in risk['limits'].values():
                if set(limits)!={'VaR','ES'} or any(not isinstance(v,(int,float)) or not np.isfinite(v) or v<0 for v in limits.values()):
                    return {'status':'BLOCKED','reason':'FINITE_NONNEGATIVE_VAR_ES_LIMITS_REQUIRED'}
    for i, packet in enumerate(packets):
        risk = build_risk_frame_asof_v2(packet.get('risk_frame', {}), decision_at=packet['issue_at'],
                                       registry=registry, verify_signature=verify_signature)
        if risk['status'] != 'PASS':
            return {**risk, 'first_blocker_index': i}
    return {'status': 'PASS', 'version': VERSION, 'claim': 'STRUCTURAL_CONTRACT_AND_EXTERNAL_SIGNATURE_GATE_ONLY',
            'warnings': warnings}
