"""W05 P03/P04 additive challenger: actual ES-tail support, never a gate swap.

Incumbent ``tail.var_es`` and its scaled-global support threshold are untouched.
The owner must explicitly name ACTUAL_TAIL_ESS or GLOBAL_TAIL_PROXY when asking
for a support gate. Formula validation without that decision remains diagnostic.
"""
from __future__ import annotations

import numpy as np

from .tail import var_es
from .contract import ALPHAS, METHODS, REGIMES, simplex, stamp

VERSION = "W05_REV357_ACTUAL_TAIL_V1"


def tail_mass_decomposition(losses, mass, alpha):
    """Return q in original scenario order, splitting VaR ties proportionally."""
    x = np.asarray(losses, float)
    p = np.full(len(x), 1 / len(x)) if mass is None and x.ndim == 1 and len(x) else np.asarray(mass, float)
    old = var_es(x, alpha, p)
    v = old['VaR']
    q = np.where(x > v, p, 0.0)
    ties = (x == v) & (p > 0)
    remaining = (1 - alpha) - q.sum()
    if remaining < -1e-12 or not np.any(ties):
        raise ValueError('TAIL_ATOM_DECOMPOSITION_INVALID')
    q[ties] = p[ties] * max(0.0, remaining) / p[ties].sum()
    tail = float(q.sum())
    if tail <= 0 or abs(tail - (1 - alpha)) > 1e-12:
        raise ValueError('TAIL_MASS_NOT_CONSERVED')
    es = float(q @ x / tail)
    return {
        **old, 'challenger_version': VERSION, 'tail_contribution_mass': q.tolist(),
        'n_eff_global': float(1 / (p @ p)),
        'n_tail_global_proxy': old['effective_tail_support'],
        'n_eff_tail_actual': float(tail * tail / (q @ q)),
        'n_positive_tail_atoms': int(np.count_nonzero(q)),
        'ES_from_tail_mass': es, 'ES_oracle_residual': es - old['ES'],
        'tail_mass_residual': tail - (1 - alpha),
        'tie_rule': 'PROPORTIONAL_TO_ORIGINAL_SCENARIO_MASS',
        'incumbent_support_semantics': 'GLOBAL_TAIL_PROXY',
        'support_gate_status': 'U', 'support_gate_reason': 'OWNER_SEMANTICS_REQUIRED',
    }


def var_es_actual_tail_v1(losses, alpha, mass=None, *, support_owner=None):
    """Both support definitions; no owner-less default minimum or promotion."""
    out = tail_mass_decomposition(losses, mass, alpha)
    if support_owner is None:
        return out
    if not isinstance(support_owner, dict) or support_owner.get('approved') is not True or not support_owner.get('owner_hash'):
        return {**out, 'support_gate_status': 'BLOCKED', 'support_gate_reason': 'OWNER_REQUIRED'}
    names = {'ACTUAL_TAIL_ESS': 'n_eff_tail_actual', 'GLOBAL_TAIL_PROXY': 'n_tail_global_proxy'}
    semantics = support_owner.get('semantics')
    minimum = support_owner.get('minimum')
    if semantics not in names or not isinstance(minimum, (int, float)) or not np.isfinite(minimum) or minimum < 0:
        return {**out, 'support_gate_status': 'BLOCKED', 'support_gate_reason': 'EXPLICIT_TAIL_SUPPORT_SEMANTICS_AND_MINIMUM_REQUIRED'}
    value = out[names[semantics]]
    return {**out, 'support_gate_status': 'PASS' if value >= minimum else 'U',
            'support_gate_reason': None if value >= minimum else 'TAIL_SUPPORT_BELOW_OWNER_MINIMUM',
            'support_owner_hash': support_owner['owner_hash'], 'support_semantics': semantics,
            'support_minimum': float(minimum), 'support_value': value}


def make_six_strategy_risk_cube_v1(issued_weights, scenario_vintages, *, alphas=ALPHAS, support_owner=None):
    """Audit cube keyed by strategy/issue/regime/horizon/basis, with empty cells.

    Each weight record names frame_id, issue_at, regime, strategy_id, weights and
    cost_fraction. Frames carry scenarios[J,6] or joint_paths[J,H,6], mass,
    available_at, observed_through, fitted_at, source_sha and source_status.
    A joint path is buy-and-hold after the single initial fee. Weekly action
    paths require externally supplied scenario-specific action schedules and
    are deliberately not approximated using square-root-time scaling.
    """
    rows, seen = [], set()
    for rec in issued_weights:
        key = (rec['strategy_id'], rec['issue_at'], rec['frame_id'])
        if key in seen:
            raise ValueError('DUPLICATE_STRATEGY_ISSUE_FRAME')
        seen.add(key)
        if rec['strategy_id'] not in (*METHODS, 'POLICY') or rec['regime'] not in REGIMES:
            raise ValueError('UNKNOWN_STRATEGY_OR_REGIME')
        frame = scenario_vintages.get(rec['frame_id'])
        if frame is None:
            rows.append({**rec, 'status': 'BLOCKED', 'reason': 'SCENARIO_FRAME_MISSING'})
            continue
        issue = stamp(rec['issue_at'])
        if not 2005 <= issue.year <= 2019:
            raise ValueError('OOS_SEALED')
        retrospective = frame.get('source_status') == 'RETROSPECTIVE'
        needed = ('available_at', 'observed_through', 'fitted_at', 'source_sha')
        if not all(frame.get(k) for k in needed) or (not retrospective and any(stamp(frame[k]) > issue for k in needed[:3])):
            rows.append({**rec, 'status': 'BLOCKED', 'reason': 'PIT_RISK_FRAME_UNVERIFIED'})
            continue
        if stamp(frame['observed_through']).year > 2019:
            rows.append({**rec, 'status': 'BLOCKED', 'reason': 'OOS_SEALED'})
            continue
        if frame.get('source_status') not in ('RETROSPECTIVE', 'RELEASE_VERIFIED'):
            rows.append({**rec, 'status': 'BLOCKED', 'reason': 'RISK_SOURCE_STATUS_UNVERIFIED'})
            continue
        w = simplex(rec['weights'], 6)
        cost = float(rec['cost_fraction'])
        if not np.isfinite(cost) or not 0 <= cost < 1:
            raise ValueError('COST_DOMAIN')
        if 'joint_paths' in frame:
            paths = np.asarray(frame['joint_paths'], float)
            if paths.ndim != 3 or paths.shape[2] != 6 or paths.shape[1] < 1:
                raise ValueError('JOINT_PATH_SHAPE')
            growth = np.prod(1 + paths, axis=1) @ w
            horizon = paths.shape[1]
            raw = paths
        else:
            raw = np.asarray(frame['scenarios'], float)
            if raw.ndim != 2 or raw.shape[1] != 6:
                raise ValueError('SIX_SCENARIO_ASSETS_REQUIRED')
            growth = 1 + raw @ w
            horizon = 1
        if not np.isfinite(raw).all() or np.any(raw <= -1):
            raise ValueError('SCENARIO_RETURN_DOMAIN')
        for basis, loss in (('GROSS_WEEKLY' if horizon == 1 else 'GROSS_JOINT_PATH', 1 - growth),
                            ('NET_AFTER_COST', 1 - (1 - cost) * growth)):
            for alpha in alphas:
                calc = var_es_actual_tail_v1(loss, alpha, frame.get('mass'), support_owner=support_owner)
                rows.append({**calc, 'status': calc['support_gate_status'], 'reason': calc['support_gate_reason'],
                             'risk_calculation_status': calc['status'],
                             'eligible': calc['support_gate_status'] == 'PASS',
                             'strategy_id': rec['strategy_id'], 'issue_at': rec['issue_at'],
                             'regime': rec['regime'], 'frame_id': rec['frame_id'], 'source_sha': frame['source_sha'],
                             'horizon': horizon, 'risk_basis': basis, 'n_native': frame.get('n_native'),
                             'source_status': frame['source_status'], 'path_policy': 'BUY_AND_HOLD_INITIAL_FEE_ONCE',
                             'claim': 'DEV_RETROSPECTIVE' if retrospective else 'CONDITIONAL_ISSUED_NOT_AUTHENTICATED'})
    coverage = []
    for method in (*METHODS, 'POLICY'):
        for regime in REGIMES:
            cells = [r for r in rows if r.get('strategy_id') == method and r.get('regime') == regime]
            issues = {r['issue_at'] for r in cells}
            n = sum(all(r.get('eligible') is True for r in cells if r['issue_at'] == issue) for issue in issues)
            status = 'PASS' if issues and n == len(issues) else ('U' if issues else 'NOT_EVALUATED')
            coverage.append({'strategy_id': method, 'regime': regime, 'n_eligible': n,
                             'n_issued_opportunities': len(issues), 'n_ineligible': len(issues)-n,
                             'status': status, 'reason': None if status == 'PASS' else
                             ('TAIL_SUPPORT_OR_SOURCE_GATE_UNRESOLVED' if issues else 'ZERO_SUPPORT')})
    status = ('BLOCKED' if not rows or any(r.get('status') == 'BLOCKED' for r in rows) else
              'PASS' if all(r.get('eligible') is True for r in rows) else 'U')
    return {'status': status,
            'version': VERSION, 'rows': rows, 'coverage': coverage, 'OOS': 'SEALED', 'promoted': False}
