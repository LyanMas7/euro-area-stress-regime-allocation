"""RQ03-27..30, D07: utility, constrained myopic choice, matched net wealth, Bellman interface.

Registered utilities (only with an approved owner record): CARA U = -exp(-gamma W),
LOG U = ln W, CRRA U = W^(1-gamma)/(1-gamma). No gamma is ever defaulted.
Return objects are distinct: REALISED (ex-post tape after issue), ANTICIPATED (scenario
mean at issue), EXPECTED (model/regime-mixture mean); net = (1-c)(1+E[w'R]) - 1.
Choice: max_a E[U(W_a)] s.t. VaR/ES of the NET loss within ALL owner caps for every alpha
-> MYOPIC_SCENARIO_OPTIMIZATION. Bellman (one step, supplied kernel only):
  V_u(s, W) = max_{a in Feasible} sum_k p_k [ U_k + beta V_{u+1}(s'_k, W'_k) ].
Without an identified kernel, reward owner and action support the continuation is
BLOCKED and RL stays DORMANT; no value estimate is fabricated.
"""
from __future__ import annotations
import math

import numpy as np

from .contract import ALPHAS, RL_STATUS, Blocked, owner_ok, result, simplex
from .tail import var_es
from .trading import cost_rates, self_financing_cost

UTILITIES = ('CARA', 'LOG', 'CRRA')
MATCH_PREREQUISITES = ('issued_upstream', 'cash_owner', 'cost_schedule', 'risk_caps', 'action_ledger')


def utility_fn(owner: dict | None):
    if not owner_ok(owner) or owner.get('utility') not in UTILITIES:
        return None
    u, g = owner['utility'], owner.get('gamma')
    if u == 'LOG':
        return lambda w: np.log(w)
    if not isinstance(g, (int, float)) or not g > 0:
        return None
    if u == 'CARA':
        return lambda w: -np.exp(-g * w)
    if g == 1:
        return lambda w: np.log(w)
    return lambda w: w ** (1 - g) / (1 - g)


def expected_utility(wealth, owner: dict | None, mass=None) -> dict:
    f = utility_fn(owner)
    if f is None:
        return result('BLOCKED', 'APPROVED_UTILITY_AND_RISK_AVERSION_OWNER_REQUIRED')
    w = np.asarray(wealth, dtype=float)
    if (w <= 0).any() and owner['utility'] in ('LOG', 'CRRA'):
        return result('FAIL', 'NONPOSITIVE_WEALTH_FOR_LOG_OR_CRRA')
    if w.ndim != 1 or not len(w) or not np.isfinite(w).all():
        return result('FAIL', 'EXPECTED_UTILITY_FINITE_WEALTH_VECTOR_REQUIRED')
    p = np.full(len(w), 1 / len(w)) if mass is None else np.asarray(mass, float)
    if p.shape != w.shape or not np.isfinite(p).all() or (p < 0).any() or abs(p.sum() - 1) > 1e-10:
        return result('FAIL', 'EXPECTED_UTILITY_PROBABILITY_MASS_INVALID')
    return result('PASS', expected_utility=float(p @ f(w)), utility=owner['utility'], gamma=owner.get('gamma'),
                  owner_hash=owner['owner_hash'])


def return_objects(weights, *, realised=None, scenarios=None, model_mean=None, cost_fraction: float = 0.0) -> dict:
    w = simplex(weights, 6)
    out = {}
    if realised is not None:
        out['REALISED'] = (1 - cost_fraction) * (1 + float(w @ np.asarray(realised, float))) - 1
    if scenarios is not None:
        out['ANTICIPATED'] = (1 - cost_fraction) * (1 + float(np.asarray(scenarios, float).mean(axis=0) @ w)) - 1
    if model_mean is not None:
        out['EXPECTED'] = (1 - cost_fraction) * (1 + float(w @ np.asarray(model_mean, float))) - 1
    return result('PASS', net_returns=out, distinct_objects=sorted(out))


def myopic_choice(pre, candidates: list[dict], scenarios, *, caps_owner: dict | None, utility_owner: dict | None,
                  schedule: dict | None, mass=None, current_wealth: float = 1.0) -> dict:
    # CARA and owner utility hurdles are wealth-scale dependent. In particular,
    # E[U(W*gross_factor)] must never be replaced by E[U(gross_factor)].
    if not isinstance(current_wealth, (int, float)) or not np.isfinite(current_wealth) or current_wealth <= 0:
        return result('FAIL', 'CURRENT_WEALTH_MUST_BE_FINITE_POSITIVE')
    missing = [k for k, v in (('risk_caps', caps_owner), ('utility', utility_owner), ('cost_schedule', schedule))
               if not owner_ok(v)]
    if missing:
        return result('BLOCKED', 'MYOPIC_CHOICE_OWNERS_MISSING:' + ','.join(missing), mode='MYOPIC_SCENARIO_OPTIMIZATION')
    if utility_fn(utility_owner) is None:
        return result('BLOCKED', 'APPROVED_UTILITY_AND_RISK_AVERSION_OWNER_REQUIRED')
    caps = caps_owner.get('caps')
    if not isinstance(caps, dict) or set(caps) != set(ALPHAS) or any(
            not isinstance(caps[a], dict) or set(caps[a]) != {'VaR', 'ES'} for a in ALPHAS):
        return result('BLOCKED', 'COMPLETE_VAR_ES_CAPS_FOR_THREE_ALPHAS_REQUIRED')
    try:
        k = cost_rates(schedule)
    except Blocked as exc:
        return result('BLOCKED', str(exc))
    if any(not isinstance(caps[a][key], (int, float)) or
           not np.isfinite(caps[a][key]) or caps[a][key] < 0
           for a in ALPHAS for key in ('VaR', 'ES')):
        return result('BLOCKED', 'APPROVED_RISK_CAP_NONNEGATIVE_LOSS_FRACTION_REQUIRED')
    r = np.asarray(scenarios, dtype=float)
    if r.ndim != 2 or r.shape[1] != 6 or len(r) < 2 or not np.isfinite(r).all() or np.min(r) <= -1:
        return result('BLOCKED', 'SIX_ASSET_ISSUED_SCENARIO_SUPPORT_INVALID')
    if mass is not None:
        mm = np.asarray(mass, dtype=float)
        if mm.shape != (len(r),) or not np.isfinite(mm).all() or mm.min() < 0 or abs(mm.sum()-1)>1e-10:
            return result('BLOCKED', 'SCENARIO_MASS_INVALID')
    rows = []
    for c in candidates:
        w = simplex(c['weights'], 6)
        cost = self_financing_cost(pre, w, k)['cost_fraction']
        net_w = (1 - cost) * (1 + r @ w)
        risks = {a: var_es(1 - net_w, a, mass) for a in ALPHAS}
        feasible = all(risks[a]['VaR'] <= caps[a]['VaR'] and risks[a]['ES'] <= caps[a]['ES'] for a in ALPHAS)
        eu = expected_utility(float(current_wealth) * net_w, utility_owner, mass)
        if eu['status'] != 'PASS' or not np.isfinite(eu['expected_utility']):
            return result('BLOCKED', 'NONFINITE_APPROVED_SCENARIO_EXPECTED_UTILITY',
                          candidate_action=c['action'])
        rows.append({'action': c['action'], 'param': c.get('param'), 'weights': w.tolist(),
                     'cost_fraction': cost, 'feasible': feasible,
                     'expected_utility': eu.get('expected_utility'),
                     'net_mean_return': float((np.mean(net_w) if mass is None else np.dot(mm, net_w)) - 1),
                     'risk': {a: (risks[a]['VaR'], risks[a]['ES']) for a in ALPHAS}})
    ok = [x for x in rows if x['feasible']]
    if not ok:
        return result('FAIL', 'NO_FEASIBLE_ACTION_INCLUDING_HOLD_ESCALATE', rows=rows, mode='MYOPIC_SCENARIO_OPTIMIZATION')
    best = max(ok, key=lambda x: x['expected_utility'])
    return result('PASS', best=best, rows=rows, mode='MYOPIC_SCENARIO_OPTIMIZATION',
                  evaluated_at_wealth=float(current_wealth), rl=RL_STATUS)


def max_drawdown(wealth_path) -> float:
    """Wealth-relative peak-to-trough drawdown, including the common start NAV."""
    v = np.asarray(wealth_path, dtype=float)
    if v.ndim != 1 or not len(v) or not np.isfinite(v).all() or (v <= 0).any():
        raise ValueError('FINITE_POSITIVE_NAV_PATH_REQUIRED')
    peaks = np.maximum.accumulate(v)
    return float(np.max(1 - v / peaks))


def matched_final_wealth(strategies: dict, prerequisites: dict) -> dict:
    """RQ03-28: five benchmarks + policy, same weeks, same tape, same start, NET only."""
    missing = [k for k in MATCH_PREREQUISITES if not owner_ok(prerequisites.get(k))]
    if missing:
        return result('BLOCKED', 'MATCHED_NET_WEALTH_PREREQUISITES_MISSING:' + ','.join(missing), missing=missing)
    expected = {'EW', 'IVOL', 'MVP', 'ERC', 'MaxDiv', 'POLICY'}
    if set(strategies) != expected:
        return result('BLOCKED', 'SIX_MATCHED_STRATEGIES_REQUIRED',
                      missing=sorted(expected - set(strategies)), unexpected=sorted(set(strategies) - expected))
    if any(not np.isfinite(s.get('terminal_wealth', float('nan'))) or
           s['terminal_wealth'] <= 0 for s in strategies.values()):
        return result('FAIL', 'NONPOSITIVE_OR_NONFINITE_NET_TERMINAL_WEALTH')
    keys = {(s['start_wealth'], s['weeks_sha'], s['tape_sha']) for s in strategies.values()}
    if len(keys) != 1 or any(s.get('mode') != 'NET' for s in strategies.values()):
        return result('FAIL', 'STRATEGIES_NOT_MATCHED_OR_NOT_NET')
    table = {k: s['terminal_wealth'] for k, s in strategies.items()}
    drawdowns = {}
    for k, s in strategies.items():
        path = s.get('wealth_path')
        if path is not None:
            try:
                if abs(float(path[0]) - float(s['start_wealth'])) > 1e-10 or \
                   abs(float(path[-1]) - float(s['terminal_wealth'])) > 1e-10:
                    return result('FAIL', 'NET_WEALTH_PATH_ENDPOINT_MISMATCH:' + k)
                drawdowns[k] = max_drawdown(path)
            except (ValueError, TypeError, IndexError):
                return result('FAIL', 'INVALID_NET_WEALTH_PATH:' + k)
    return result('PASS', terminal_wealth=table, max_drawdown=drawdowns,
                  ranking=sorted(table, key=table.get, reverse=True))


def gross_diagnostic_comparison(strategies: dict) -> dict:
    keys = {(s['start_wealth'], s['weeks_sha'], s['tape_sha']) for s in strategies.values()}
    if len(keys) != 1:
        return result('FAIL', 'STRATEGIES_NOT_MATCHED')
    return result('PASS', terminal_wealth={k: s['terminal_wealth'] for k, s in strategies.items()},
                  claim='GROSS_DIAGNOSTIC_NOT_NET', net_wealth='BLOCKED')


def bellman_step(wealth: float, pre, actions: list, *, kernel=None, continuation=None, feasible=None,
                 schedule: dict | None = None, beta: float | None = None, utility_owner: dict | None = None) -> dict:
    """RQ03-29: one finite-action step with a SUPPLIED kernel; no optimality claim beyond it."""
    if kernel is None or continuation is None or feasible is None or beta is None:
        return result('BLOCKED', 'IDENTIFIED_KERNEL_CONTINUATION_FEASIBILITY_DISCOUNT_REQUIRED', rl=RL_STATUS)
    try:
        k = cost_rates(schedule)
    except Blocked as e:
        return result('BLOCKED', str(e), rl=RL_STATUS)
    f = utility_fn(utility_owner) if utility_owner is not None else None
    if not 0 < beta <= 1:
        raise ValueError('DISCOUNT_DOMAIN')
    values = []
    for a in actions:
        w = simplex(a, 6)
        if not feasible(w):
            continue
        branches = list(kernel(w))
        mass = np.asarray([b[0] for b in branches], dtype=float)
        if not len(mass) or not np.isfinite(mass).all() or mass.min() < 0 or abs(mass.sum() - 1) > 1e-10:
            raise ValueError('INVALID_KERNEL_MASS')
        c = self_financing_cost(pre, w, k)['cost_fraction']
        q = 0.0
        for p, ret, nxt in branches:
            w_next = wealth * (1 - c) * (1 + float(w @ np.asarray(ret, float)))
            q += p * ((float(f(w_next)) if f else 0.0) + beta * continuation(w_next, nxt))
        values.append({'weights': w.tolist(), 'value': q, 'cost_fraction': c})
    if not values:
        return result('FAIL', 'NO_FEASIBLE_ACTION')
    return result('PASS', best=max(values, key=lambda x: x['value']), actions=values,
                  claim='SUPPLIED_KERNEL_FINITE_ACTION_STEP_ONLY')


def continuation_status(evidence: dict) -> dict:
    """RQ03-30 / D07: continuation value only with an identified sequential kernel."""
    need = ('transition_likelihood', 'action_support', 'reward_utility', 'discount')
    missing = [k for k in need if not owner_ok(evidence.get(k))]
    if missing:
        return result('BLOCKED', 'SEQUENTIAL_KERNEL_NOT_IDENTIFIED:' + ','.join(missing),
                      mode='MYOPIC_SCENARIO_OPTIMIZATION', rl=RL_STATUS, continuation_value=None)
    return result('PASS', mode='IDENTIFIED_KERNEL_CONTINUATION', rl=RL_STATUS)


def certainty_equivalent(eu: float, owner: dict) -> float:
    u, g = owner['utility'], owner.get('gamma')
    if u == 'LOG' or (u == 'CRRA' and g == 1):
        return math.exp(eu)
    if u == 'CARA':
        return -math.log(-eu) / g
    return (eu * (1 - g)) ** (1 / (1 - g))
