"""RQ03-23, RQ03-24, RQ03-26, D05: actions, self-financing costs, weekly wealth recursion.

Pretrade weights are DRIFTED holdings: p' = w (1 + R) / (1 + w'R).
Rebalance: w_post = (1 - lambda) p + lambda w_target, lambda in [0, 1].
Self-financing fee fraction c (of pretrade wealth W) solves the fixed point
    c = sum_i k_i |(1 - c) w_post_i - p_i|        (unique: slope <= max k - 1 < 0)
so post-trade holdings are W(1-c) w_post and net trade cash flows sum to -c W: the fee
is charged exactly ONCE. Wealth: W_{u+1} = W_u (1 - c_u)(1 + w_post' R_{u+1}), where the
return window starts strictly after the issue time u; consecutive windows do not overlap and all prior windows end before subsequent issues. No owner cost schedule -> BLOCKED;
cost-free paths exist only as GROSS_DIAGNOSTIC_NOT_NET.
"""
from __future__ import annotations
from datetime import datetime, timezone

import numpy as np
from scipy.optimize import brentq

from .contract import ASSETS, BONDS, EQUITY, Blocked, owner_ok, result, simplex, stamp, require_dev

ACTIONS = ('HOLD', 'CASH', 'BOND_HEDGE', 'PARTIAL', 'FULL', 'EMERGENCY_RISK_REPAIR')
GROSS = 'GROSS_DIAGNOSTIC_NOT_NET'


def drift(w_post, returns) -> np.ndarray:
    w = simplex(w_post, 6)
    r = np.asarray(returns, dtype=float)
    if r.shape != (6,) or not np.isfinite(r).all() or r.min() <= -1:
        raise ValueError('INVALID_HOLDING_RETURNS')
    return simplex(w * (1 + r) / (1 + w @ r), 6)


def rebalance(pre, target, lam: float) -> np.ndarray:
    if not 0 <= lam <= 1:
        raise ValueError('LAMBDA_DOMAIN')
    return simplex((1 - lam) * simplex(pre, 6) + lam * simplex(target, 6), 6)


def cost_rates(schedule: dict | None) -> np.ndarray:
    if not owner_ok(schedule):
        raise Blocked('AUTHENTIC_TRANSACTION_COST_SCHEDULE_MISSING')
    if not isinstance(schedule.get('rates'), dict) or \
       set(schedule['rates']) != set(ASSETS):
        raise Blocked('EXACT_SIX_ASSET_COST_RATES_REQUIRED')
    try:
        k = np.asarray([schedule['rates'][a] for a in ASSETS], dtype=float)
    except (TypeError, ValueError, KeyError) as exc:
        raise Blocked('UNPARSABLE_TRANSACTION_COST_RATES') from exc
    if not np.isfinite(k).all() or k.min() < 0 or k.max() >= 1:
        raise Blocked('INVALID_AUTHORIZED_COST_RATES')
    return k


def self_financing_cost(pre, post, k) -> dict:
    p, w = simplex(pre, 6), simplex(post, 6)
    k = np.asarray(k, dtype=float)
    if k.shape != (6,) or not np.isfinite(k).all() or np.min(k) < 0 or np.max(k) >= 1:
        raise ValueError('SIX_AUTHORIZED_COST_RATES_IN_0_TO_1_REQUIRED')

    def g(c):
        return float(k @ np.abs((1 - c) * w - p) - c)

    c = 0.0 if g(0.0) == 0 else float(brentq(g, 0.0, 1.0 - 1e-12, xtol=1e-15))
    trades = (1 - c) * w - p
    return result('PASS', cost_fraction=c, trades=trades.tolist(), net_trade_cash=float(trades.sum()),
                  identity_residual=float(trades.sum() + c), turnover=float(np.abs(trades).sum()))


def candidate_actions(pre, target, *, lambdas=(0.25, 0.5, 0.75), cash_fracs=(0.5, 1.0),
                      bond_proxy_owner: dict | None = None, emergency_owner: dict | None = None) -> dict:
    """D05: complete, labelled action set; unapproved branches are listed as BLOCKED, not dropped."""
    p, t = simplex(pre, 6), simplex(target, 6)
    out, blocked = [('HOLD', None, p)], []
    out.append(('FULL', 1.0, t))
    out += [('PARTIAL', lam, rebalance(p, t, lam)) for lam in lambdas if 0 < lam < 1]
    for x in cash_fracs:
        c = p.copy()
        move = c[:5].sum() * x
        c[:5] *= 1 - x
        c[5] += move
        out.append(('CASH', x, c))
    eq = [ASSETS.index(a) for a in EQUITY]
    for b in BONDS:
        if not owner_ok(bond_proxy_owner):
            blocked.append(('BOND_HEDGE', b, 'SYNTHETIC_BOND_PROXY_NOT_APPROVED_INVESTABLE'))
            continue
        for x in cash_fracs:
            h = p.copy()
            move = h[eq].sum() * x
            h[eq] *= 1 - x
            h[ASSETS.index(b)] += move
            out.append(('BOND_HEDGE', (b, x), h))
    if owner_ok(emergency_owner):
        out.append(('EMERGENCY_RISK_REPAIR', None, np.r_[np.zeros(5), 1.0]))
    else:
        blocked.append(('EMERGENCY_RISK_REPAIR', None, 'EMERGENCY_OVERRIDE_OWNER_MISSING'))
    return result('PASS', candidates=[{'action': a, 'param': x, 'weights': w.tolist()} for a, x, w in out],
                  blocked=[{'action': a, 'param': x, 'reason': r} for a, x, r in blocked],
                  action_set=list(ACTIONS), no_trade_included=True)


def _window_start(rec) -> datetime:
    d = stamp(rec['start']).date()
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)


def wealth_step(wealth: float, pre, post, ret_record: dict, issue_at, *, schedule: dict | None = None,
                cash_owner: dict | None = None, mode: str = 'NET') -> dict:
    """ret_record: {'start': first day of holding window, 'end', 'returns': six returns}."""
    if not np.isfinite(wealth) or wealth <= 0:
        raise ValueError('INVALID_WEALTH')
    if not all(key in ret_record for key in ('start', 'end', 'returns')):
        return result('BLOCKED', 'RETURN_WINDOW_START_END_RETURNS_REQUIRED', issue_times=[issue_at])
    try:
        require_dev(issue_at)
        require_dev(ret_record['start'])
        require_dev(ret_record['end'])
    except Blocked as exc:
        return result('BLOCKED', str(exc), issue_times=[issue_at])
    start = _window_start(ret_record)
    end = stamp(ret_record['end'])
    if end < start:
        return result('FAIL', 'RETURN_WINDOW_END_BEFORE_START', issue_times=[issue_at])
    if start <= stamp(issue_at):
        return result('FAIL', 'RETURN_WINDOW_NOT_STRICTLY_AFTER_ISSUE', issue_times=[issue_at])
    w = simplex(post, 6)
    r = np.asarray(ret_record['returns'], dtype=float)
    if r.shape != (6,) or not np.isfinite(r).all() or np.any(r <= -1):
        return result('FAIL', 'INVALID_HOLDING_RETURNS', issue_times=[issue_at])
    if mode == 'NET':
        if w[5] > 0 and not owner_ok(cash_owner):
            return result('BLOCKED', 'CASH_FUNDING_OWNER_UNAPPROVED', issue_times=[issue_at])
        try:
            k = cost_rates(schedule)
        except Blocked as e:
            return result('BLOCKED', str(e), issue_times=[issue_at])
        cost = self_financing_cost(pre, w, k)
    elif mode == GROSS:
        cost = result('PASS', cost_fraction=0.0, trades=(w - simplex(pre, 6)).tolist())
    else:
        raise ValueError('UNKNOWN_WEALTH_MODE')
    c = cost['cost_fraction']
    gross_ret = float(w @ r)
    after = float(wealth * (1 - c) * (1 + gross_ret))
    return result('PASS', issue_times=[issue_at], wealth=after, wealth_pre=float(wealth),
                  cost_fraction=c, cost_money=float(wealth * c), portfolio_return=gross_ret,
                  net_return=after / wealth - 1, pretrade_weights=simplex(pre, 6).tolist(),
                  posttrade_weights=w.tolist(), turnover=cost.get('turnover'),
                  signed_trades=cost.get('trades'), pretrade_next=drift(w, r).tolist(), mode=mode)


def wealth_path(w0: float, initial_pre, decisions: list[dict], *, schedule: dict | None = None,
                cash_owner: dict | None = None, mode: str = 'NET') -> dict:
    """decisions: [{'issue_at', 'target', 'lambda', 'ret': ret_record}] in time order."""
    wealth, pre = float(w0), simplex(initial_pre, 6)
    path, last_issue, last_end = [], None, None
    for d in decisions:
        issue = stamp(d['issue_at'])
        if last_issue is not None and issue <= last_issue:
            raise ValueError('NONINCREASING_ISSUE_TIMES')
        last_issue = issue
        ret = d['ret']
        if not all(k in ret for k in ('start', 'end', 'returns')):
            return result('BLOCKED', 'RETURN_WINDOW_START_END_RETURNS_REQUIRED',
                          path=path, stopped_at=str(d['issue_at']))
        start, end = _window_start(ret), stamp(ret['end'])
        if last_end is not None and start <= last_end:
            return result('FAIL', 'OVERLAPPING_RETURN_WINDOWS',
                          path=path, stopped_at=str(d['issue_at']))
        if last_end is not None and issue <= last_end:
            return result('FAIL', 'NEXT_ISSUE_BEFORE_PRIOR_RETURN_END',
                          path=path, stopped_at=str(d['issue_at']))
        post = rebalance(pre, d['target'], d.get('lambda', 1.0))
        step = wealth_step(wealth, pre, post, d['ret'], d['issue_at'], schedule=schedule, cash_owner=cash_owner,
                           mode=mode)
        if step['status'] != 'PASS':
            return result(step['status'], step['reason'], path=path, stopped_at=str(d['issue_at']))
        wealth, pre = step['wealth'], np.asarray(step['pretrade_next'])
        last_end = end
        path.append({'issue_at': str(d['issue_at']), 'return_start': str(ret['start']),
                     'return_end': str(ret['end']), 'wealth': wealth,
                     'cost_fraction': step['cost_fraction'], 'net_return': step['net_return'],
                     'turnover': step['turnover'], 'cost_money': step['cost_money']})
    return result('PASS', terminal_wealth=wealth, path=path, mode=mode,
                  claim='NET_WEALTH' if mode == 'NET' else GROSS)
