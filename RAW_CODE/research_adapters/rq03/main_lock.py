"""RQ03 R23: Main-only Short-Confirmed -> Bottom-Confirmed restructuring lock.

From the first SHORT_CONFIRMED that is already AVAILABLE at the decision time until the first decision
at which a BOTTOM_CONFIRMED of the SAME episode is available, the Main model (MAIN == POLICY) takes NO
discretionary restructuring action: no Common rebalance, no Bottom overlay, no optimiser call, no fill.
NAV and holdings drift passively, turnover and cost are exactly 0, the risk monitor keeps running and only
ESCALATES (LOCKED_RISK_BREACH); it never trades.  The five benchmarks EW/IVOL/MVP/ERC/MaxDiv are EXEMPT:
they never read or write the Main lock state, so their ledgers are identical with and without the events.

Live and PIT use the FIRST available Short confirmation; "the last Short before Bottom" is hindsight and is
only a post-hoc audit label (`retrospective_last_short`).  There is no timeout, price trigger or
"assume Normal" release: without an available same-episode Bottom the lock persists.

Integration contract (issued_chain.execute and new_workflow_rev357.run_six_issued_net_paths_v1):
  * lock_events_from_packet(packet)  - issued SHORT/BOTTOM/RECOVERY/EXIT/EPISODE_ENDED records -> events
                                       (a Short without episode_id fails closed: BLOCKED_INPUT_CONFLICT);
  * gate_strategy('POLICY', state, events, issue)  - MAIN only; benchmarks short-circuit untouched;
  * benchmark_zone()                 - regime-only zone for the five benchmarks (no Short/Bottom effect);
  * main_unlocked_zone(...)          - Common ON zone for an unlocked Main, Bottom overlay only per asset
                                       with a granted permission and only in BOTTOM/RECOVERY phases;
  * passive_locked_step / locked_risk_monitor / locked_tail_check - the NO_TRADE_LOCKED ledger.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from . import regime_policy as P
from .contract import ALPHAS, METHODS, RISKY, STRESS, Blocked, owner_ok, require_dev, result, simplex, stamp
from .tail import portfolio_tail
from .trading import cost_rates

BENCHMARKS = tuple(METHODS)                        # EW, IVOL, MVP, ERC, MaxDiv
MAIN_IDS = ('MAIN', 'POLICY')
EVENT_TYPES = ('SHORT_CONFIRMED', 'BOTTOM_CONFIRMED', 'RECOVERY_CONFIRMED', 'EXIT_CONFIRMED', 'EPISODE_ENDED')

ACTIVE_PRE_SHORT = 'ACTIVE_PRE_SHORT'
FROZEN = 'FROZEN_SHORT_TO_BOTTOM'
BOTTOM_ACTIVE = 'BOTTOM_CONFIRMED_ACTIVE'
RECOVERY_ACTIVE = 'RECOVERY_ACTIVE'
EXIT_ACTIVE = 'EXIT_ACTIVE'
POST_EXIT = 'POST_EXIT_ACTIVE'
PHASES = (ACTIVE_PRE_SHORT, FROZEN, BOTTOM_ACTIVE, RECOVERY_ACTIVE, EXIT_ACTIVE, POST_EXIT)
OVERLAY_PHASES = (BOTTOM_ACTIVE, RECOVERY_ACTIVE)

# per-phase switches (spec R23 section 3 table)
F_COMMON = {ACTIVE_PRE_SHORT: True, FROZEN: False, BOTTOM_ACTIVE: True, RECOVERY_ACTIVE: True,
            EXIT_ACTIVE: True, POST_EXIT: True}
F_BOTTOM = {ACTIVE_PRE_SHORT: 'OFF', FROZEN: 'OFF', BOTTOM_ACTIVE: 'ELIGIBLE', RECOVERY_ACTIVE: 'ELIGIBLE',
            EXIT_ACTIVE: 'OFF', POST_EXIT: 'OFF'}

NO_TRADE_LOCKED = 'NO_TRADE_LOCKED'
ALLOWED = 'MAIN_RESTRUCTURE_ALLOWED_SUBJECT_TO_ALL_GATES'
BENCHMARK_UNAFFECTED = 'BENCHMARK_UNAFFECTED'
DECISION_LABELS = (NO_TRADE_LOCKED, 'HOLD_BY_UTILITY', 'BLOCKED_INVALID_INPUT', 'BLOCKED_RISK_BREACH')
HARD_ENVELOPE = {'asset_cap': 0.60, 'normal_band': (0.50, 0.75), 'stress_max': 0.50}


@dataclass
class MainLockState:
    phase: str = ACTIVE_PRE_SHORT
    episode_id: str | None = None
    last_short_confirmed_at: object = None        # event time of the newest Short seen in this lock
    last_short_available_at: object = None
    first_short_available_at: object = None
    bottom_confirmed_at: object = None
    seen: set = field(default_factory=set)
    seen_core: dict = field(default_factory=dict)  # event_id -> payload core (a changed vintage is a conflict)
    trace: list = field(default_factory=list)
    closed_episodes: set = field(default_factory=set)


def _blocked(code: str, **extra) -> dict:
    return {'status': code, 'effective': code, **extra}


def lock_events_from_packet(packet: dict) -> list:
    """Map an issued packet to lock events.  Missing episode ids are kept (None) so validation blocks.

    The legacy RQ02 `exit_record` (first-N forecast) is NOT a lock event: an EXIT before a Bottom never
    releases the Main lock.  R23 EXIT/RECOVERY/EPISODE_ENDED events come only from their explicit lists.
    """
    out = []
    for s in packet.get('short_confirmations', []) or []:
        if s.get('confirmed') is True:
            out.append({'event_id': s.get('short_id') or s.get('record_id'), 'type': 'SHORT_CONFIRMED',
                        'episode_id': s.get('episode_id'), 'issued_at': s.get('issued_at'),
                        'confirmed_at': s.get('confirmed_at', s.get('issued_at')),
                        'available_at': s.get('available_at', s.get('issued_at')),
                        'issuer_seq': s.get('issuer_seq'), 'status': s.get('status')})
    for key, etype in (('bottom_confirmations', 'BOTTOM_CONFIRMED'), ('recovery_confirmations', 'RECOVERY_CONFIRMED'),
                       ('exit_confirmations', 'EXIT_CONFIRMED'), ('episode_end_records', 'EPISODE_ENDED')):
        for b in packet.get(key, []) or []:
            out.append({'event_id': b.get('record_id') or b.get('event_id'), 'type': etype,
                        'episode_id': b.get('episode_id'), 'issued_at': b.get('issued_at'),
                        'confirmed_at': b.get('confirmed_at', b.get('issued_at')),
                        'available_at': b.get('available_at', b.get('issued_at')),
                        'issuer_seq': b.get('issuer_seq'), 'status': b.get('status')})
    return out


def _core(e: dict) -> tuple:
    return (e['type'], e['episode_id'], e['_avail'], e['_conf'])


def visible_events(events, decision_at) -> dict:
    """Only events whose publication AND availability precede the decision are visible (as-of)."""
    t = stamp(decision_at)
    vis = []
    for e in events or []:
        for req in ('event_id', 'type', 'episode_id', 'issued_at', 'confirmed_at', 'available_at'):
            if not e.get(req):
                return _blocked('BLOCKED_INPUT_CONFLICT', reason='EVENT_FIELD_MISSING:' + req)
        if e['type'] not in EVENT_TYPES:
            return _blocked('BLOCKED_INPUT_CONFLICT', reason='UNKNOWN_EVENT_TYPE:' + str(e['type']))
        try:
            issued, avail, conf = stamp(e['issued_at']), stamp(e['available_at']), stamp(e['confirmed_at'])
        except (TypeError, ValueError):
            return _blocked('BLOCKED_INPUT_CONFLICT', reason='EVENT_CLOCK_INVALID')
        if conf > issued or avail < issued:
            return _blocked('BLOCKED_INPUT_CONFLICT', reason='EVENT_CLOCK_ORDER_INVALID')
        if e.get('status') == 'WITHDRAWN':
            continue
        if avail > t or issued > t:
            continue                       # known only in hindsight: invisible, never a retroactive trade
        vis.append({**e, '_avail': avail, '_conf': conf})
    uniq = {}
    for e in vis:
        # an identical snapshot repeated inside one packet is fine; a different payload under one id is not
        if uniq.setdefault(e['event_id'], e) is not e and _core(uniq[e['event_id']]) != _core(e):
            return _blocked('BLOCKED_INPUT_CONFLICT', reason='EVENT_ID_REUSED_WITH_DIFFERENT_PAYLOAD')
    vis = list(uniq.values())
    vis.sort(key=lambda e: (e['_avail'], e['issuer_seq'] if e.get('issuer_seq') is not None else -1))
    for a, b in zip(vis, vis[1:]):
        if a['_avail'] == b['_avail'] and a['type'] != b['type'] and a['episode_id'] == b['episode_id'] \
                and (a.get('issuer_seq') is None or b.get('issuer_seq') is None or a['issuer_seq'] == b['issuer_seq']):
            return _blocked('BLOCKED_AMBIGUOUS_ORDER', reason='SAME_AVAILABLE_TIME_WITHOUT_PROVEN_SEQUENCE',
                            event_ids=[a['event_id'], b['event_id']])
    return {'status': 'OK', 'events': vis}


def _step(state: MainLockState, e: dict):
    """Apply one visible event to `state` in place; return an error dict or None."""
    typ, ep = e['type'], e['episode_id']
    if typ == 'SHORT_CONFIRMED':
        if state.phase == FROZEN:
            if ep != state.episode_id:
                return _blocked('BLOCKED_EPISODE_MISMATCH', reason='SECOND_SHORT_FOR_DIFFERENT_EPISODE')
            # an additional Short before Bottom only moves the audit anchor; the lock stays continuous
            if state.last_short_confirmed_at is None or e['_conf'] > state.last_short_confirmed_at:
                state.last_short_confirmed_at = e['_conf']
            state.last_short_available_at = max(state.last_short_available_at, e['_avail'])
            return None
        if ep in state.closed_episodes:
            return _blocked('BLOCKED_EPISODE_MISMATCH', reason='SHORT_REUSES_A_RELEASED_EPISODE_ID')
        state.phase, state.episode_id = FROZEN, ep
        state.last_short_confirmed_at = e['_conf']
        state.first_short_available_at = state.last_short_available_at = e['_avail']
        state.bottom_confirmed_at = None
        return None
    if typ == 'BOTTOM_CONFIRMED':
        if state.phase != FROZEN:
            return _blocked('BLOCKED_MISSING_SHORT_EPISODE')
        if ep != state.episode_id:
            return _blocked('BLOCKED_EPISODE_MISMATCH', reason='BOTTOM_FOR_ANOTHER_EPISODE')
        if e['_conf'] < state.last_short_confirmed_at:
            return _blocked('BLOCKED_AMBIGUOUS_ORDER', reason='BOTTOM_CONFIRMED_BEFORE_SHORT_CONFIRMED')
        state.phase, state.bottom_confirmed_at = BOTTOM_ACTIVE, e['_avail']
        state.closed_episodes.add(ep)
        return None
    chain = {'RECOVERY_CONFIRMED': ((BOTTOM_ACTIVE,), RECOVERY_ACTIVE),
             'EXIT_CONFIRMED': ((BOTTOM_ACTIVE, RECOVERY_ACTIVE), EXIT_ACTIVE),
             'EPISODE_ENDED': ((EXIT_ACTIVE,), POST_EXIT)}[typ]
    if state.phase not in chain[0] or ep != state.episode_id:
        return _blocked('BLOCKED_EPISODE_MISMATCH' if state.episode_id not in (None, ep)
                        else 'BLOCKED_MISSING_SHORT_EPISODE', reason=typ + '_OUT_OF_SEQUENCE')
    state.phase = chain[1]
    return None


def apply_events(state: MainLockState, events, decision_at) -> dict:
    """Update `state` with the events visible at `decision_at`.  On any block the state is left UNCHANGED."""
    vis = visible_events(events, decision_at)
    if vis['status'] != 'OK':
        return vis
    work = copy.deepcopy(state)
    for e in vis['events']:
        if e['event_id'] in work.seen:
            if work.seen_core.get(e['event_id']) != _core(e):
                return _blocked('BLOCKED_INPUT_CONFLICT', reason='EVENT_VINTAGE_CHANGED_ACROSS_PACKETS',
                                event_id=e['event_id'])
            continue
        err = _step(work, e)
        if err is not None:
            return {**err, 'event_id': e['event_id']}
        work.seen.add(e['event_id'])
        work.seen_core[e['event_id']] = _core(e)
        work.trace.append({'event_id': e['event_id'], 'type': e['type'], 'episode_id': e['episode_id'],
                           'available_at': e['_avail'].isoformat(), 'phase_after': work.phase})
    state.__dict__.update(work.__dict__)
    return {'status': 'OK', 'phase': state.phase}


def gate_strategy(strategy_id: str, state: MainLockState, events, decision_at) -> dict:
    """Strategy router.  Benchmarks short-circuit BEFORE the events or the Main state are touched."""
    if strategy_id in BENCHMARKS:
        return {'status': 'OK', 'effective': BENCHMARK_UNAFFECTED, 'locked': False, 'F_common': True,
                'F_bottom': 'NOT_APPLICABLE', 'main_state_read': False}
    if strategy_id not in MAIN_IDS:
        return _blocked('BLOCKED_UNKNOWN_STRATEGY')
    res = apply_events(state, events, decision_at)
    if res['status'] != 'OK':
        return {**res, 'locked': None, 'F_common': False, 'F_bottom': 'OFF', 'trade_allowed': False,
                'main_state_read': True}
    locked = state.phase == FROZEN
    return {'status': 'OK', 'effective': NO_TRADE_LOCKED if locked else ALLOWED, 'phase': state.phase,
            'locked': locked, 'F_common': F_COMMON[state.phase], 'F_bottom': F_BOTTOM[state.phase],
            'episode_id': state.episode_id, 'last_short_confirmed_at': state.last_short_confirmed_at,
            'bottom_confirmed_at': state.bottom_confirmed_at, 'main_state_read': True,
            'trade_allowed': not locked}


def retrospective_last_short(short_confirmed_at: list, bottom_confirmed_at) -> object:
    """Post-hoc audit label t_S,last = max{t_S < t_B}.  NEVER usable to decide whether to wait live."""
    t_b = stamp(bottom_confirmed_at)
    prior = [stamp(s) for s in short_confirmed_at if stamp(s) < t_b]
    return max(prior) if prior else None


# ----------------------------------------------------------------------------- zones for the two classes
def benchmark_zone() -> dict:
    """Regime-only zone of the five EXEMPT benchmarks: no Short/Bottom/permission dependence at all."""
    return {'status': 'PASS', 'zone': 'NORMAL', 'assets': {a: 'NORMAL' for a in RISKY},
            'basis': 'REGIME_ONLY_R23_BENCHMARK_EXEMPT'}


def main_unlocked_zone(phase: str, issue, bottoms: dict, permissions: dict, *, latest_short=None) -> dict:
    """Zone of an UNLOCKED Main: Common is ON everywhere; the Bottom overlay is ELIGIBLE only in the
    BOTTOM_CONFIRMED/RECOVERY phases and only for assets whose issued Bottom AND granted permission are
    in force.  Otherwise the asset is plain 'NORMAL' (a confirmed Bottom with no permission is a HOLD for
    the overlay, never a forced buy and never a reason to keep Common off)."""
    if phase not in PHASES:
        raise ValueError('UNKNOWN_MAIN_PHASE')
    if phase == FROZEN:
        raise ValueError('LOCKED_MAIN_HAS_NO_TRADING_ZONE')
    assets = {a: 'NORMAL' for a in RISKY}
    if phase in OVERLAY_PHASES:
        latest = None if latest_short is None else stamp(latest_short)
        for a in RISKY:
            z = P.asset_bottom_zone(a, issue, bottoms or {}, permissions or {}, latest)
            if z in ('BOTTOM_REENTRY_ZONE', 'POST_BOTTOM_STRESS'):
                assets[a] = z
    vals = set(assets.values())
    label = ('BOTTOM_REENTRY_ZONE' if 'BOTTOM_REENTRY_ZONE' in vals else
             'POST_BOTTOM_STRESS' if 'POST_BOTTOM_STRESS' in vals else 'NORMAL')
    return {'status': 'PASS', 'zone': label, 'assets': assets, 'lock_phase': phase,
            'overlay_eligible': phase in OVERLAY_PHASES, 'basis': 'R23_MAIN_UNLOCKED'}


# ----------------------------------------------------------------------------- the passive (locked) ledger
def passive_locked_step(weights_pre, returns_next, nav: float) -> dict:
    """No fill: turnover = cost = 0, X_t = X_{t-}, NAV_{t+1} = NAV_t (1 + X'r), weights drift."""
    w = simplex(weights_pre, 6)
    r = np.asarray(returns_next, dtype=float)
    if r.shape != (6,) or not np.isfinite(r).all() or r.min() <= -1:
        raise ValueError('INVALID_HOLDING_RETURNS')
    if not np.isfinite(nav) or nav <= 0:
        raise ValueError('INVALID_NAV')
    gross = float(w @ r)
    nxt = w * (1 + r) / (1 + gross)
    return {'action': NO_TRADE_LOCKED, 'turnover': 0.0, 'transaction_cost': 0.0, 'cost_fraction': 0.0,
            'weights_pre': w.tolist(), 'weights_post': w.tolist(), 'portfolio_return': gross,
            'nav_pre': float(nav), 'nav_next': float(nav * (1 + gross)), 'weights_next': simplex(nxt, 6).tolist(),
            'signed_trades': [0.0] * 6}


def locked_wealth_step(wealth: float, pre, ret_record: dict, issue_at, *, schedule: dict | None = None,
                       cash_owner: dict | None = None) -> dict:
    """NET wealth step of a LOCKED week with the same date/owner gates as trading.wealth_step.

    The ledger of a locked week is EXACTLY passive: post == pre, turnover = 0.0 and cost = 0.0 (not
    ~1e-16 as a self-financing-cost solve over a re-normalised vector can return), NAV and weights
    follow passive_locked_step.  Returns the wealth_step field names the runners consume."""
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
    d = stamp(ret_record['start']).date()
    start = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
    if stamp(ret_record['end']) < start:
        return result('FAIL', 'RETURN_WINDOW_END_BEFORE_START', issue_times=[issue_at])
    if start <= stamp(issue_at):
        return result('FAIL', 'RETURN_WINDOW_NOT_STRICTLY_AFTER_ISSUE', issue_times=[issue_at])
    w = simplex(pre, 6)
    if w[5] > 0 and not owner_ok(cash_owner):
        return result('BLOCKED', 'CASH_FUNDING_OWNER_UNAPPROVED', issue_times=[issue_at])
    if schedule is not None:
        try:
            cost_rates(schedule)
        except Blocked as exc:
            return result('BLOCKED', str(exc), issue_times=[issue_at])
    try:
        p = passive_locked_step(w, ret_record['returns'], wealth)
    except ValueError:
        return result('FAIL', 'INVALID_HOLDING_RETURNS', issue_times=[issue_at])
    return result('PASS', issue_times=[issue_at], wealth=p['nav_next'], wealth_pre=float(wealth),
                  cost_fraction=0.0, cost_money=0.0, portfolio_return=p['portfolio_return'],
                  net_return=p['nav_next'] / wealth - 1, pretrade_weights=p['weights_pre'],
                  posttrade_weights=p['weights_post'], turnover=0.0, signed_trades=p['signed_trades'],
                  pretrade_next=p['weights_next'], mode='NET_PASSIVE_NO_TRADE')


def held_tail_table(scenarios, weights, *, alphas=ALPHAS, mass=None) -> dict:
    """{alpha: {'VaR','ES',...}} of the HELD weights (no trade, so gross == net)."""
    return portfolio_tail(scenarios, weights, alphas=tuple(alphas), mass=mass)


def locked_tail_check(scenarios, weights, caps_by_alpha: dict, *, mass=None):
    """VaR/ES of the HELD (not traded) weights against the owner caps -> (measures, caps) for the monitor.

    caps_by_alpha: {alpha: {'VaR': cap, 'ES': cap}}.  The loss basis is gross = net because no trade."""
    tails = portfolio_tail(scenarios, weights, alphas=tuple(caps_by_alpha), mass=mass)
    measures, caps = {}, {}
    for a, c in caps_by_alpha.items():
        for k in ('VaR', 'ES'):
            measures[f'{k}@{a}'] = float(tails[a][k])
            caps[f'{k}@{a}'] = float(c[k])
    return measures, caps


def locked_risk_monitor(weights, regime: str, *, envelope: dict | None = None, asset_caps: dict | None = None,
                        risk_measures: dict | None = None, risk_caps: dict | None = None,
                        emergency_override: dict | None = None) -> dict:
    """Keeps measuring while locked.  A breach is ESCALATED, never repaired by a trade."""
    env = {**HARD_ENVELOPE, **(envelope or {})}
    w = simplex(weights, 6)
    risky = float(w[:5].sum())
    breaches = []
    for i, a in enumerate(RISKY):
        cap = (asset_caps or {}).get(a, env['asset_cap'])
        if w[i] > cap + 1e-12:
            breaches.append('ASSET_CAP:' + a)
    if regime in STRESS:
        if risky > env['stress_max'] + 1e-12:
            breaches.append('STRESS_RISKY_SHARE')
    elif regime == 'N':
        lo, hi = env['normal_band']
        if not lo - 1e-12 <= risky <= hi + 1e-12:
            breaches.append('NORMAL_RISKY_BAND')
    else:
        return {'status': 'NOT_EVALUATED', 'reason': 'REGIME_UNKNOWN', 'risky_share': risky, 'breaches': [],
                'action': 'NONE'}
    for k, cap in (risk_caps or {}).items():
        v = (risk_measures or {}).get(k)
        if v is None:
            return {'status': 'NOT_EVALUATED', 'reason': 'RISK_MEASURE_MISSING:' + k, 'risky_share': risky,
                    'breaches': breaches, 'action': 'NONE'}
        if v > cap + 1e-12:
            breaches.append(k)
    if not breaches:
        return {'status': 'OK', 'risky_share': risky, 'breaches': [], 'action': 'NONE'}
    override = isinstance(emergency_override, dict) and emergency_override.get('approved') is True \
        and bool(emergency_override.get('owner_hash'))
    return {'status': 'LOCKED_RISK_BREACH', 'risky_share': risky, 'breaches': breaches,
            'action': 'ESCALATE_TO_OWNER', 'forced_deleveraging': False,
            'separate_emergency_mandate_present': override,
            'note': 'a trade needs its own owner-signed emergency mandate; it is never inferred from this lock'}


def contract_violation_check(fills) -> dict:
    """A real fill while locked is an audit CONTRACT_VIOLATION, not a valid trade."""
    traded = [f for f in (fills or []) if abs(float(f.get('quantity', f.get('notional', 0.0)) or 0.0)) > 0]
    if traded:
        return {'status': 'CONTRACT_VIOLATION', 'n_fills': len(traded)}
    return {'status': 'OK', 'n_fills': 0}


def total_two_layer_cost(c_common: float, c_bottom: float) -> dict:
    """C_total = C_G + (1 - C_G) C_B (Bottom fee is on the NAV AFTER Common); V_post = (1-C_G)(1-C_B)."""
    for c in (c_common, c_bottom):
        if not (np.isfinite(c) and 0 <= c < 1):
            raise ValueError('FEE_FRACTION_DOMAIN')
    total = c_common + (1 - c_common) * c_bottom
    post = (1 - c_common) * (1 - c_bottom)
    if abs(1 - total - post) > 1e-15:
        raise AssertionError('TWO_LAYER_FEE_IDENTITY')
    return {'C_total': total, 'V_post': post, 'naive_sum': c_common + c_bottom}
