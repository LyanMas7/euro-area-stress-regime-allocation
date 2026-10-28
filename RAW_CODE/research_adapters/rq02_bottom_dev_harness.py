"""DEV-only Bottom 3F research harness built on the canonical issue-time functions.

Replaces the earlier in-line implementations in rev363/rq02.py::bottom_analysis and
rq02_next.py::bottom_zones, which (spec RQ02 Bottom, 6.3 and 12.2)
  * took the 'trough' as argmin of weekly RETURNS (worst week, not lowest level),
  * measured the 4-week price return as pct_change(4) of a return column,
  * shared one EA5 F3 across the five assets and aggregated it with groupby(month).max()
    (a signal from a week after the issue leaked into the issue),
  * started the zone AT the issue date instead of strictly after it, and
  * reported the interval score in days against an alpha of 0.05.
Here F3 is asset-specific (returns volatility of the asset's own level path), taken from the newest
week <= issue, the zone starts strictly after the issue, the trough is the lowest INDEX LEVEL inside
a strictly future window, and the interval score is in weeks.  EA5 survives only as a labelled control.

ISSUE SELECTION IS STILL EX-POST: the first issued month after (episode start + 3 months) uses the ESRB
episode start, so this harness measures zone quality and is NOT an issued forecast tape.  The flag is
written into every row.
"""
from __future__ import annotations

import calendar

import numpy as np
import pandas as pd

from . import rq02_bottom_factors as BF

DEV_END = pd.Timestamp('2019-12-31')
ISSUE_SELECTION = 'FIRST_SIGNAL_AFTER_EPISODE_START_PLUS_3M__EXPOST_EPISODE_LABEL_NOT_AN_ISSUED_TAPE'
TRUTH_WINDOW_POLICY = 'STRICT_FUTURE_ISSUE_TO_EPISODE_NORMAL_PLUS_6M_CAPPED_AT_DEV_END'
CONTROL = 'EA5'


def _month_end(month: str) -> pd.Timestamp:
    y, m = int(month[:4]), int(month[5:7])
    return pd.Timestamp(y, m, calendar.monthrange(y, m)[1])


def _month_add(month: str, k: int) -> str:
    idx = int(month[:4]) * 12 + int(month[5:7]) - 1 + k
    return f'{idx // 12:04d}-{idx % 12 + 1:02d}'


def asset_level_frame(weekly_returns: pd.DataFrame) -> pd.DataFrame:
    """Native INDEX LEVELS from the weekly return tape (cumprod of 1+r); DEV only."""
    if weekly_returns.empty or not weekly_returns.index.is_monotonic_increasing:
        raise ValueError('SORTED_WEEKLY_RETURN_TAPE_REQUIRED')
    if pd.Timestamp(weekly_returns.index.max()) > DEV_END:
        raise PermissionError('OOS_SEALED: weekly tape extends past 2019-12-31')
    missing = [a for a in BF.ASSETS if a not in weekly_returns.columns]
    if missing:
        raise ValueError('MISSING_ASSET_COLUMNS:' + ','.join(missing))
    r = weekly_returns[list(BF.ASSETS)].astype(float)
    if not np.isfinite(r.to_numpy()).all() or (r <= -1).any().any():
        raise ValueError('WEEKLY_RETURNS_MUST_BE_FINITE_AND_ABOVE_MINUS_ONE')
    lv = (1.0 + r).cumprod()
    lv[CONTROL] = lv[list(BF.ASSETS)].mean(axis=1)           # control composite, never one of the five outputs
    return lv


def _f3_series(levels: pd.Series, issue: pd.Timestamp):
    sub = levels[levels.index <= issue]
    if len(sub) < BF.MIN_WEEKLY_LEVELS:
        raise BF.BottomEligibilityError('F3_NEEDS_53_POSITIVE_WEEKLY_LEVELS')
    return BF.compute_f3_asset_cooling(sub.iloc[-BF.MIN_WEEKLY_LEVELS:].tolist())


def issue_month_signal(panel_by_key: dict, peers_by_month: dict, country: str, month: str, asset: str,
                       levels: pd.Series, native_weeks, target_window_end: pd.Timestamp) -> dict:
    issue = _month_end(month)
    row = panel_by_key.get((country, month))
    if row is None:
        return {'status': 'NOT_EVALUATED', 'reason': 'NO_RQ01_PANEL_ROW', 'issued_zone': False}
    phase, dd, z = row.get('phase'), row.get('dd_CLIFS'), row.get('z_CISS')
    phase_ok = isinstance(phase, str) and bool(phase)
    values_ok = dd is not None and z is not None and np.isfinite(dd) and np.isfinite(z)
    try:
        f1 = BF.compute_f1_ewi_cooling(phase, dd if values_ok else 0.0, z if values_ok else 0.0,
                                       phase_eligible=phase_ok, values_eligible=values_ok)
        f2 = BF.compute_f2_peer_recovery(country, peers_by_month.get(month, {}))
        f3 = _f3_series(levels, issue)
        vote = BF.factor_vote(f1['F1'], f2['F2'], f3['F3'])
        zone = BF.predict_bottom_zone(issue=issue.date(), factor_vote_2of3=vote, native_weeks=native_weeks,
                                      target_window_end=target_window_end.date())
    except BF.BottomEligibilityError as exc:
        return {'status': 'NOT_EVALUATED', 'reason': str(exc), 'issued_zone': False}
    return {'status': 'EVALUATED', 'f1': f1['F1'], 'f2': f2['F2'], 'f3': f3['F3'],
            'votes': f1['F1'] + f2['F2'] + f3['F3'], **zone}


def bottom_dev_zones(panel: pd.DataFrame, episodes, weekly_returns: pd.DataFrame, *, alpha: float = 0.1,
                     zone_months: int = 3, start_offset_months: int = 3, window_tail_months: int = 6) -> pd.DataFrame:
    """One row per (episode, asset incl. the EA5 control).  No OOS data may be passed in."""
    if zone_months != 3:
        raise ValueError('BASELINE_ZONE_IS_THREE_CALENDAR_MONTHS')
    lv = asset_level_frame(weekly_returns)
    weeks = [pd.Timestamp(d).date() for d in lv.index]
    last_week = pd.Timestamp(lv.index.max())
    need = {'country', 'month', 'phase', 'dd_CLIFS', 'z_CISS'}
    if not need.issubset(panel.columns):
        raise ValueError('PANEL_COLUMNS_MISSING:' + ','.join(sorted(need - set(panel.columns))))
    pan = panel[list(need)].copy()
    pan = pan[pan['month'] <= DEV_END.strftime('%Y-%m')]
    by_key = {(r['country'], r['month']): r for r in pan.to_dict('records')}
    peers: dict = {}
    for r in pan.to_dict('records'):
        if isinstance(r['phase'], str) and r['phase']:
            peers.setdefault(r['month'], {})[r['country']] = r['phase']
    rows = []
    for e in episodes:
        c = e['country']
        if c not in BF.EA7:
            continue
        start = pd.Timestamp(e['start'])
        normal = pd.Timestamp(e['normal'])
        if start > DEV_END:
            continue
        win_end = min(normal + pd.DateOffset(months=window_tail_months), DEV_END)
        s_month = start.strftime('%Y-%m')
        first_issue_month = _month_add(s_month, start_offset_months)
        last_issue_month = win_end.strftime('%Y-%m')
        for asset in (*BF.ASSETS, CONTROL):
            base = {'country': c, 'asset': asset, 'episode_start': start.strftime('%Y-%m-%d'),
                    'episode_normal': normal.strftime('%Y-%m-%d'), 'is_control': asset == CONTROL,
                    'issue_selection': ISSUE_SELECTION, 'truth_window_policy': TRUTH_WINDOW_POLICY,
                    'outcome_basis': 'ASSET_LEVEL_MIN', 'n_not_evaluated_months': 0}
            chosen, n_ne, n_months, m = None, 0, 0, first_issue_month
            while m <= last_issue_month:
                n_months += 1
                sig = issue_month_signal(by_key, peers, c, m, asset, lv[asset], weeks, win_end)
                if sig['status'] != 'EVALUATED':
                    n_ne += 1
                elif sig['issued_zone']:
                    chosen = (m, sig)
                    break
                m = _month_add(m, 1)
            base['n_not_evaluated_months'] = n_ne
            if chosen is None:
                rows.append({**base, 'issue_month': None, 'issue_date': None, 'zone_low': None, 'zone_high': None,
                             'width_days': None, 'trough_week': None, 'trough_month': None,
                             # GLOBAL NO-U: NO_ZONE (evaluated, vote < 2) vs NOT_EVALUATED (no evaluable month)
                             'status': 'NOT_EVALUATED' if n_ne >= n_months else 'NO_ZONE',
                             'votes': None, 'f1': None, 'f2': None, 'f3': None, 'interval_score': None,
                             'lead_days': None})
                continue
            m, sig = chosen
            issue = _month_end(m)
            lo, hi = pd.Timestamp(sig['lower']), pd.Timestamp(sig['upper'])
            truth_levels = lv[asset][(lv.index > issue) & (lv.index <= win_end)]
            mature = (win_end <= last_week) and len(truth_levels) > 0
            rec = {**base, 'issue_month': m, 'issue_date': issue.strftime('%Y-%m-%d'),
                   'zone_low': lo.strftime('%Y-%m-%d'), 'zone_high': hi.strftime('%Y-%m-%d'),
                   'width_days': int((hi - lo).days), 'votes': sig['votes'], 'f1': sig['f1'], 'f2': sig['f2'],
                   'f3': sig['f3']}
            if not mature:
                rec.update(trough_week=None, trough_month=None, status='IMMATURE', interval_score=None, lead_days=None)
            else:
                tr = BF.realized_trough([(d.strftime('%Y-%m-%d'), float(v)) for d, v in truth_levels.items()])
                tw = pd.Timestamp(tr['trough_week'])
                rec.update(trough_week=tr['trough_week'], trough_month=tw.strftime('%Y-%m'),
                           status='HIT' if lo <= tw <= hi else 'MISS',
                           interval_score=BF.interval_score_weeks((lo - issue).days / 7, (hi - issue).days / 7,
                                                                  (tw - issue).days / 7, alpha),
                           lead_days=int((tw - issue).days))
            rows.append(rec)
    return pd.DataFrame(rows)


def realised_troughs(episodes, weekly_returns: pd.DataFrame, *, window_tail_months: int = 6) -> pd.DataFrame:
    """Episode-wide lowest INDEX LEVEL per asset (descriptive truth table; never a feature)."""
    lv = asset_level_frame(weekly_returns)
    out = []
    for e in episodes:
        if e['country'] not in BF.EA7:
            continue
        start, normal = pd.Timestamp(e['start']), pd.Timestamp(e['normal'])
        end = min(normal + pd.DateOffset(months=window_tail_months), DEV_END)
        w = lv[(lv.index >= start) & (lv.index <= end)]
        if w.empty:
            continue
        for asset in (*BF.ASSETS, CONTROL):
            tr = BF.realized_trough([(d.strftime('%Y-%m-%d'), float(v)) for d, v in w[asset].items()])
            out.append({'country': e['country'], 'asset': asset, 'episode_start': start.strftime('%Y-%m-%d'),
                        'episode_normal': normal.strftime('%Y-%m-%d'), 'trough_week': tr['trough_week'],
                        'trough_month': tr['trough_week'][:7], 'trough_level': tr['trough_level']})
    return pd.DataFrame(out)


def summarize_bottom(zones: pd.DataFrame) -> dict:
    """HitRate and ForecastCoverage are different ratios; the EA5 control is reported separately."""
    def block(z: pd.DataFrame) -> dict:
        if z.empty:
            return {'n_records': 0}
        c = z['status'].value_counts().to_dict()
        hit, miss, ab, im = (int(c.get(k, 0)) for k in ('HIT', 'MISS', 'NO_ZONE', 'IMMATURE'))
        ne = int(c.get('NOT_EVALUATED', 0))
        issued = hit + miss
        elig = issued + ab
        scored = z[z['status'].isin(['HIT', 'MISS'])]
        return {'n_records': int(len(z)), 'n_hit': hit, 'n_miss': miss, 'n_no_zone': ab,
                'n_not_evaluated': ne, 'n_immature': im,
                'n_eligible_matured': elig, 'n_issued_matured': issued,
                'hit_rate': hit / issued if issued else None,
                'forecast_coverage': issued / elig if elig else None,
                'abstention_rate': ab / elig if elig else None,
                'median_width_days': float(z['width_days'].dropna().median()) if z['width_days'].notna().any() else None,
                'mean_interval_score_weeks': float(scored['interval_score'].mean()) if len(scored) else None,
                'n_country_episodes': int(z[['country', 'episode_start']].drop_duplicates().shape[0])}
    if zones.empty:
        return {'status': 'NO_ZONE', **block(zones)}
    five = zones[~zones['is_control']]
    return {'status': 'THREE_FAMILY_ZONES_EX_POST_ISSUE_SELECTION', 'five_assets': block(five),
            'control_EA5': block(zones[zones['is_control']]), **block(five),
            'families': ['F1_EWI_COOLING', 'F2_PEER_RECOVERY', 'F3_ASSET_RETURN_VOL_COOLING'],
            'issue_selection': ISSUE_SELECTION, 'truth_window_policy': TRUTH_WINDOW_POLICY}
