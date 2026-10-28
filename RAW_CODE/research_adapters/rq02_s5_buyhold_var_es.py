"""RQ02 ENTRY S5 only: five Buy-and-Hold equity portfolios, VaR/ES peak, RoundBack, last Short confirmation.

Spec IJP-S5 section F / appendix A.2.  S5 belongs to RQ02 Pre-Onset ONLY: it is NOT used by RQ01 and
NOT by RQ02 EXIT (EXIT has S1..S4, no portfolio, no VaR/ES, no RoundBack).  Everything after H6 is a
RETROSPECTIVE research quantity; none of it may enter the forecast issued at H6.

  portfolios  EW | IVOL | MVP | ERC | MaxDiv   long-only, optimised ONCE from the covariance known at H6
  buy & hold  q = V0 w0 / P(H6), NAV_t = sum q P_t, weights drift, turnover after H6 is exactly 0
  risk        VaR and ES at alpha in {.95,.975,.99}  ->  5 x 2 x 3 = 30 candidate series per method
  peak        t*_g = first argmax of series g in (H6, T]; g* = lexicographic min (t*_g, -Risk_g(t*_g), keys)
  RoundBack   end of previous month if the peak is a month end, else end of the month two back
  endpoints   peak_at  !=  risk_cutoff_at = RoundBack(peak_at)  !=  last_short_confirm_at
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

ALPHAS = (0.95, 0.975, 0.99)
PORTFOLIOS = ('EW', 'IVOL', 'MVP', 'ERC', 'MaxDiv')
MEASURES = ('VaR', 'ES')
METHODS = ('EWMA_NORMAL', 'HISTORICAL_52W')


def initial_weights(cov) -> dict:
    """Five long-only three-asset portfolios from the covariance known at H6 (sum 1, w >= 0)."""
    cov = np.asarray(cov, dtype=float)
    if cov.shape != (3, 3) or not np.isfinite(cov).all() or np.abs(cov - cov.T).max() > 1e-12 * max(1.0, np.abs(cov).max()):
        raise ValueError('3x3 finite symmetric prior covariance needed')
    vol = np.sqrt(np.maximum(np.diag(cov), 1e-15))

    def solve(fun):
        res = minimize(fun, [1 / 3] * 3, method='SLSQP', bounds=[(0., 1.)] * 3,
                       constraints=[{'type': 'eq', 'fun': lambda w: sum(w) - 1}],
                       options={'maxiter': 1500, 'ftol': 1e-12})
        if not res.success:
            raise ValueError('H6 portfolio optimization not converged')   # never fall back to a made-up weight
        w = np.maximum(res.x, 0)
        return w / w.sum()

    def erc(w):
        normalized = cov / max(float(np.mean(np.diag(cov))), 1e-12)
        rc = w * (normalized @ w)
        return float(((rc - rc.mean()) ** 2).sum())

    def neg_div(w):
        variance = max(float(w @ cov @ w), 1e-18)
        return -float(w @ vol) / math.sqrt(variance)

    inv = 1 / vol
    return {'EW': np.full(3, 1 / 3), 'IVOL': inv / inv.sum(), 'MVP': solve(lambda w: float(w @ cov @ w)),
            'ERC': solve(erc), 'MaxDiv': solve(neg_div)}


def buy_hold(prices: pd.DataFrame, anchor, w0, capital: float = 1.0):
    """NAV, drifting weights and CONSTANT shares.  There is no rebalancing after the anchor."""
    a = pd.Timestamp(anchor)
    if a not in prices.index:
        raise ValueError('need actual weekly pricing anchor')
    w0 = np.asarray(w0, float)
    if not np.isclose(w0.sum(), 1) or np.min(w0) < 0:
        raise ValueError('invalid H6 weights')
    base = prices.loc[a].to_numpy(float)
    if np.any(base <= 0):
        raise ValueError('nonpositive equity index level')
    shares = capital * w0 / base
    p = prices.loc[prices.index >= a]
    vals = p.to_numpy(float) * shares
    nav = pd.Series(vals.sum(axis=1), index=p.index, name='NAV')
    drift = pd.DataFrame(vals / nav.to_numpy()[:, None], index=p.index, columns=p.columns)
    if not (np.allclose(drift.sum(axis=1), 1.) and np.allclose(drift.iloc[0].to_numpy(), w0)):
        raise AssertionError('BUY_AND_HOLD_WEIGHT_INVARIANT')
    return nav, drift, shares


def turnover_after_anchor(shares_by_date: pd.DataFrame) -> float:
    """Traded units after the anchor; Buy-and-Hold requires exactly 0."""
    return float(np.abs(np.diff(shares_by_date.to_numpy(float), axis=0)).sum()) if len(shares_by_date) > 1 else 0.0


def risk_table_ewma_normal(asset_returns: pd.DataFrame, drift: pd.DataFrame, cov_h6, lam: float = 0.94) -> pd.DataFrame:
    """One-sided EWMA covariance; 1-week zero-mean Gaussian VaR/ES of the drifting NAV."""
    if not 0 < lam < 1:
        raise ValueError('lambda must be in (0,1)')
    cov = np.asarray(cov_h6, float).copy()
    rows = []
    for t, w in drift.iterrows():
        if t not in asset_returns.index:
            continue
        rt = asset_returns.loc[t].to_numpy(float)
        cov = lam * cov + (1 - lam) * np.outer(rt, rt)
        sigma = math.sqrt(max(float(w.to_numpy() @ cov @ w.to_numpy()), 0.))
        for alpha in ALPHAS:
            z = norm.ppf(alpha)
            rows.append(dict(date=t, measure='VaR', alpha=alpha, risk=z * sigma))
            rows.append(dict(date=t, measure='ES', alpha=alpha, risk=norm.pdf(z) / (1 - alpha) * sigma))
    return pd.DataFrame(rows)


def risk_table_historical(nav_returns: pd.Series, post_h6_dates, lookback: int = 52) -> pd.DataFrame:
    """Trailing historical weekly losses; the tail sample size is logged (0.99 over 52w = 1 obs)."""
    records = []
    for t in post_h6_dates:
        x = (-nav_returns.loc[nav_returns.index <= t]).tail(lookback).to_numpy(float)
        if len(x) < lookback:
            raise ValueError('not enough prior completed weeks')
        for alpha in ALPHAS:
            var = float(np.quantile(x, alpha, method='higher'))
            tail = x[x >= var]
            records += [dict(date=t, measure='VaR', alpha=alpha, risk=var, tail_n=len(tail)),
                        dict(date=t, measure='ES', alpha=alpha, risk=float(tail.mean()), tail_n=len(tail))]
    return pd.DataFrame(records)


def earliest_risk_peak(all_risk: pd.DataFrame, *, h6=None, onset=None):
    """One max per portfolio/measure/alpha, then the EARLIEST peak date; loss only breaks date ties.

    Window is (H6, T] when h6/onset are given.  Choosing the global loss maximum first is the
    wrong order (the spec wants the peak farthest from Onset), so it is never done here.
    """
    needed = {'portfolio', 'date', 'measure', 'alpha', 'risk'}
    if not needed.issubset(all_risk.columns) or all_risk.empty:
        raise ValueError('need 5x2x3 risk series with real dates')
    frame = all_risk
    if h6 is not None:
        frame = frame[frame['date'] > pd.Timestamp(h6)]
    if onset is not None:
        frame = frame[frame['date'] <= pd.Timestamp(onset)]
    if frame.empty:
        raise ValueError('no risk observation inside (H6, T]')
    candidates = []
    for (p, m, a), g in frame.groupby(['portfolio', 'measure', 'alpha']):
        g = g.sort_values('date')
        vmax = g.risk.max()
        row = g.loc[g.risk.eq(vmax)].iloc[0]                       # earliest date among equal maxima
        candidates.append(dict(portfolio=p, measure=m, alpha=a, peak_at=pd.Timestamp(row['date']),
                               peak_loss=float(vmax)))
    candidates.sort(key=lambda r: (r['peak_at'], -r['peak_loss'], r['portfolio'], r['measure'], r['alpha']))
    return candidates[0], candidates


def round_back_s5(peak_date) -> pd.Timestamp:
    """EOM(previous month) if the peak is a month end, else EOM(two months back).  Leap-year safe."""
    p = pd.Timestamp(peak_date)
    back = 1 if p.is_month_end else 2
    return (p.to_period('M') - back).to_timestamp(how='end').normalize()


def last_short_confirmation(short_votes: pd.DataFrame, h6, cutoff):
    """max issue in (H6, cutoff] whose complete 7-EWI + 21-pair Short decision is 1; None -> NO_CONFIRMED_HIT.

    A positive is NEVER imputed at the risk cutoff, and a missing/NOT_EVALUATED issue is not a 0.
    """
    eligible = short_votes[(short_votes['issue'] > pd.Timestamp(h6)) & (short_votes['issue'] <= pd.Timestamp(cutoff))]
    if not eligible.empty and not eligible['vote'].isin([0, 1]).all():
        raise ValueError('binary eligible votes only')
    hits = eligible.loc[eligible['vote'].eq(1), 'issue']
    return None if hits.empty else pd.Timestamp(hits.max())


def s5_endpoints(all_risk: pd.DataFrame, short_votes: pd.DataFrame, *, h6, onset) -> dict:
    """The three DISTINCT S5 fields.  Retrospective only; the issued H6 forecast is never touched."""
    if all_risk.groupby(['portfolio', 'measure', 'alpha']).ngroups != len(PORTFOLIOS) * len(MEASURES) * len(ALPHAS):
        raise ValueError('EXACTLY_30_CANDIDATE_SERIES_REQUIRED')
    best, cands = earliest_risk_peak(all_risk, h6=h6, onset=onset)
    cutoff = round_back_s5(best['peak_at'])
    if cutoff <= pd.Timestamp(h6):
        status, last = 'RISK_CUTOFF_NOT_AFTER_H6', None
    else:
        last = last_short_confirmation(short_votes, h6, cutoff)
        status = 'CONFIRMED' if last is not None else 'NO_CONFIRMED_HIT'
    return {'status': status, 'peak_at': best['peak_at'], 'risk_cutoff_at': cutoff, 'last_short_confirm_at': last,
            'peak_portfolio': best['portfolio'], 'peak_measure': best['measure'], 'peak_alpha': best['alpha'],
            'peak_loss': best['peak_loss'], 'n_candidates': len(cands), 'role': 'RETROSPECTIVE_NOT_A_H6_FEATURE'}


def s5_retrospective_evaluation(events, *, issued_h6_sha_before: str | None = None,
                                issued_h6_sha_after: str | None = None) -> dict:
    """Descriptive S5 outcome table over Onset events; it can never edit an issued H6 forecast.

    events: dicts with country, onset, h6 and the `s5_endpoints` fields.  Lead times are calendar days
    from the endpoint to the Onset.  The three equity series are common Euro indices, so the events are
    NOT independent assets and `independent_asset_claim` is always False.
    """
    if (issued_h6_sha_before is None) != (issued_h6_sha_after is None):
        raise ValueError('BOTH_ISSUED_H6_DIGESTS_OR_NONE')
    if issued_h6_sha_before is not None and issued_h6_sha_before != issued_h6_sha_after:
        raise ValueError('ISSUED_H6_FORECAST_WAS_MODIFIED_BY_RETROSPECTIVE_S5')
    rows, lead_last, lead_cut = [], [], []
    for e in events:
        status = e['status']
        onset = pd.Timestamp(e['onset'])
        row = {'country': e['country'], 'status': status, 'peak_at': e['peak_at'],
               'risk_cutoff_at': e['risk_cutoff_at'], 'last_short_confirm_at': e.get('last_short_confirm_at')}
        row['cutoff_lead_days'] = int((onset - pd.Timestamp(e['risk_cutoff_at'])).days)
        lead_cut.append(row['cutoff_lead_days'])
        if status == 'CONFIRMED':
            row['last_short_lead_days'] = int((onset - pd.Timestamp(e['last_short_confirm_at'])).days)
            lead_last.append(row['last_short_lead_days'])
        rows.append(row)
    n = len(rows)
    return {'rows': rows, 'n_events': n, 'n_confirmed': sum(r['status'] == 'CONFIRMED' for r in rows),
            'n_no_confirmed_hit': sum(r['status'] == 'NO_CONFIRMED_HIT' for r in rows),
            'n_cutoff_not_after_h6': sum(r['status'] == 'RISK_CUTOFF_NOT_AFTER_H6' for r in rows),
            'median_last_short_lead_days': float(np.median(lead_last)) if lead_last else None,
            'median_cutoff_lead_days': float(np.median(lead_cut)) if lead_cut else None,
            'n_distinct_countries': len({r['country'] for r in rows}),
            'independent_asset_claim': False, 'shared_equity_tape': True,
            'issued_h6_forecast': 'UNCHANGED' if issued_h6_sha_before is not None else 'NOT_CHECKED',
            'role': 'RETROSPECTIVE_DESCRIPTIVE_NOT_A_FORECAST_EVALUATION'}