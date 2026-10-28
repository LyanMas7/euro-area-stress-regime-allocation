"""RQ03-12, RQ03-25, D03, D06: scenario VaR/ES, regime mixtures and horizon risk.

Loss L = -w'R on discrete scenarios with mass p (uniform if not given).
  VaR_alpha = inf{l : P(L <= l) >= alpha}  (lower alpha-quantile of the weighted distribution)
  ES_alpha  = VaR + E[(L - VaR)^+] / (1 - alpha)   (Rockafellar-Uryasev / Acerbi-Tasche):
              the mean of exactly (1-alpha) tail mass, splitting the atom at VaR.
Mixture: scenarios of regime s carry mass p_s / n_s; ES(mixture) generally differs
from sum_s p_s ES_s (D03), so the weighted-ES shortcut is refused as a substitute.
Horizon H: cumulative path return prod(1 + w'R_h) - 1 from JOINT path scenarios;
sqrt(H) scaling is reported only as a diagnostic, never as the risk value (D06).
Estimates are not owner risk limits.
"""
from __future__ import annotations

import numpy as np

from .contract import ALPHAS, REGIMES, covariance, result, simplex


def var_es(losses, alpha: float, mass=None, *, min_tail_effective: float | None = None) -> dict:
    x = np.asarray(losses, dtype=float)
    if x.ndim != 1 or len(x) == 0 or not np.isfinite(x).all() or not 0 < alpha < 1:
        raise ValueError('INVALID_LOSS_OR_ALPHA')
    p = np.full(len(x), 1 / len(x)) if mass is None else np.asarray(mass, dtype=float)
    if p.shape != x.shape or not np.isfinite(p).all() or p.min() < 0 or abs(p.sum() - 1) > 1e-10:
        raise ValueError('INVALID_SCENARIO_MASS')
    keep = p > 0
    x, p = x[keep], p[keep]
    o = np.argsort(x, kind='stable')
    x, p = x[o], p[o]
    cdf = np.cumsum(p)
    k = int(np.searchsorted(cdf, alpha - 1e-15, side='left'))
    var = float(x[min(k, len(x) - 1)])
    es = var + float(np.dot(p, np.maximum(x - var, 0))) / (1 - alpha)
    above = float(p[x > var].sum())
    eff = 1 / float(np.sum((p / p.sum()) ** 2)) * (1 - alpha)
    out = result('PASS', VaR=var, ES=es, alpha=alpha, n_scenarios=int(len(x)), tail_mass=1 - alpha,
                 mass_strictly_above_VaR=above, atom_share_at_VaR=max(0.0, 1 - alpha - above),
                 effective_tail_support=eff, convention='LOWER_QUANTILE_VaR_RU_ES_ATOM_SPLIT')
    if min_tail_effective is not None and eff < min_tail_effective:
        return {**out, 'status': 'U', 'reason': 'TAIL_SUPPORT_BELOW_OWNER_MINIMUM'}
    return out


def portfolio_tail(scenarios, weights, *, alphas=ALPHAS, mass=None) -> dict:
    """GROSS weekly tail of the normalised weights, L = -w'R: an ESTIMATE, not a cap test.

    Owner caps are tested on the NET loss on OPENING NAV, L = C - X'R with X = (1-C) w+ (see
    portfolio_tail_net / risk_tolerance); this function is labelled so it cannot be mistaken for it.
    """
    r = np.asarray(scenarios, dtype=float)
    w = simplex(weights, r.shape[1])
    loss = -(r @ w)
    return {a: {**var_es(loss, a, mass), 'basis': 'GROSS_WEEKLY_NOT_FOR_CAP_TESTS'} for a in alphas}


def portfolio_tail_net(scenarios, post_weights, cost_fraction: float, *, alphas=ALPHAS, mass=None) -> dict:
    """NET-after-cost tail on opening NAV: L_j = C - X'r_j with X = (1 - C) w+  <=>  1 - (1-C)(1 + w+'r_j)."""
    r = np.asarray(scenarios, dtype=float)
    w = simplex(post_weights, r.shape[1])
    c = float(cost_fraction)
    if not (np.isfinite(c) and 0 <= c < 1):
        raise ValueError('COST_FRACTION_DOMAIN')
    loss = c - (1 - c) * (r @ w)
    return {a: {**var_es(loss, a, mass), 'basis': 'NET_AFTER_COST'} for a in alphas}


def regime_benchmark_table(scen_by_regime: dict, weights_by_benchmark: dict, *, alphas=ALPHAS) -> dict:
    """VaR/ES for every benchmark x regime x alpha; missing regime scenarios -> U cell."""
    rows = []
    for b, w in weights_by_benchmark.items():
        for s in REGIMES:
            sc = scen_by_regime.get(s)
            if sc is None or len(sc) == 0:
                rows.append({'benchmark': b, 'regime': s, 'status': 'U', 'reason': 'REGIME_SCENARIOS_MISSING'})
                continue
            for a, t in portfolio_tail(sc, w, alphas=alphas).items():
                rows.append({'benchmark': b, 'regime': s, 'alpha': a, 'VaR': t['VaR'], 'ES': t['ES'],
                             'n': t['n_scenarios'], 'status': 'PASS'})
    n_u = sum(r['status'] != 'PASS' for r in rows)
    return result('PASS' if n_u == 0 else 'U', None if n_u == 0 else 'SOME_REGIME_CELLS_U', rows=rows,
                  claim='RISK_ESTIMATES_NOT_OWNER_LIMITS')


def mixture(scen_by_regime: dict, probs: dict):
    if set(probs) != set(REGIMES) or any(v < 0 for v in probs.values()) or abs(sum(probs.values()) - 1) > 1e-10:
        raise ValueError('FOUR_REGIME_PROBABILITY_SIMPLEX_REQUIRED')
    blocks, mass = [], []
    for s in REGIMES:
        if probs[s] == 0:
            continue
        sc = np.asarray(scen_by_regime[s], dtype=float)
        blocks.append(sc)
        mass.append(np.full(len(sc), probs[s] / len(sc)))
    return np.vstack(blocks), np.concatenate(mass)


def mixture_vs_weighted_es(scen_by_regime: dict, probs: dict, weights, alpha: float) -> dict:
    sc, m = mixture(scen_by_regime, probs)
    w = simplex(weights, sc.shape[1])
    mix = var_es(-(sc @ w), alpha, m)
    weighted = sum(probs[s] * var_es(-(np.asarray(scen_by_regime[s]) @ w), alpha)['ES']
                   for s in REGIMES if probs[s] > 0)
    return result('PASS', mixture_ES=mix['ES'], mixture_VaR=mix['VaR'], weighted_conditional_ES=weighted,
                  difference=mix['ES'] - weighted, substitute_allowed=False,
                  object='ES_OF_MIXTURE_DISTRIBUTION (actual target)')


def mixture_covariance(means: dict, covs: dict, probs: dict) -> dict:
    """Law of total covariance: sum p_s (S_s + m_s m_s') - m m'; PSD verified."""
    mu = sum(probs[s] * np.asarray(means[s], float) for s in REGIMES)
    second = sum(probs[s] * (covariance(covs[s]) + np.outer(means[s], means[s])) for s in REGIMES)
    total = covariance(second - np.outer(mu, mu))
    return result('PASS', mean=mu.tolist(), covariance=total.tolist(),
                  min_eigenvalue=float(np.linalg.eigvalsh(total).min()))


def horizon_path_risk(paths, weights, alpha: float, *, mass=None) -> dict:
    """paths: S x H x n joint path scenarios; buy-and-hold w over H weeks (no rebalance)."""
    p = np.asarray(paths, dtype=float)
    if p.ndim != 3:
        raise ValueError('S_BY_H_BY_N_PATHS_REQUIRED')
    w = simplex(weights, p.shape[2])
    growth = np.prod(1 + p, axis=1) @ w
    multi = var_es(1 - growth, alpha, mass)
    week1 = var_es(-(p[:, 0, :] @ w), alpha, mass)
    h = p.shape[1]
    return result('PASS', horizon_weeks=h, horizon_VaR=multi['VaR'], horizon_ES=multi['ES'], weekly_VaR=week1['VaR'],
                  weekly_ES=week1['ES'], sqrt_time_VaR_diagnostic=week1['VaR'] * np.sqrt(h),
                  sqrt_time_used_as_risk=False, effective_tail_support=multi['effective_tail_support'])


def overlapping_paths(returns, horizon: int) -> np.ndarray:
    """Historical joint H-week paths (overlapping windows): tail support is NOT S independent draws."""
    r = np.asarray(returns, dtype=float)
    if horizon < 1 or len(r) < horizon:
        raise ValueError('HORIZON_DOMAIN')
    return np.stack([r[i:i + horizon] for i in range(len(r) - horizon + 1)])
