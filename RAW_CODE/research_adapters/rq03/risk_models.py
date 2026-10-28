"""RQ03-11 / C04: GARCH(1,1) + DCC(1,1) fitted engine, as-of projection, other risk signals.

GARCH:  h_t = omega + alpha e_{t-1}^2 + beta h_{t-1}, omega > 0, alpha, beta >= 0, alpha+beta < 1
        Gaussian QML  NLL = 1/2 sum_t (ln h_t + e_t^2 / h_t)   (constants dropped)
        reparametrised: alpha+beta = logistic(p), alpha = (alpha+beta) logistic(q), omega = exp(w)
DCC:    Q_t = (1-a-b) Qbar + a z_{t-1} z_{t-1}' + b Q_{t-1}, Q_1 = Qbar = corr(z), a, b >= 0, a+b < 1
        R_t = diag(Q_t)^-1/2 Q_t diag(Q_t)^-1/2,  NLL_2 = 1/2 sum (ln|R_t| + z'R^-1 z - z'z)
Sigma_t = D_t R_t D_t, D_t = diag(sqrt h_t); every Sigma_t is checked PSD (no repair).
Parameters are estimated once on the whole DEV block (File01: no folds/refits); the
FILTER is prefix-causal holding parameters fixed, but the estimated parameters
contain later DEV observations for earlier DEV dates. Therefore in-sample paths
must NOT be described as authentic issued/PIT forecasts or independent validation.
C04 roles: GARCH = volatility level, DCC = covariance contagion, EWMA = smoother vol
challenger, RV = realised check, FHS = tail veto scenarios. They share the same return
tape and are NOT independent votes.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .contract import covariance, result

ROLES = {'GARCH': 'VOLATILITY_LEVEL', 'DCC': 'COVARIANCE_CONTAGION', 'EWMA': 'VOLATILITY_SMOOTHED_CHALLENGER',
         'RV': 'REALISED_VOLATILITY_CHECK', 'FHS': 'TAIL_VETO_SCENARIOS'}


def _logistic(x):
    return 1.0 / (1.0 + np.exp(-x))


def garch_filter(e, omega: float, alpha: float, beta: float, h0: float) -> np.ndarray:
    """h[t] = Var(e_t | e_<t) for t < T and h[T] = one-step-ahead forecast."""
    e = np.asarray(e, dtype=float)
    if not np.isfinite(e).all() or omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1 or h0 <= 0:
        raise ValueError('GARCH_DOMAIN')
    h = np.empty(len(e) + 1)
    h[0] = h0
    for t in range(len(e)):
        h[t + 1] = omega + alpha * e[t] ** 2 + beta * h[t]
    return h


def _unpack(theta):
    pers = min(float(_logistic(theta[1])), 1 - 1e-9)  # saturation guard keeps alpha+beta < 1
    a = pers * float(_logistic(theta[2]))
    return float(np.exp(np.clip(theta[0], -700, 700))), a, pers - a


def fit_garch11(x, *, starts=((-3.0, 2.0, -2.0), (-1.0, 3.0, -1.5), (-2.0, 1.0, -0.5))) -> dict:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or len(x) < 30 or not np.isfinite(x).all():
        raise ValueError('GARCH_SERIES_LEN30_FINITE_REQUIRED')
    mu, sd = float(x.mean()), float(x.std(ddof=1))
    if sd <= 0:
        raise ValueError('ZERO_VARIANCE_SERIES')
    e = (x - mu) / sd

    def nll(theta):
        w, a, b = _unpack(theta)
        if not w > 0:
            return 1e20
        h = garch_filter(e, w, a, b, 1.0)[:-1]
        return float(0.5 * np.sum(np.log(h) + e ** 2 / h))

    fits = [minimize(nll, np.asarray(s0), method='Nelder-Mead',
                     options={'xatol': 1e-9, 'fatol': 1e-11, 'maxiter': 8000, 'maxfev': 16000}) for s0 in starts]
    best = min(fits, key=lambda f: f.fun)
    w, a, b = _unpack(best.x)
    return result('PASS', mean=mu, scale=sd, omega=w * sd * sd, alpha=a, beta=b, persistence=a + b,
                  stationarity_boundary=bool(a + b > 1 - 1e-6),
                  nll_scaled=float(best.fun), constant_variance_nll=float(0.5 * np.sum(e ** 2)),
                  converged=bool(best.success), starts=len(starts), h0=sd * sd,
                  estimation_scope='FULL_DEV_ONE_BLOCK_PARAMETERS_CAUSAL_FILTER')


def garch_path(x, fit: dict) -> np.ndarray:
    return garch_filter(np.asarray(x, float) - fit['mean'], fit['omega'], fit['alpha'], fit['beta'], fit['h0'])


def asof_variance(x_upto_u, fit: dict) -> float:
    """As-of one-step forecast h_{u+1|u}: depends on returns <= u only."""
    return float(garch_path(x_upto_u, fit)[-1])


def dcc_filter(z, qbar, a: float, b: float) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    q = covariance(qbar, z.shape[1], positive=True).copy()
    if not np.isfinite(z).all() or a < 0 or b < 0 or a + b >= 1:
        raise ValueError('DCC_DOMAIN')
    out = np.empty((len(z) + 1, z.shape[1], z.shape[1]))
    for t in range(len(z) + 1):
        d = np.sqrt(np.diag(q))
        out[t] = q / np.outer(d, d)
        if t < len(z):
            q = (1 - a - b) * qbar + a * np.outer(z[t], z[t]) + b * q
    return out


def fit_dcc(z, *, starts=((2.0, -3.0), (3.0, -2.0), (0.5, -1.0))) -> dict:
    z = np.asarray(z, dtype=float)
    qbar = np.corrcoef(z, rowvar=False)

    def nll(theta):
        pers = min(float(_logistic(theta[0])), 1 - 1e-9)
        a = pers * float(_logistic(theta[1]))
        r = dcc_filter(z, qbar, a, pers - a)[:-1]
        sign, logdet = np.linalg.slogdet(r)
        if np.any(sign <= 0):
            return 1e20
        sol = np.linalg.solve(r, z[..., None])[..., 0]
        return float(0.5 * np.sum(logdet + np.einsum('ij,ij->i', z, sol) - np.einsum('ij,ij->i', z, z)))

    fits = [minimize(nll, np.asarray(s0), method='Nelder-Mead', options={'xatol': 1e-8, 'fatol': 1e-10,
                                                                         'maxiter': 4000}) for s0 in starts]
    best = min(fits, key=lambda f: f.fun)
    pers = min(float(_logistic(best.x[0])), 1 - 1e-9)
    a = float(pers * _logistic(best.x[1]))
    return result('PASS', a=a, b=pers - a, qbar=qbar.tolist(), nll=float(best.fun), constant_corr_nll=nll((-50.0, 0.0)),
                  converged=bool(best.success))


def fit_garch_dcc(returns) -> dict:
    """returns: T x k risky returns (DEV only). Sigma_t for t = 1..T+1 (last = as-of forecast)."""
    x = np.asarray(returns, dtype=float)
    if x.ndim != 2 or x.shape[1] < 2 or not np.isfinite(x).all():
        raise ValueError('T_BY_K_FINITE_RETURNS_REQUIRED')
    fits = [fit_garch11(x[:, i]) for i in range(x.shape[1])]
    h = np.column_stack([garch_path(x[:, i], f) for i, f in enumerate(fits)])
    z = (x - np.array([f['mean'] for f in fits])) / np.sqrt(h[:-1])
    dcc = fit_dcc(z)
    r = dcc_filter(z, np.asarray(dcc['qbar']), dcc['a'], dcc['b'])
    sd = np.sqrt(h)
    cov = r * sd[:, :, None] * sd[:, None, :]
    mins = np.array([np.linalg.eigvalsh(c).min() for c in cov])
    if mins.min() <= 0:
        return result('FAIL', 'NON_PD_CONDITIONAL_COVARIANCE', min_eigen=float(mins.min()))
    return result('PASS', marginals=fits, dcc={k: dcc[k] for k in ('a', 'b', 'nll', 'constant_corr_nll', 'converged')},
                  covariances=cov, correlations=r, variances=h, standardized=z, min_eigenvalue=float(mins.min()),
                  claim='FULL_DEV_IN_SAMPLE_PARAMETERS_PREFIX_CAUSAL_FILTER_NOT_ISSUED')


def ewma_variance(x, lam: float, h0: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if not 0 < lam < 1 or h0 <= 0:
        raise ValueError('EWMA_DOMAIN')
    h = np.empty(len(x) + 1)
    h[0] = h0
    for t in range(len(x)):
        h[t + 1] = lam * h[t] + (1 - lam) * x[t] ** 2
    return h


def realized_vol(x, window: int) -> np.ndarray:
    """Backward rolling root-mean-square; value at t uses x[t-window+1..t]."""
    x = np.asarray(x, dtype=float)
    out = np.full(len(x), np.nan)
    for t in range(window - 1, len(x)):
        out[t] = float(np.sqrt(np.mean(x[t - window + 1:t + 1] ** 2)))
    return out


def fhs_scenarios(standardized, correlations, origin: int, target_sd, target_corr) -> np.ndarray:
    """Whiten historical z_s with their own R_s, recolour with the as-of D_u R_u (no double correlation)."""
    z = np.asarray(standardized, float)
    r = np.asarray(correlations, float)
    if not 1 <= origin <= len(z):
        raise ValueError('INVALID_FHS_ORIGIN')
    eps = np.linalg.solve(np.linalg.cholesky(r[:origin]), z[:origin, :, None])[:, :, 0]
    return eps @ (np.diag(target_sd) @ np.linalg.cholesky(covariance(target_corr, positive=True))).T


def risk_signal_roles(signals: list[dict]) -> dict:
    """signals: [{'name','kind','input_sha','vol_forecast'?}]; shared inputs collapse to one vote."""
    rows, groups = [], {}
    for s in signals:
        if s['kind'] not in ROLES:
            raise ValueError('UNREGISTERED_RISK_SIGNAL_KIND')
        rows.append({**s, 'role': ROLES[s['kind']]})
        groups.setdefault(s['input_sha'], []).append(s['name'])
    vols = [s['vol_forecast'] for s in signals if s.get('vol_forecast') is not None and s['kind'] in ('GARCH', 'EWMA', 'RV')]
    disagreement = (max(vols) - min(vols)) / float(np.median(vols)) if len(vols) >= 2 and np.median(vols) > 0 else None
    return result('PASS', signals=rows, n_signals=len(signals), n_independent_input_sets=len(groups),
                  shared_input_groups=groups, model_disagreement=disagreement,
                  independence='NOT_INDEPENDENT' if len(groups) < len(signals) else 'DISTINCT_INPUTS')
