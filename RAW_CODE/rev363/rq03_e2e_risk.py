"""RQ03-03 / RQ03-04 risk engine for the two-W0 run: one-sided GARCH-DCC, EWMA challenger, tail scenarios,
tail ESS, opening-NAV VaR/ES and exceedance backtests.

Estimation (DEV one block, File01 contract): GARCH(1,1) per risky asset and DCC(1,1) by Gaussian QML on the
2005-2019 DEV returns only (research_adapters.rq03.risk_models.fit_garch11 / fit_dcc, unchanged).  The
parameters therefore contain later DEV weeks for earlier DEV dates: DEV risk paths are RETROSPECTIVE
(in-sample parameters, prefix-causal filter), OOS paths are frozen-DEV-parameter one-sided forecasts.

Filter (repaired): h_t = Var(e_t | e_<t) starts from the WARM-UP variance of the first `warm` weeks, never
the full-sample variance (rev363.rq03.garch11_filter used np.var(x) of the whole series: a full-sample
initialisation); Sigma_t for week t uses returns strictly before t.  The OOS filter runs through the DEV
history first (as-of warm-up permitted by the master spec section J1) and never sees a later OOS week.

Loss on OPENING NAV (spec section 1): candidate post-trade weights w (6, sum 1), fee C, X = (1-C) w,
    L_j = C - X'r_j ;  Gaussian VaR/ES = C - mu'X + {z_a, phi(z_a)/(1-a)} sqrt(X'SX)
mu = (0,0,0,0,0, c1_t): the C1 accrual of week t is fixed by a fixing released before the week starts (as-of
known); the risky mean is ASSUMED zero (ASSUMED_RESEARCH_ONLY).  Student-t(5) is scaled to unit variance.
Tail ESS of scenario weights q over the tail: ESS_tail = (sum q)^2 / sum q^2 (Kish), via new_tail_rev357.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import chi2, norm, t as student_t

from research_adapters.rq03 import risk_models as RM
from research_adapters.rq03.new_tail_rev357 import tail_mass_decomposition

ALPHAS = (0.95, 0.975, 0.99)
WARM = 52                     # registered warm-up (rev363.rq03 dcc/ewma min_obs)
EWMA_LAMBDA = 0.94            # registered EWMA challenger (ASSUMED_RESEARCH_ONLY)
NU_T = 5.0                    # registered Student-t robustness (rev363.rq03.var_es)
HS_WINDOW = 104               # spec section 1 example window: 104 weeks -> ~5.2 tail scenario-weeks at 95%


# --------------------------------------------------------------------------- estimation and filtering
def fit_dev_garch_dcc(dev_risky: np.ndarray) -> dict:
    x = np.asarray(dev_risky, float)
    if x.ndim != 2 or x.shape[1] != 5 or not np.isfinite(x).all():
        raise ValueError("DEV_RISKY_RETURNS_T_BY_5_REQUIRED")
    marg = [RM.fit_garch11(x[:, i]) for i in range(5)]
    h = _garch_paths(x, marg, warm=WARM)
    z = (x - np.array([m["mean"] for m in marg])) / np.sqrt(h[:-1])
    dcc = RM.fit_dcc(z)
    return {"marginals": [{k: m[k] for k in ("mean", "omega", "alpha", "beta", "persistence", "stationarity_boundary",
                                             "converged", "nll_scaled", "constant_variance_nll")} for m in marg],
            "dcc": {k: dcc[k] for k in ("a", "b", "nll", "constant_corr_nll", "converged")},
            "qbar": dcc["qbar"], "n_dev_weeks": int(len(x)), "warm_up_weeks": WARM,
            "estimation_scope": "FULL_DEV_ONE_BLOCK_QML_PARAMETERS",
            "filter": "PREFIX_CAUSAL_WARMUP_VARIANCE_INIT"}


def _garch_paths(x: np.ndarray, marg: list, *, warm: int) -> np.ndarray:
    """(T+1) x 5 conditional variances; h[t] = Var(e_t | e_<t), h[0] = warm-up variance of e_0..e_{warm-1}."""
    cols = []
    for i, m in enumerate(marg):
        e = x[:, i] - m["mean"]
        h0 = float(np.mean(e[:warm] ** 2))
        cols.append(RM.garch_filter(e, m["omega"], m["alpha"], m["beta"], h0))
    return np.column_stack(cols)


def filter_dcc(all_risky: np.ndarray, fit: dict) -> dict:
    """Causal covariances Sigma[t] (week t, information < t) over the continuous 2005.. history."""
    x = np.asarray(all_risky, float)
    h = _garch_paths(x, fit["marginals"], warm=WARM)
    sd = np.sqrt(h)                                            # T+1
    z = (x - np.array([m["mean"] for m in fit["marginals"]])) / sd[:-1]
    r = RM.dcc_filter(z, np.asarray(fit["qbar"]), fit["dcc"]["a"], fit["dcc"]["b"])   # T+1
    cov = r * sd[:, :, None] * sd[:, None, :]
    ready = np.arange(len(x) + 1) >= WARM
    return {"cov": cov[:-1], "corr": r[:-1], "sd": sd[:-1], "z": z, "ready": ready[:-1],
            "cov_next": cov[-1], "min_eig": float(min(np.linalg.eigvalsh(c).min() for c in cov[WARM:]))}


def filter_ewma(all_risky: np.ndarray, *, lam: float = EWMA_LAMBDA) -> dict:
    x = np.asarray(all_risky, float)
    n = len(x)
    cov = np.full((n, 5, 5), np.nan)
    s = np.cov(x[:WARM].T, ddof=1)
    for t in range(n):
        if t >= WARM:
            cov[t] = s
            s = lam * s + (1 - lam) * np.outer(x[t], x[t])
    return {"cov": cov, "ready": np.arange(n) >= WARM}


# --------------------------------------------------------------------------- opening-NAV loss measures
def _quantiles(alpha: float, measure: str) -> tuple[float, float]:
    """(VaR multiplier, ES multiplier) of a unit-variance loss distribution."""
    if measure == "GAUSS":
        z = float(norm.ppf(alpha))
        return z, float(norm.pdf(z) / (1 - alpha))
    if measure == "T5":
        nu = NU_T
        sc = math.sqrt((nu - 2) / nu)
        q = float(student_t.ppf(alpha, nu))
        es = float(student_t.pdf(q, nu) * (nu + q * q) / ((nu - 1) * (1 - alpha)))
        return q * sc, es * sc
    raise ValueError("PARAMETRIC_MEASURE_GAUSS_OR_T5")


_QCACHE: dict = {}


def mult(alpha: float, measure: str) -> tuple[float, float]:
    k = (alpha, measure)
    if k not in _QCACHE:
        _QCACHE[k] = _quantiles(alpha, measure)
    return _QCACHE[k]


def parametric_var_es(w6, cost: float, sigma5, c1: float, alpha: float, measure: str) -> tuple[float, float, float]:
    """(VaR, ES, sd) of L = C - X'r with X = (1-C) w; risky mean 0, C1 mean = its known accrual c1."""
    w = np.asarray(w6, float)
    x = (1.0 - cost) * w
    sd = math.sqrt(max(float(x[:5] @ sigma5 @ x[:5]), 0.0))
    m = float(x[5] * c1)
    qv, qe = mult(alpha, measure)
    return cost - m + qv * sd, cost - m + qe * sd, sd


def scenario_losses(w6, cost: float, scen6: np.ndarray) -> np.ndarray:
    return cost - (1.0 - cost) * (np.asarray(scen6, float) @ np.asarray(w6, float))


def hist_es(losses: np.ndarray, alpha: float) -> tuple[float, float]:
    """Uniform-mass lower-quantile VaR and Rockafellar-Uryasev ES (atom split) - same convention as tail.var_es."""
    x = np.sort(np.asarray(losses, float))
    n = len(x)
    k = int(math.ceil(alpha * n - 1e-12)) - 1
    k = min(max(k, 0), n - 1)
    var = float(x[k])
    es = var + float(np.maximum(x - var, 0.0).sum()) / (n * (1 - alpha))
    return var, es


def fhs_scenarios(dcc: dict, t: int, c1: float, *, window: int | None = None) -> np.ndarray:
    """Filtered historical scenarios for week t: standardised residuals s < t whitened by their own R_s and
    recoloured by the as-of D_t R_t (risk_models.fhs_scenarios), C1 = its known accrual."""
    lo = 0 if window is None else max(0, t - window)
    z = dcc["z"][lo:t]
    r = dcc["corr"][lo:t]
    sc = RM.fhs_scenarios(z, r, len(z), dcc["sd"][t], dcc["corr"][t])
    return np.column_stack([sc, np.full(len(sc), c1)])


def fhs_whitened(dcc: dict) -> np.ndarray:
    """eps_s = chol(R_s)^-1 z_s for every week, via risk_models.fhs_scenarios recoloured by the identity."""
    return RM.fhs_scenarios(dcc["z"], dcc["corr"], len(dcc["z"]), np.ones(5), np.eye(5))


def fhs_scenarios_fast(eps: np.ndarray, dcc: dict, t: int, c1: float) -> np.ndarray:
    """Same scenarios as fhs_scenarios(dcc, t, c1): eps_{s<t} recoloured by D_t chol(R_t)."""
    lt = np.diag(dcc["sd"][t]) @ np.linalg.cholesky(dcc["corr"][t])
    sc = eps[:t] @ lt.T
    return np.column_stack([sc, np.full(len(sc), c1)])


def tail_ess(losses: np.ndarray, alpha: float, mass=None) -> dict:
    d = tail_mass_decomposition(np.asarray(losses, float), mass, alpha)
    return {"n_scenarios": int(len(losses)), "nominal_tail_scenarios": float(len(losses) * (1 - alpha)),
            "ESS_tail": float(d["n_eff_tail_actual"]), "n_positive_tail_atoms": int(d["n_positive_tail_atoms"]),
            "VaR": float(d["VaR"]), "ES": float(d["ES"]), "dependence": "SERIALLY_DEPENDENT_SCENARIO_WEEKS"}


# --------------------------------------------------------------------------- backtests and proper scores
def kupiec_pof(hits: np.ndarray, alpha: float) -> dict:
    h = np.asarray(hits, int)
    n, x = int(len(h)), int(h.sum())
    p = 1 - alpha
    if n == 0:
        return {"n": 0, "exceed": 0, "LR_pof": None, "p_pof": None}
    ph = x / n
    ll0 = (n - x) * math.log(1 - p) + x * math.log(p)
    ll1 = ((n - x) * math.log(1 - ph) if ph < 1 else 0.0) + (x * math.log(ph) if ph > 0 else 0.0)
    lr = -2 * (ll0 - ll1)
    return {"n": n, "exceed": x, "expected": n * p, "rate": ph, "LR_pof": lr, "p_pof": float(1 - chi2.cdf(lr, 1))}


def christoffersen(hits: np.ndarray) -> dict:
    h = np.asarray(hits, int)
    if len(h) < 2:
        return {"LR_ind": None, "p_ind": None}
    a, b = h[:-1], h[1:]
    n00, n01 = int(((a == 0) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    n10, n11 = int(((a == 1) & (b == 0)).sum()), int(((a == 1) & (b == 1)).sum())
    def ll(p, k0, k1):
        return (k0 * math.log(1 - p) if p < 1 and k0 else 0.0) + (k1 * math.log(p) if p > 0 and k1 else 0.0)
    p01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    p11 = n11 / (n10 + n11) if n10 + n11 else 0.0
    p1 = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    lr = -2 * (ll(p1, n00 + n10, n01 + n11) - ll(p01, n00, n01) - ll(p11, n10, n11))
    return {"n01": n01, "n11": n11, "LR_ind": lr, "p_ind": float(1 - chi2.cdf(lr, 1))}


def acerbi_z2(loss: np.ndarray, var: np.ndarray, es: np.ndarray, alpha: float, sd: np.ndarray, cost: np.ndarray,
              c1m: np.ndarray, measure: str, *, n_sim: int = 2000, seed: int = 363) -> dict:
    """Acerbi-Szekely Z2 = sum(L I)/(T(1-a) ES) - 1 (> 0: ES under-estimates); p-value by simulation under the
    issued model (each week's own location/scale and distribution)."""
    loss, var, es = (np.asarray(v, float) for v in (loss, var, es))
    n = len(loss)
    if n == 0:
        return {"Z2": None, "p_Z2": None}
    def z2(L):
        return float(np.sum(L * (L > var) / es) / (n * (1 - alpha)) - 1.0)
    obs = z2(loss)
    rng = np.random.default_rng(seed)
    loc = np.asarray(cost, float) - np.asarray(c1m, float)
    if measure == "T5":
        draws = student_t.rvs(NU_T, size=(n_sim, n), random_state=rng) * math.sqrt((NU_T - 2) / NU_T)
    else:
        draws = rng.standard_normal((n_sim, n))
    sims = np.array([z2(loc + np.asarray(sd) * d) for d in draws])
    return {"Z2": obs, "p_Z2": float(np.mean(sims >= obs))}


def pinball(loss: np.ndarray, var: np.ndarray, alpha: float) -> float:
    u = np.asarray(loss, float) - np.asarray(var, float)
    return float(np.mean(np.maximum(alpha * u, (alpha - 1) * u)))


def fz0(loss: np.ndarray, var: np.ndarray, es: np.ndarray, alpha: float) -> float | None:
    """Fissler-Ziegel FZ0 joint (VaR, ES) loss on returns y = -L (Patton-Ziegel-Chen 2019); needs ES > 0."""
    y, v, e = -np.asarray(loss, float), -np.asarray(var, float), -np.asarray(es, float)
    if np.any(e >= 0):
        return None
    a = 1 - alpha
    return float(np.mean(-((y <= v) * (v - y)) / (a * e) + v / e + np.log(-e) - 1.0))
