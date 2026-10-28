"""RQ03-06..10, RQ03-13, D04: five benchmark shapes on the risky simplex and their own tolerances.

Delta5 = {v >= 0, 1'v = 1}, Sigma PSD, sigma = sqrt(diag Sigma).
  EW risky sleeve  v_i = 1/5            (EW_6_ASSET w_i = 1/6 is a DIFFERENT benchmark)
  IVOL             v_i = (1/sigma_i) / sum_j (1/sigma_j), sigma from a NAMED estimator
  MVP              argmin_{Delta5} v'Sigma v            exact: active-set enumeration + KKT
  ERC              v_i (Sigma v)_i / v'Sigma v = 1/5     Newton on 1/2 y'Sy - 1/5 sum ln y_i
  MaxDiv           argmax_{Delta5} sigma'v / sqrt(v'Sigma v)
                   = normalise argmin{x'Sigma x : sigma'x = 1, x >= 0}  exact enumeration + KKT
Tolerance geometries (each its own unit, never a VaR/ES cap):
  EW, IVOL: L1 shape drift; MVP: variance regret; ERC: max |RC share - 1/5|; MaxDiv: DR regret.
"""
from __future__ import annotations
from itertools import combinations

import numpy as np

from .contract import METHODS, Blocked, covariance, owner_ok, result, simplex

SIGMA_ESTIMATORS = {
    'GARCH11_ASOF': 'FORECAST_CONDITIONAL', 'DCC_GARCH_ASOF': 'FORECAST_CONDITIONAL',
    'EWMA': 'FORECAST_SMOOTHED', 'REALIZED_ROLLING': 'REALIZED_BACKWARD', 'SAMPLE_DEV': 'REALIZED_FULL_DEV',
    'OPTION_IMPLIED': 'MARKET_IMPLIED',
}
GEOMETRY = {'EW': 'L1_RISKY_SHAPE_DRIFT', 'IVOL': 'L1_RISKY_SHAPE_DRIFT', 'MVP': 'VARIANCE_REGRET',
            'ERC': 'RC_SHARE_MAX_ABS_DEVIATION', 'MaxDiv': 'DIVERSIFICATION_RATIO_REGRET'}
# Units of each native tolerance.  A tau is only meaningful together with its unit: one shared number across
# these geometries (e.g. 0.05 for EW, MVP and ERC) is a unit mismatch, never a common risk tolerance (RT06).
GEOMETRY_UNITS = {'L1_RISKY_SHAPE_DRIFT': 'L1_OF_RISKY_SHAPE_SIMPLEX_5',
                  'VARIANCE_REGRET': 'WEEKLY_RETURN_VARIANCE',
                  'RC_SHARE_MAX_ABS_DEVIATION': 'RISK_CONTRIBUTION_SHARE_OF_PORTFOLIO_VARIANCE',
                  'DIVERSIFICATION_RATIO_REGRET': 'DIVERSIFICATION_RATIO_POINTS'}
KKT_TOL = 1e-9


def benchmark_registry() -> list[dict]:
    return [
        {'id': 'EW_RISKY_SLEEVE', 'n': 5, 'weights': '1/5 of risky sleeve; cash share a is a separate owner'},
        {'id': 'EW_6_ASSET', 'n': 6, 'weights': '1/6 of all six assets incl. cash; NOT the same benchmark'},
        {'id': 'IVOL', 'n': 5, 'weights': 'inverse named-estimator volatility'},
        {'id': 'MVP', 'n': 5, 'weights': 'long-only global minimum variance'},
        {'id': 'ERC', 'n': 5, 'weights': 'equal risk contribution'},
        {'id': 'MaxDiv', 'n': 5, 'weights': 'maximum diversification ratio'},
    ]


def ew_risky_sleeve(n: int = 5) -> np.ndarray:
    return np.full(n, 1.0 / n)


def ew_6_asset() -> np.ndarray:
    return np.full(6, 1.0 / 6)


def resolve_ew_definition(owner: dict | None) -> dict:
    """C04-EW conflict: owner must choose; no silent default."""
    if not owner_ok(owner) or owner.get('choice') not in ('EW_RISKY_SLEEVE', 'EW_6_ASSET'):
        return result('BLOCKED', 'EW_DEFINITION_OWNER_CHOICE_REQUIRED', candidates=['EW_RISKY_SLEEVE', 'EW_6_ASSET'])
    w = ew_6_asset() if owner['choice'] == 'EW_6_ASSET' else None
    return result('PASS', choice=owner['choice'], six_asset_weights=None if w is None else w.tolist(),
                  owner_hash=owner['owner_hash'])


def ivol_shape(sigma, *, estimator: str, source: dict | None = None) -> dict:
    if estimator not in SIGMA_ESTIMATORS:
        raise ValueError('UNREGISTERED_SIGMA_ESTIMATOR')
    if estimator == 'OPTION_IMPLIED' and not owner_ok(source):
        return result('BLOCKED', 'OPTION_IMPLIED_VOL_SOURCE_NOT_SUPPLIED', estimator=estimator)
    s = np.asarray(sigma, dtype=float)
    if s.ndim != 1 or not np.isfinite(s).all() or s.min() <= 0:
        raise ValueError('POSITIVE_FINITE_SIGMA_REQUIRED')
    v = (1 / s) / (1 / s).sum()
    kind = SIGMA_ESTIMATORS[estimator]
    # REALIZED_FULL_DEV swaps full-sample volatility into every historical issue: retrospective, not PIT-safe.
    return result('PASS', shape=v.tolist(), estimator=estimator, estimator_kind=kind,
                  pit_safe=kind != 'REALIZED_FULL_DEV')


def risk_contributions(v, sigma_m) -> np.ndarray:
    v = simplex(v)
    s = covariance(sigma_m, len(v))
    var = float(v @ s @ v)
    if var <= 0:
        raise ValueError('ZERO_PORTFOLIO_VARIANCE')
    return v * (s @ v) / var


def diversification_ratio(v, sigma_m) -> float:
    v = simplex(v)
    s = covariance(sigma_m, len(v))
    var = float(v @ s @ v)
    if var <= 0:
        raise ValueError('ZERO_PORTFOLIO_VARIANCE')
    return float(v @ np.sqrt(np.diag(s)) / np.sqrt(var))


def _enumerate(s: np.ndarray, b: np.ndarray):
    """Exact long-only min x'Sx s.t. b'x = 1, x >= 0 (b > 0) by support enumeration + KKT."""
    n = len(b)
    best = None
    for k in range(1, n + 1):
        for supp in combinations(range(n), k):
            idx = list(supp)
            try:
                y = np.linalg.solve(s[np.ix_(idx, idx)], b[idx])
            except np.linalg.LinAlgError:
                continue
            denom = float(b[idx] @ y)
            if denom <= 0:
                continue
            x = np.zeros(n)
            x[idx] = y / denom
            if x.min() < -KKT_TOL:
                continue
            x = np.clip(x, 0, None)
            q = float(x @ s @ x)
            grad = s @ x
            off = [j for j in range(n) if j not in supp]
            # KKT: (Sx)_j >= q b_j off support (multiplier of b'x=1 equals q).
            viol = max([0.0] + [q * b[j] - grad[j] for j in off])
            if viol > KKT_TOL * max(1.0, abs(q)):
                continue
            if best is None or q < best[1] - 1e-15:
                best = (x, q, supp, viol)
    if best is None:
        raise RuntimeError('NO_KKT_SUPPORT_FOUND')
    return best


def mvp_shape(sigma_m) -> dict:
    s = covariance(sigma_m, positive=True)
    x, q, supp, viol = _enumerate(s, np.ones(len(s)))
    return result('PASS', shape=x.tolist(), variance=q, support=list(supp), kkt_violation=viol,
                  solver='EXACT_ACTIVE_SET_ENUMERATION_KKT')


def maxdiv_shape(sigma_m) -> dict:
    s = covariance(sigma_m, positive=True)
    sig = np.sqrt(np.diag(s))
    x, q, supp, viol = _enumerate(s, sig)
    v = x / x.sum()
    return result('PASS', shape=v.tolist(), diversification_ratio=diversification_ratio(v, s), support=list(supp),
                  kkt_violation=viol, solver='EXACT_ACTIVE_SET_ENUMERATION_KKT_ON_SIGMA_NORMALISED_QP')


def erc_shape(sigma_m, *, tol: float = 1e-10, floor: float = 1e-8, max_iter: int = 200) -> dict:
    """Shape is scale invariant, so Sigma is normalised by trace/n before Newton steps.
    A residual that stops improving below `floor` is reported as a precision-floor PASS."""
    s = covariance(sigma_m, positive=True)
    n = len(s)
    s = s / (np.trace(s) / n)
    y = 1 / np.sqrt(np.diag(s))
    y = y / np.sqrt(n * float(y @ s @ y))
    trace = []

    def f(z):
        return 0.5 * float(z @ s @ z) - float(np.log(z).sum()) / n

    for it in range(max_iter):
        v = y / y.sum()
        rc = v * (s @ v) / float(v @ s @ v)
        trace.append(float(np.abs(rc - 1 / n).max()))
        stalled = len(trace) > 3 and trace[-1] >= min(trace[:-1]) and trace[-1] < floor
        if trace[-1] < tol or stalled:
            return result('PASS', shape=v.tolist(), iterations=it, residual=trace[-1], trace=trace,
                          solver='NEWTON_LOG_BARRIER_RISK_BUDGET', precision_floor_stop=bool(stalled and trace[-1] >= tol))
        g = s @ y - 1 / (n * y)
        h = s + np.diag(1 / (n * y * y))
        step = np.linalg.solve(h, g)
        t = 1.0
        while np.any(y - t * step <= 0) or f(y - t * step) > f(y) - 1e-4 * t * float(g @ step):
            t *= 0.5
            if t < 1e-16:
                return result('FAIL', 'ERC_LINE_SEARCH_FAILED', trace=trace)
        y = y - t * step
    return result('FAIL', 'ERC_NOT_CONVERGED', trace=trace)


def benchmark_shape(method: str, sigma_m, *, sigma_estimator: str = 'SAMPLE_DEV') -> dict:
    s = covariance(sigma_m, 5, positive=True)
    if method == 'EW':
        return result('PASS', shape=ew_risky_sleeve().tolist(), solver='EXACT')
    if method == 'IVOL':
        return ivol_shape(np.sqrt(np.diag(s)), estimator=sigma_estimator)
    if method == 'MVP':
        return mvp_shape(s)
    if method == 'ERC':
        return erc_shape(s)
    if method == 'MaxDiv':
        return maxdiv_shape(s)
    raise ValueError('UNKNOWN_BENCHMARK')


def structural_deviation(method: str, v, sigma_m, *, target=None, sigma_estimator: str = 'SAMPLE_DEV') -> dict:
    """Native geometry of the CURRENT risky shape v against benchmark `method`.

    Undefined geometry is REPORTED, never coerced into a pass: a zero risky sleeve (v = w_risky / a with
    a = 0, RT07), a degenerate covariance (MVP/MaxDiv not identifiable), a negative risk contribution or a
    zero portfolio variance (ERC, RT04) all return status 'U' with a reason code.  Malformed inputs
    (wrong shape, non-finite, off-simplex, asymmetric covariance) still raise.
    """
    if method not in METHODS:
        raise ValueError('UNKNOWN_BENCHMARK')
    geo = GEOMETRY[method]
    x = np.asarray(v, dtype=float)
    if x.shape != (5,) or not np.isfinite(x).all():
        raise ValueError('INVALID_WEIGHT_VECTOR')
    if abs(float(x.sum())) <= 1e-12:
        return result('U', 'RISKY_SLEEVE_ZERO_SHAPE_UNDEFINED', geometry=geo)
    v = simplex(x, 5)
    try:
        s = covariance(sigma_m, 5, positive=True)
    except ValueError as exc:
        if str(exc) == 'NON_PSD_OR_DEGENERATE_COVARIANCE':
            return result('U', 'COVARIANCE_DEGENERATE_NOT_IDENTIFIABLE', geometry=geo)
        raise
    if target is None:
        try:
            shp = benchmark_shape(method, s, sigma_estimator=sigma_estimator)
        except RuntimeError as exc:
            return result('U', 'TARGET_NOT_IDENTIFIABLE:' + str(exc), geometry=geo)
        if shp['status'] != 'PASS':
            return result('U', 'TARGET_SHAPE_UNAVAILABLE:' + str(shp.get('reason')), geometry=geo)
        t = np.asarray(shp['shape'])
    else:
        t = np.asarray(target)
    try:
        if method in ('EW', 'IVOL'):
            d = float(np.abs(v - t).sum())
        elif method == 'MVP':
            d = max(0.0, float(v @ s @ v - t @ s @ t))
        elif method == 'ERC':
            rc = risk_contributions(v, s)
            if rc.min() < 0:
                return result('U', 'NEGATIVE_RISK_CONTRIBUTION_ERC_GEOMETRY_UNDEFINED', geometry=geo)
            d = float(np.abs(rc - 0.2).max())
        else:
            d = max(0.0, diversification_ratio(t, s) - diversification_ratio(v, s))
    except ValueError as exc:
        if str(exc) == 'ZERO_PORTFOLIO_VARIANCE':
            return result('U', 'ZERO_PORTFOLIO_VARIANCE_GEOMETRY_UNDEFINED', geometry=geo)
        raise
    return result('PASS', method=method, deviation=d, geometry=geo, unit=GEOMETRY_UNITS[geo], target=t.tolist())


def tolerance_check(method: str, deviation: dict, owner_tolerance: dict | None) -> dict:
    if deviation.get('status') != 'PASS':
        return result('U', 'DEVIATION_UNAVAILABLE:' + str(deviation.get('reason')))
    if not owner_ok(owner_tolerance):
        return result('BLOCKED', 'STRUCTURAL_TOLERANCE_OWNER_MISSING', geometry=GEOMETRY[method])
    if owner_tolerance.get('method') != method or owner_tolerance.get('geometry') != GEOMETRY[method]:
        return result('FAIL', 'TOLERANCE_GEOMETRY_MISMATCH', expected=GEOMETRY[method],
                      supplied=owner_tolerance.get('geometry'))
    tol = owner_tolerance.get('value')
    if not isinstance(tol, (int, float)) or not np.isfinite(tol) or tol < 0:
        raise ValueError('TOLERANCE_VALUE_DOMAIN')
    breach = deviation['deviation'] > tol
    # tau_max (the shape drift that may never be left unattended) is a SEPARATE owner number.  It is
    # never derived from a DEV quantile; absent -> None, present -> must not be below the trigger.
    tau_max = owner_tolerance.get('tau_max')
    max_breached = None
    if tau_max is not None:
        if not isinstance(tau_max, (int, float)) or not np.isfinite(tau_max) or tau_max < tol:
            raise ValueError('TAU_MAX_DOMAIN_MUST_BE_FINITE_AND_AT_LEAST_TRIGGER')
        max_breached = bool(deviation['deviation'] > tau_max)
    return result('PASS', within=not breach, deviation=deviation['deviation'], tolerance=tol,
                  geometry=GEOMETRY[method], unit=GEOMETRY_UNITS[GEOMETRY[method]],
                  tau_max=tau_max, max_breached=max_breached, owner_hash=owner_tolerance['owner_hash'])
