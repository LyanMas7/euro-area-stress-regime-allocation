"""RQ03 risk tolerance - one canonical module (spec 'RQ03 risk tolerance for 5 benchmarks and Main', R22/R23).

Three DIFFERENT objects, never merged into one number:
  structural tolerance tau_b   per-benchmark drift from its OWN geometry (L1 / variance regret / RC share / DR regret)
  portfolio loss tolerance     Theta_{s,t}: hard capital envelope K_N / K_S, per-asset caps, vol, VaR, ES, HES,
                               turnover, fee - all measured on OPENING NAV AFTER the fee
  action / sizing tolerance    lambda, b, B_max (not implemented here)
Hard risk limits are a VETO that runs independently of tau: D <= tau_trigger never turns a VaR/ES breach into HOLD.

Post-fee accounting (spec 1), w- = pre-trade weights, Delta = self-financing trade (1'Delta = 0), C = fee in NAV units:
    X = w- + Delta - C e_C        V = 1'X = 1 - C        w+ = X / V        L_j = C - X'r_j   (loss on OPENING NAV)
    VaR_a = C - mu'X + z_a sqrt(X'SX)      ES_a = C - mu'X + phi(z_a)/(1-a) sqrt(X'SX)
    HES_a = min_eta [ eta + sum_j p_j max(L_j - eta, 0) / (1-a) ]
Weight limits are tested on w+, loss limits on (X, C).  The 50-75% / <=50% bands are RISKY-CAPITAL SHARES, not
VaR/ES numbers; the 60% cap binds each risky asset only, never C1.

Nothing here has a numeric default for an owner quantity: a missing cap, fee, tau or utility is BLOCKED, and the
research numbers used elsewhere in the code (ES95 = 2%, 10 bp, lambda = .94, 50% unknown-regime cap) stay labelled
ASSUMED_RESEARCH_ONLY.  Main has a vector Theta_M and NO benchmark tau; RL stays DORMANT.  Estimates from a DEV
quantile grid are retrospective sensitivities (RT08/RT17): never 'optimal', never a derived tau_max, never PIT.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from . import benchmarks as B
from .contract import (ALPHAS, ASSETS, ASSUMED_RESEARCH_ONLY, CAPITAL_SHARE_UNITS, HARD_ENVELOPE, MAIN_IDS, METHODS,
                       REGIMES, RISK_BASIS, RISK_UNIT, RISKY, RL_STATUS, STRESS, Blocked, covariance, owner_ok,
                       require_dev, result, simplex, stamp)
from .new_tail_rev357 import tail_mass_decomposition
from .tail import var_es as scenario_var_es
from .trading import cost_rates
from .utility import utility_fn

EPS = 1e-12
STRATEGY_IDS = (*METHODS, 'MAIN')
FORBIDDEN_STRATEGY_IDS = frozenset(('A', 'B', 'C', 'MODEL_A', 'MODEL_B', 'MODEL_C', 'OWNER_A', 'OWNER_B', 'OWNER_C'))
ALPHA_KEYS = tuple(str(a) for a in ALPHAS)
RETRO_LABEL = 'RETROSPECTIVE_SENSITIVITY_NOT_OPTIMAL_NOT_PIT_CERTIFIED_NOT_OWNER_THRESHOLD'


def _is_num(x) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool) and bool(np.isfinite(x))


# ------------------------------------------------------------------------------------------ (b) hard envelope
def envelope_for(regime: str, *, benchmark_unknown_research: bool = False) -> dict:
    """K_N / K_S.  Regime 'U' has NO envelope; only the opt-in benchmark research fallback maps it to the Stress band."""
    lo_n, hi_n = HARD_ENVELOPE['normal_band']
    cap = HARD_ENVELOPE['asset_cap']
    if regime == 'N':
        return {'name': 'K_N', 'a_lo': lo_n, 'a_hi': hi_n, 'asset_cap': cap}
    if regime in STRESS:
        return {'name': 'K_S', 'a_lo': 0.0, 'a_hi': HARD_ENVELOPE['stress_max'], 'asset_cap': cap}
    if regime == 'U' and benchmark_unknown_research:
        return {'name': 'K_S_UNKNOWN_REGIME_BENCHMARK_RESEARCH_ONLY', 'a_lo': 0.0,
                'a_hi': HARD_ENVELOPE['stress_max'], 'asset_cap': cap, 'label': ASSUMED_RESEARCH_ONLY}
    raise Blocked('REGIME_ENVELOPE_UNDEFINED:' + str(regime))


def risky_share(w6) -> float:
    return float(simplex(w6, 6)[:5].sum())


def in_envelope(w6, regime: str, *, benchmark_unknown_research: bool = False, tol: float = 1e-9) -> dict:
    """Membership of the post-trade weights w+ in K_N / K_S.  Reports every breach; never repairs."""
    w = np.asarray(w6, dtype=float)
    if w.shape != (6,) or not np.isfinite(w).all():
        raise ValueError('SIX_ASSET_WEIGHT_VECTOR_REQUIRED')
    try:
        env = envelope_for(regime, benchmark_unknown_research=benchmark_unknown_research)
    except Blocked as exc:
        return result('U', str(exc), regime=regime)
    a = float(w[:5].sum())
    breaches = []
    if w.min() < -tol:
        breaches.append('LONG_ONLY_OR_CASH_NONNEGATIVE')           # w >= 0 includes X_C1 >= 0
    if abs(float(w.sum()) - 1.0) > 1e-8:
        breaches.append('WEIGHTS_DO_NOT_SUM_TO_ONE')
    if a < env['a_lo'] - tol:
        breaches.append('RISKY_SHARE_BELOW_FLOOR')
    if a > env['a_hi'] + tol:
        breaches.append('RISKY_SHARE_ABOVE_CEILING')
    over = [ASSETS[i] for i in range(5) if w[i] > env['asset_cap'] + tol]   # C1 is never capped
    if over:
        breaches.append('RISKY_ASSET_CAP:' + ','.join(over))
    return result('PASS' if not breaches else 'FAIL', None if not breaches else 'ENVELOPE_BREACH:' + ','.join(breaches),
                  regime=regime, envelope=env['name'], risky_share=a, cash_weight=float(w[5]), breaches=breaches,
                  in_envelope=not breaches, bands_are='RISKY_CAPITAL_SHARES_NOT_VAR_ES')


# ------------------------------------------------------------------------------------ (a) post-fee accounting
def postfee_state(w_pre, delta, cost: float) -> dict:
    """X = w- + Delta - C e_C, V = 1 - C, w+ = X/V.  The fee is paid out of C1 (cash)."""
    p = simplex(w_pre, 6)
    d = np.asarray(delta, dtype=float)
    if d.shape != (6,) or not np.isfinite(d).all():
        raise ValueError('SIX_ASSET_TRADE_VECTOR_REQUIRED')
    if abs(float(d.sum())) > 1e-9:
        raise ValueError('TRADE_VECTOR_MUST_BE_SELF_FINANCING_SUM_ZERO')
    c = float(cost)
    if not (np.isfinite(c) and 0 <= c < 1):
        raise ValueError('COST_FRACTION_DOMAIN')
    x = p + d
    x[5] -= c
    if x.min() < -1e-10:
        raise ValueError('NEGATIVE_POST_TRADE_HOLDING_FEE_EXCEEDS_CASH_OR_SHORT_POSITION')
    x = np.clip(x, 0.0, None)
    v = float(x.sum())
    if abs(v - (1.0 - c)) > 1e-9:
        raise AssertionError('POST_FEE_NAV_IDENTITY')
    return {'X': x, 'V': v, 'w_plus': x / v, 'C': c, 'turnover': float(np.abs(d).sum())}


def scenario_losses(X, C: float, scenarios) -> np.ndarray:
    """L_j = C - X'r_j on OPENING NAV (the fee is a loss; X is NOT the normalised weight vector)."""
    r = np.asarray(scenarios, dtype=float)
    x = np.asarray(X, dtype=float)
    if r.ndim != 2 or r.shape[1] != x.shape[0] or not np.isfinite(r).all() or r.min() <= -1:
        raise ValueError('SCENARIO_MATRIX_MUST_MATCH_SIX_ASSETS_AND_BE_ABOVE_MINUS_ONE')
    return float(C) - r @ x


def _embed_covariance(sigma, n: int) -> np.ndarray:
    s = np.asarray(sigma, dtype=float)
    if s.shape == (n, n):
        return covariance(s, n)
    if s.shape == (5, 5) and n == 6:               # five risky assets; cash has zero variance and covariance
        out = np.zeros((6, 6))
        out[:5, :5] = covariance(s, 5)
        return out
    raise ValueError('COVARIANCE_SHAPE_MUST_MATCH_ASSET_VECTOR')


def gaussian_var_es(X, C: float, sigma, alpha: float, *, mu=None) -> dict:
    """VaR = C - mu'X + z_a sd,  ES = C - mu'X + phi(z_a)/(1-a) sd,  sd = sqrt(X'SX); opening-NAV loss fractions."""
    if not 0 < alpha < 1:
        raise ValueError('ALPHA_DOMAIN')
    x = np.asarray(X, dtype=float)
    s = _embed_covariance(sigma, len(x))
    sd = float(np.sqrt(max(float(x @ s @ x), 0.0)))
    m = 0.0 if mu is None else float(np.asarray(mu, dtype=float) @ x[:len(np.asarray(mu))])
    z = float(norm.ppf(alpha))
    return {'VaR': float(C) - m + z * sd, 'ES': float(C) - m + float(norm.pdf(z)) / (1 - alpha) * sd, 'alpha': alpha,
            'sd_opening_nav': sd, 'basis': RISK_BASIS, 'unit': RISK_UNIT,
            'mean_assumption': 'AS_OF_MU_SUPPLIED' if mu is not None else 'ASSUMED_ZERO_MEAN_' + ASSUMED_RESEARCH_ONLY}


def historical_es(losses, alpha: float, mass=None) -> dict:
    """HES_a = min_eta [eta + E(L - eta)+ / (1-a)] (Rockafellar-Uryasev).  The minimiser is a sample loss."""
    x = np.asarray(losses, dtype=float)
    if x.ndim != 1 or len(x) == 0 or not np.isfinite(x).all() or not 0 < alpha < 1:
        raise ValueError('INVALID_LOSS_OR_ALPHA')
    p = np.full(len(x), 1 / len(x)) if mass is None else np.asarray(mass, dtype=float)
    if p.shape != x.shape or not np.isfinite(p).all() or p.min() < 0 or abs(float(p.sum()) - 1) > 1e-10:
        raise ValueError('INVALID_SCENARIO_MASS')
    cand = np.unique(x)
    vals = np.array([eta + float(np.dot(p, np.maximum(x - eta, 0.0))) / (1 - alpha) for eta in cand])
    k = int(np.argmin(vals))
    return {'HES': float(vals[k]), 'eta_star': float(cand[k]), 'alpha': alpha, 'n_scenarios': int(len(x)),
            'basis': RISK_BASIS, 'unit': RISK_UNIT}


def tail_support_report(losses, alpha: float, mass=None, *, overlap_horizon: int | None = None) -> dict:
    """What a tail estimate actually rests on.  Tail scenario-weeks are NOT independent crisis episodes."""
    x = np.asarray(losses, dtype=float)
    d = tail_mass_decomposition(x, mass, alpha)
    n = int(len(x))
    out = {'n_scenarios': n, 'nominal_tail_scenario_count': n * (1 - alpha), 'n_positive_tail_atoms': d['n_positive_tail_atoms'],
           'effective_tail_sample': d['n_eff_tail_actual'], 'effective_global_sample': d['n_eff_global'],
           'independent_episode_count': None,
           'dependence': 'SCENARIO_WEEKS_ARE_SERIALLY_DEPENDENT_NOT_INDEPENDENT_CRISES', 'alpha': alpha}
    if overlap_horizon is not None:
        if type(overlap_horizon) is not int or overlap_horizon < 1:
            raise ValueError('HORIZON_DOMAIN')
        out['overlapping_windows'] = overlap_horizon > 1
        out['max_disjoint_windows_upper_bound'] = n // overlap_horizon
    return out


def two_layer_cost(c_common: float, c_bottom: float) -> dict:
    """C_total = C_G + (1-C_G) C_B,  V_post = (1-C_G)(1-C_B)  (reuses main_lock; the naive C_G + C_B is wrong)."""
    from .main_lock import total_two_layer_cost
    return total_two_layer_cost(c_common, c_bottom)


def evaluate_candidate_risk(w_pre, delta, cost: float, *, regime: str, caps, sigma=None, mu=None, scenarios=None,
                            mass=None, benchmark_unknown_research: bool = False) -> dict:
    """Global veto on ONE candidate: envelope on w+, then every ACTIVATED loss cap on (X, C).

    caps = {'risk_basis': NET_AFTER_COST, 'unit': DECIMAL_LOSS_FRACTION_OF_NAV, 'horizon_weeks': 1,
            'limits': {'0.95': {'VaR': v, 'ES': e}, ...},   # tested on Gaussian(sigma, mu) AND on scenarios, if supplied
            'hes': {'0.95': h, ...},                        # optional, tested on the historical scenarios
            'vol_cap': s, 'turnover_cap': T, 'cost_cap': c}  # optional
    No default exists for any of them: missing caps or an activated cap without its input is BLOCKED.
    """
    if not isinstance(caps, dict) or not caps.get('limits'):
        return result('BLOCKED', 'RISK_CAPS_OWNER_REQUIRED_NO_DEFAULT_EXISTS')
    if caps.get('risk_basis') != RISK_BASIS or caps.get('unit') != RISK_UNIT:
        return result('BLOCKED', 'RISK_OWNER_UNIT_OR_BASIS_MUST_BE_NET_AFTER_COST_OPENING_NAV_FRACTION')
    if caps.get('horizon_weeks', 1) != 1:
        return result('BLOCKED', 'MULTI_WEEK_HORIZONS_NEED_JOINT_PATHS_NOT_SQRT_TIME_SCALING')
    st = postfee_state(w_pre, delta, cost)
    X, C, w_plus = st['X'], st['C'], st['w_plus']
    env = in_envelope(w_plus, regime, benchmark_unknown_research=benchmark_unknown_research)
    if env['status'] == 'U':
        return result('U', env['reason'], regime=regime)
    breaches = list(env['breaches'])
    measures: dict = {}
    limits = caps['limits']
    if sigma is None and scenarios is None:
        return result('BLOCKED', 'NO_RISK_INPUT_SIGMA_OR_SCENARIOS_FOR_ACTIVATED_CAPS')
    for ak, lim in limits.items():
        if ak not in ALPHA_KEYS or set(lim) - {'VaR', 'ES'} or not all(_is_num(v) and v >= 0 for v in lim.values()):
            return result('BLOCKED', 'FINITE_NONNEGATIVE_VAR_ES_LIMITS_FOR_REGISTERED_ALPHAS_REQUIRED')
        alpha = float(ak)
        if sigma is not None:
            g = gaussian_var_es(X, C, sigma, alpha, mu=mu)
            measures[f'GAUSSIAN@{ak}'] = g
            breaches += [f'GAUSSIAN_{k}@{ak}' for k in lim if g[k] > lim[k] + 1e-12]
        if scenarios is not None:
            loss = scenario_losses(X, C, scenarios)
            sv = scenario_var_es(loss, alpha, mass)
            measures[f'SCENARIO@{ak}'] = sv
            breaches += [f'SCENARIO_{k}@{ak}' for k in lim if sv[k] > lim[k] + 1e-12]
    for ak, h in (caps.get('hes') or {}).items():
        if scenarios is None:
            return result('BLOCKED', 'HES_CAP_ACTIVATED_BUT_NO_HISTORICAL_SCENARIOS')
        if ak not in ALPHA_KEYS or not (_is_num(h) and h >= 0):
            return result('BLOCKED', 'FINITE_NONNEGATIVE_HES_LIMIT_REQUIRED')
        hv = historical_es(scenario_losses(X, C, scenarios), float(ak), mass)
        measures[f'HES@{ak}'] = hv
        if hv['HES'] > h + 1e-12:
            breaches.append(f'HES@{ak}')
    if caps.get('vol_cap') is not None:
        if sigma is None or not _is_num(caps['vol_cap']) or caps['vol_cap'] < 0:
            return result('BLOCKED', 'VOL_CAP_NEEDS_SIGMA_AND_A_FINITE_NONNEGATIVE_LIMIT')
        s6 = _embed_covariance(sigma, 6)
        vol = float(np.sqrt(max(float(w_plus @ s6 @ w_plus), 0.0)))
        measures['VOL_WEIGHT_BASIS'] = vol
        if vol > caps['vol_cap'] + 1e-12:
            breaches.append('VOL')
    if caps.get('turnover_cap') is not None and st['turnover'] > caps['turnover_cap'] + 1e-12:
        breaches.append('TURNOVER')
    if caps.get('cost_cap') is not None and C > caps['cost_cap'] + 1e-12:
        breaches.append('COST')
    ok = not breaches
    return result('PASS' if ok else 'FAIL', None if ok else 'RISK_CAP_BREACH:' + ','.join(breaches), feasible=ok,
                  breaches=breaches, measures=measures, envelope=env['envelope'], risky_share=env['risky_share'],
                  cost_fraction=C, turnover=st['turnover'], w_plus=w_plus.tolist(),
                  precedence='HARD_VETO_BEFORE_UTILITY_AND_INDEPENDENT_OF_STRUCTURAL_TAU')


# ------------------------------------------------------------------ (d) one-sided structural deviation series
def risky_shape(w6) -> dict:
    """v = w_risky / a.  a = 0 -> the shape is UNDEFINED (RT07): no 0/0, activation needs its own owner mandate."""
    w = simplex(w6, 6)
    a = float(w[:5].sum())
    if a <= EPS:
        return result('U', 'RISKY_SLEEVE_ZERO_SHAPE_UNDEFINED_ACTIVATION_NEEDS_OWNER_MANDATE', a=a)
    return result('PASS', a=a, shape=(w[:5] / a).tolist())


def drift_shape(v_star_prev, r_prev) -> dict:
    """v~_t = v*_{t-1} (x) (1 + r_{1:5,t-1}) / 1'[...]   (drift of last week's target over the week it earned)."""
    v = simplex(v_star_prev, 5)
    r = np.asarray(r_prev, dtype=float)[:5]
    if r.shape != (5,) or not np.isfinite(r).all() or r.min() <= -1:
        raise ValueError('FIVE_RISKY_RETURNS_ABOVE_MINUS_ONE_REQUIRED')
    g = v * (1 + r)
    s = float(g.sum())
    if s <= EPS:
        return result('U', 'DRIFTED_SLEEVE_ZERO')
    return result('PASS', shape=(g / s).tolist())


def structural_deviation_series(method: str, shapes, week_returns, covs, *, week_ends=None,
                                cov_observed_through=None) -> dict:
    """D_{b,t} = D_b(v~_{b,t}, v*_{b,t}, Sigma_t) for t = 1..n-1, strictly one-sided.

    shapes[k]       v*_k  target issued at issue k                  covs[k]  Sigma_k as-of issue k
    week_returns[k] returns EARNED after issue k (during week k+1)  -> v~_t drifts shapes[t-1] with week_returns[t-1]
    Sigma_t may not contain week_returns[t] (the week about to be traded): with `cov_observed_through` and
    `week_ends` (week_ends[k] = end of week_returns[k]) the information stamp of Sigma_t must be <= week_ends[t-1].
    Undefined geometries stay in the table with status 'U' (RT04/RT07); nothing is imputed.
    """
    v = np.asarray(shapes, dtype=float)
    r = np.asarray(week_returns, dtype=float)
    s = np.asarray(covs, dtype=float)
    n = len(v)
    if v.ndim != 2 or v.shape[1] != 5 or r.ndim != 2 or len(r) not in (n, n - 1) or r.shape[1] not in (5, 6) \
            or s.shape != (n, 5, 5):
        raise ValueError('SHAPES_n5_RETURNS_n_or_n_minus_1_x5or6_COVARIANCES_n55_REQUIRED')
    if week_ends is not None:
        if len(week_ends) != len(r):
            raise ValueError('WEEK_ENDS_LENGTH_MUST_MATCH_WEEK_RETURNS')
        for we in week_ends:
            require_dev(we)                                    # OOS dates -> Blocked('OOS_SEALED:...')
    if cov_observed_through is not None:
        if week_ends is None or len(cov_observed_through) != n:
            raise ValueError('INFORMATION_STAMPS_NEED_WEEK_ENDS_AND_MATCHING_LENGTH')
        for t in range(1, n):
            if stamp(cov_observed_through[t]) > stamp(week_ends[t - 1]):
                raise Blocked('SAME_WEEK_OR_FUTURE_INFORMATION_IN_COVARIANCE')
    rows, n_defined = [], 0
    for t in range(1, n):
        row = {'t': t, 'method': method, 'geometry': B.GEOMETRY[method]}
        if week_ends is not None:
            row['week_end'] = str(week_ends[t - 1])
        dr = drift_shape(v[t - 1], r[t - 1])
        if dr['status'] != 'PASS':
            rows.append({**row, 'status': 'U', 'reason': dr['reason'], 'deviation': None})
            continue
        d = B.structural_deviation(method, dr['shape'], s[t], target=v[t])
        ok = d['status'] == 'PASS'
        n_defined += ok
        rows.append({**row, 'status': d['status'], 'reason': d.get('reason'),
                     'deviation': d['deviation'] if ok else None})
    return result('PASS', rows=rows, n_rows=len(rows), n_defined=n_defined, n_undefined=len(rows) - n_defined,
                  unit=B.GEOMETRY_UNITS[B.GEOMETRY[method]], information='ONE_SIDED_TARGET_PREV_WEEK_DRIFT_COV_ASOF')


def retrospective_threshold_candidates(rows, quantiles=(0.0, 0.5, 0.9, 1.0)) -> dict:
    """Candidate tau grid = quantiles of the DEV drift series D_b.  A SENSITIVITY GRID, not a selection.

    The quantiles are of the structural-deviation distribution (not loss quantiles); tau_max is NEVER derived from
    them; nothing is optimal, PIT-certified or owner-approved (RT08/RT17).  Rows dated after 2019-12-31 -> BLOCKED.
    """
    q = np.asarray(quantiles, dtype=float)
    if q.ndim != 1 or not len(q) or not np.isfinite(q).all() or q.min() < 0 or q.max() > 1:
        raise ValueError('THRESHOLD_QUANTILES_MUST_BE_IN_0_TO_1')
    vals = []
    for r in rows:
        if r.get('week_end') is not None:
            try:
                require_dev(r['week_end'])
            except Blocked as exc:
                return result('BLOCKED', str(exc))
        if r.get('status') == 'PASS' or (r.get('status') is None and r.get('deviation') is not None):
            vals.append(float(r['deviation']))
    if not vals:
        return result('U', 'NO_DEFINED_STRUCTURAL_DEVIATION')
    cand = sorted(set(float(z) for z in np.quantile(vals, q)))
    return result('PASS', candidates=cand, grid_quantiles=q.tolist(), n_observations=len(vals), claim=RETRO_LABEL,
                  selected=None, tau_max='UNDEFINED_NOT_DERIVED_FROM_QUANTILE', owner_approved=False,
                  pit_certified=False, status_of_best_dev_result='RESEARCH_FINDING_ONLY')


def trigger_decision(deviation: dict, tolerance: dict | None, *, risk_breach: bool) -> dict:
    """tau_trigger / tau_max / Theta_risk stay separate.  A risk breach is never hidden by D <= tau."""
    if not isinstance(deviation, dict) or deviation.get('status') != 'PASS':
        return result('U', 'DEVIATION_UNAVAILABLE:' + str((deviation or {}).get('reason')),
                      action='NO_SHAPE_DECISION', risk_repair_required=bool(risk_breach))
    if not isinstance(tolerance, dict) or not _is_num(tolerance.get('tau_trigger')) or tolerance['tau_trigger'] < 0:
        return result('BLOCKED', 'TAU_TRIGGER_OWNER_REQUIRED_NO_DEFAULT', risk_repair_required=bool(risk_breach))
    d, trig = float(deviation['deviation']), float(tolerance['tau_trigger'])
    tmax = tolerance.get('tau_max')
    if tmax is not None and (not _is_num(tmax) or tmax < trig):
        return result('BLOCKED', 'TAU_MAX_DOMAIN_MUST_BE_FINITE_AND_AT_LEAST_TRIGGER')
    breached = d > trig
    if risk_breach:
        action = 'RISK_REPAIR_REQUIRED'
    elif breached:
        action = 'REBALANCE_PROPOSAL_SUBJECT_TO_ALL_HARD_GATES'
    else:
        action = 'HOLD_ALLOWED_IF_HOLD_IS_RISK_VALID'
    return result('PASS', deviation=d, tau_trigger=trig, trigger_breached=breached, tau_max=tmax,
                  max_breached=None if tmax is None else bool(d > tmax), risk_repair_required=bool(risk_breach),
                  action=action, tolerance_overrides_risk=False)


# ------------------------------------------------------------------------------ (c) tolerance registry
def _risk_caps_errors(rc) -> list:
    err = []
    if not isinstance(rc, dict):
        return ['RISK_CAPS_REQUIRED']
    if rc.get('risk_basis') != RISK_BASIS or rc.get('unit') != RISK_UNIT:
        err.append('RISK_CAPS_UNIT_OR_BASIS_MUST_BE_NET_AFTER_COST_OPENING_NAV_FRACTION')
    h = rc.get('horizon_weeks')
    if type(h) is not int or h < 1:
        err.append('RISK_CAPS_HORIZON_WEEKS_REQUIRED')
    lim = rc.get('limits')
    if not isinstance(lim, dict) or not lim:
        err.append('RISK_CAPS_AT_LEAST_ONE_ACTIVATED_ALPHA_REQUIRED')
    else:
        for ak, v in lim.items():
            if ak not in ALPHA_KEYS or not isinstance(v, dict) or set(v) != {'VaR', 'ES'} \
                    or not all(_is_num(x) and x >= 0 for x in v.values()):
                err.append('RISK_CAPS_ALPHA_LIMITS_MALFORMED:' + str(ak))
    for ak, hv in (rc.get('hes') or {}).items():
        if ak not in ALPHA_KEYS or not (_is_num(hv) and hv >= 0):
            err.append('RISK_CAPS_HES_MALFORMED:' + str(ak))
    for k in ('vol_cap', 'turnover_cap', 'cost_cap'):
        if rc.get(k) is not None and not (_is_num(rc[k]) and rc[k] >= 0):
            err.append('RISK_CAPS_' + k.upper() + '_MALFORMED')
    return err


def _asset_caps_errors(caps) -> list:
    if not isinstance(caps, dict):
        return ['ASSET_CAPS_REQUIRED']
    err = []
    if 'C1' in caps:
        err.append('ASSET_CAP_NOT_APPLICABLE_TO_C1_CASH_IS_THE_RESIDUAL')
    if set(caps) - {'C1'} != set(RISKY):
        err.append('ASSET_CAPS_MUST_COVER_EXACTLY_THE_FIVE_RISKY_ASSETS')
    else:
        for a in RISKY:
            if not (_is_num(caps[a]) and 0 <= caps[a] <= HARD_ENVELOPE['asset_cap']):
                err.append('ASSET_CAP_OUTSIDE_0_TO_ENVELOPE_60PCT:' + a)
    return err


def _band_errors(bands) -> list:
    if not isinstance(bands, dict) or set(bands) != set(REGIMES):
        return ['RISKY_SHARE_BAND_FOR_ALL_FOUR_REGIMES_REQUIRED']
    err = []
    for reg, band in bands.items():
        try:
            lo, hi = band
        except (TypeError, ValueError):
            err.append('RISKY_SHARE_BAND_MALFORMED:' + reg)
            continue
        env = envelope_for(reg)
        if not (_is_num(lo) and _is_num(hi)) or not env['a_lo'] - 1e-12 <= lo <= hi <= env['a_hi'] + 1e-12:
            err.append(f"RISKY_SHARE_BAND_OUTSIDE_{env['name']}:{reg}")
    return err


def _owner_errors(rec) -> tuple[list, list]:
    err, labels = [], []
    if not isinstance(rec.get('owner_hash'), str) or not rec['owner_hash']:
        err.append('OWNER_HASH_REQUIRED')
    if rec.get('approved') is not True:
        if rec.get('assumed_research_only') is True:
            labels.append(ASSUMED_RESEARCH_ONLY)
        else:
            err.append('OWNER_APPROVAL_REQUIRED_OR_EXPLICIT_ASSUMED_RESEARCH_ONLY_LABEL')
    if not rec.get('evidence_ids'):
        err.append('EVIDENCE_IDS_REQUIRED')
    if not rec.get('effective_from'):
        err.append('EFFECTIVE_FROM_REQUIRED')
    return err, labels


def _nested(x, assumed: bool):
    """A nested owner record counts when approved, or - only inside an ASSUMED_RESEARCH_ONLY record - when it is
    explicitly flagged as an assumption (it never becomes 'approved')."""
    if owner_ok(x):
        return x
    if assumed and isinstance(x, dict) and x.get('assumed_research_only') is True and x.get('owner_hash'):
        return {**x, 'approved': True}
    return None


def _common_errors(rec) -> tuple[list, list]:
    err, labels = _owner_errors(rec)
    assumed = rec.get('assumed_research_only') is True
    if rec.get('risk_unit') not in CAPITAL_SHARE_UNITS:
        err.append('RISK_UNIT_MUST_BE_ONE_OF_' + '|'.join(CAPITAL_SHARE_UNITS))
    err += _asset_caps_errors(rec.get('asset_caps'))
    err += _band_errors(rec.get('risky_share_band'))
    err += _risk_caps_errors(rec.get('risk_caps'))
    if not (_is_num(rec.get('turnover_cap')) and rec['turnover_cap'] >= 0):
        err.append('TURNOVER_CAP_REQUIRED')
    cs = _nested(rec.get('cost_schedule'), assumed)
    if cs is None:
        err.append('COST_SCHEDULE_OWNER_REQUIRED_NO_DEFAULT_FEE')
    else:
        try:
            cost_rates(cs)
        except Blocked as exc:
            err.append('COST_SCHEDULE:' + str(exc))
    ut = _nested(rec.get('utility'), assumed)
    if ut is None or utility_fn(ut) is None:
        err.append('APPROVED_UTILITY_AND_RISK_AVERSION_OWNER_REQUIRED')
    for k in ('forced_repair_owner', 'emergency_override'):
        v = rec.get(k)
        if v is not None and _nested(v, assumed) is None:
            err.append(k.upper() + '_MUST_BE_AN_APPROVED_OWNER_RECORD')
    return err, labels


def geometry_threshold_errors(strategy_id: str, rec, *, reference_benchmark: str | None = None) -> tuple[list, list]:
    """Early check of the per-strategy `geometry_threshold` record that the issued runners feed to
    benchmarks.tolerance_check.  Returns (errors, warnings).

    MAIN has a vector Theta_M and no benchmark tau: the issued runners still anchor POLICY to an owner-chosen
    reference benchmark's geometry, which is reported as a WARNING (it is a deviation, not silently accepted).
    """
    err, warn = [], []
    if strategy_id in MAIN_IDS:
        geo = reference_benchmark
        if geo not in METHODS:
            err.append('POLICY_REFERENCE_BENCHMARK_OWNER_REQUIRED')
        warn.append('MAIN_BORROWS_REFERENCE_BENCHMARK_TAU:' + str(geo))
    elif strategy_id in METHODS:
        geo = strategy_id
    else:
        return ['UNREGISTERED_STRATEGY_ID'], warn
    if not owner_ok(rec):
        return err + ['TOLERANCE_OWNER_RECORD_REQUIRED'], warn
    if geo in METHODS:
        if rec.get('method') != geo or rec.get('geometry') != B.GEOMETRY[geo]:
            err.append('TOLERANCE_GEOMETRY_MISMATCH')
        if rec.get('unit') is not None and rec['unit'] != B.GEOMETRY_UNITS[B.GEOMETRY[geo]]:
            err.append('TOLERANCE_UNIT_MISMATCH')
    v = rec.get('value')
    if not (_is_num(v) and v >= 0):
        err.append('TAU_VALUE_DOMAIN')
    tm = rec.get('tau_max')
    if tm is not None and not (_is_num(tm) and _is_num(v) and tm >= v):
        err.append('TAU_MAX_DOMAIN')
    return err, warn


def validate_theta_main(theta) -> dict:
    """Theta_M = (K_s by regime, asset caps, vol/VaR/ES/HES caps, turnover, fee, utility hurdle).  No benchmark tau."""
    if not isinstance(theta, dict):
        return result('BLOCKED', 'THETA_MAIN_RECORD_REQUIRED')
    err, labels = _common_errors(theta)
    err = [e for e in err if not e.startswith('RISKY_SHARE_BAND_')]        # Main states per-regime envelopes instead
    if theta.get('strategy_id') not in MAIN_IDS:
        err.append('STRATEGY_ID_MUST_BE_MAIN')
    bad = sorted(k for k in theta if str(k).lower().startswith('tau') or k in ('native_geometry', 'geometry_threshold',
                                                                              'tolerances', 'reference_benchmark_tau'))
    if bad:
        err.append('MAIN_USES_NO_BENCHMARK_TAU:' + ','.join(bad))
    env = theta.get('regime_envelopes')
    if not isinstance(env, dict) or set(env) != set(REGIMES):
        err.append('REGIME_ENVELOPES_FOR_ALL_FOUR_REGIMES_REQUIRED')
    else:
        for reg, e in env.items():
            k = envelope_for(reg)
            if not (isinstance(e, dict) and _is_num(e.get('a_lo')) and _is_num(e.get('a_hi'))
                    and _is_num(e.get('asset_cap'))):
                err.append('REGIME_ENVELOPE_MALFORMED:' + reg)
            elif not (k['a_lo'] - 1e-12 <= e['a_lo'] <= e['a_hi'] <= k['a_hi'] + 1e-12
                      and 0 <= e['asset_cap'] <= k['asset_cap'] + 1e-12):
                err.append(f"REGIME_ENVELOPE_OUTSIDE_{k['name']}:{reg}")
    hurdle = theta.get('utility_hurdle')
    if not (isinstance(hurdle, dict) and _is_num(hurdle.get('value')) and hurdle['value'] >= 0
            and hurdle.get('unit') == 'EXPECTED_UTILITY_DELTA'):
        err.append('UTILITY_HURDLE_IN_EXPECTED_UTILITY_UNITS_REQUIRED')
    if theta.get('rl_status') != RL_STATUS:
        err.append('RL_MUST_STAY_DORMANT')
    bp = theta.get('bottom_permissions')
    if bp is not None:
        if not isinstance(bp, dict) or not bp.get('episode_id') or not bp.get('expires_at') \
                or bp.get('entry_gate') is not True or not set(bp.get('assets', ())) <= set(RISKY) or not bp.get('assets'):
            err.append('BOTTOM_PERMISSIONS_NEED_EPISODE_ASSETS_ENTRY_GATE_EXPIRY')
    if err:
        return result('BLOCKED', 'THETA_MAIN_INVALID:' + err[0], errors=err, labels=labels, official_ready=False)
    return result('PASS', errors=[], labels=labels, official_ready=not labels, uses_benchmark_tau=False,
                  rl=RL_STATUS)


def validate_tolerance_registry(registry) -> dict:
    """Registry of the five benchmark taus + Theta_M.  REFUSES (BLOCKED) a shared tau across geometries or units,
    model A/B/C as benchmarks, a missing owner number (never defaulted), a tau_max derived from a quantile, a
    benchmark tau on Main and bottom permissions on a benchmark."""
    if not isinstance(registry, dict) or not isinstance(registry.get('strategies'), dict):
        return result('BLOCKED', 'REGISTRY_WITH_STRATEGIES_MAPPING_REQUIRED')
    errors, labels = [], []

    def add(sid, code):
        errors.append({'strategy': sid, 'code': code})

    for k in ('tau', 'shared_tau', 'tau_trigger', 'tolerance'):
        if k in registry:
            add('*', 'GEOMETRY_UNIT_MISMATCH_SHARED_TAU_ACROSS_BENCHMARKS')
    recs = {}
    for sid, rec in registry['strategies'].items():
        key = 'MAIN' if sid in MAIN_IDS else sid
        if sid in FORBIDDEN_STRATEGY_IDS or str(sid).upper().startswith('MODEL_'):
            add(sid, 'MODEL_A_B_C_ARE_NOT_RQ03_BENCHMARKS')
        elif key not in STRATEGY_IDS:
            add(sid, 'UNREGISTERED_STRATEGY_ID')
        elif key in recs:
            add(sid, 'DUPLICATE_STRATEGY_RECORD')
        elif not isinstance(rec, dict):
            add(sid, 'RECORD_MUST_BE_A_MAPPING')
        else:
            recs[key] = rec
    for sid in sorted(set(STRATEGY_IDS) - set(recs)):
        add(sid, 'STRATEGY_RECORD_MISSING')
    geo_units = B.GEOMETRY_UNITS
    seen_record, seen_value = {}, {}
    per = {}
    for sid, rec in recs.items():
        if sid == 'MAIN':
            th = validate_theta_main(rec)
            per[sid] = th
            for code in th.get('errors', []):
                add(sid, code)
            labels += [(sid, x) for x in th.get('labels', [])]
            continue
        err, lab = _common_errors(rec)
        geo = B.GEOMETRY[sid]
        if rec.get('native_geometry') != geo:
            err.append('NATIVE_GEOMETRY_MUST_BE_' + geo)
        if rec.get('tau_unit') is None:
            err.append('TAU_UNIT_REQUIRED')
        elif rec['tau_unit'] != geo_units[geo]:
            err.append('TAU_UNIT_GEOMETRY_MISMATCH')
        trig = rec.get('tau_trigger')
        if not (_is_num(trig) and trig >= 0):
            err.append('TAU_TRIGGER_OWNER_NUMBER_REQUIRED')
        tmax = rec.get('tau_max')
        if tmax is not None and not (_is_num(tmax) and _is_num(trig) and tmax >= trig):
            err.append('TAU_MAX_MUST_BE_FINITE_AND_AT_LEAST_TAU_TRIGGER')
        if 'QUANTILE' in str(rec.get('tau_max_basis', '')).upper():
            err.append('TAU_MAX_NOT_DERIVABLE_FROM_QUANTILE')
        basis = str(rec.get('tau_trigger_basis', '')).upper()
        if any(w in basis for w in ('RETROSPECTIVE', 'FRONTIER', 'QUANTILE')):
            lab.append('RESEARCH_PRIORITY_NOT_OWNER_THRESHOLD')
        if rec.get('covariance_estimator') not in B.SIGMA_ESTIMATORS:
            err.append('COVARIANCE_ESTIMATOR_MUST_BE_A_NAMED_REGISTERED_ONE')
        elif B.SIGMA_ESTIMATORS[rec['covariance_estimator']] == 'REALIZED_FULL_DEV':
            lab.append('RETROSPECTIVE_ESTIMATOR_NOT_PIT')
        if rec.get('bottom_permissions') is not None:
            err.append('BOTTOM_PERMISSIONS_ARE_MAIN_ONLY')
        rid = rec.get('tau_record_id')
        if rid is not None:
            if rid in seen_record:
                err.append('TAU_RECORD_SHARED_ACROSS_STRATEGIES:' + seen_record[rid])
            seen_record[rid] = sid
        if _is_num(trig) and isinstance(rec.get('owner_hash'), str):
            same = seen_value.setdefault((rec['owner_hash'], float(trig)), [])
            if any(B.GEOMETRY[o] != geo for o in same):
                err.append('GEOMETRY_UNIT_MISMATCH_SAME_TAU_VALUE_AND_OWNER_RECORD_ACROSS_GEOMETRIES')
            same.append(sid)
        per[sid] = {'errors': err, 'labels': lab}
        for code in err:
            add(sid, code)
        labels += [(sid, x) for x in lab]
    if errors:
        return result('BLOCKED', f"REGISTRY_INVALID:{errors[0]['strategy']}:{errors[0]['code']}", errors=errors,
                      labels=labels, official_ready=False, per_strategy=per)
    return result('PASS', errors=[], labels=labels, official_ready=not labels, per_strategy=per,
                  note='STRUCTURAL_TAU_PER_GEOMETRY_AND_THETA_MAIN_VALIDATED_NO_VALUE_DEFAULTED')
