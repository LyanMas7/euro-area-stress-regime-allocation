"""W05 REV358 exact topology of a registered proportional-cost action ray.

The REV357 comparator is untouched. This challenger never merges a forbidden
gap because it is smaller than a numerical tolerance, and records open/closed
endpoints of tail-support cells. This is 1D completeness, not six-dimensional
global portfolio optimality. No model/threshold fit or OOS data is consumed.
"""
from __future__ import annotations

from itertools import combinations
import numpy as np
from scipy.optimize import minimize_scalar

from .contract import ALPHAS, canonical_sha256, simplex
from .tail import var_es
from .new_tail_rev357 import var_es_actual_tail_v1, tail_mass_decomposition
from .trading import self_financing_cost, cost_rates
from .utility import utility_fn
from .new_optimizer_rev357 import legal_partial_lambda_intervals

VERSION = 'W05_REV358_RAY_TOPOLOGY_V1'


def legal_partial_lambda_intervals_v2(pre, target, cap6, permission, *, regime,
                                     normal_floor_activation_approved=False, tolerance=1e-10):
    """Preserve incumbent schema, with no tolerance relaxation of hard caps."""
    old = legal_partial_lambda_intervals(pre, target, cap6, permission, regime=regime,
        normal_floor_activation_approved=normal_floor_activation_approved, tolerance=tolerance)
    p,t = simplex(pre,6),simplex(target,6)
    if regime == 'N' and p[:5].sum() < .5 and not normal_floor_activation_approved:
        return {'status':'BLOCKED','reason':'NORMAL_FLOOR_ACTIVATION_OWNER_REQUIRED','segments':[]}
    constraints = old.get('constraints',old.get('certificate'))
    if constraints is None:
        return old
    interval=(0.,1.)
    for name,intercept,slope in constraints:
        interval=_intersect_affine(*interval,-intercept,-slope)
        if interval is None:
            return {'status':'INFEASIBLE','reason':'EXACT_LINEAR_CAP_OR_PERMISSION_CONFLICT',
                    'binding_constraint':name,'segments':[],'certificate':constraints}
    return {'status':'PASS','segments':[list(interval)],'constraints':constraints,
            'version':VERSION,'tolerance_relaxes_hard_constraints':False}


def _intersect_affine(lo, hi, intercept, slope):
    """Return exact interval where intercept+slope*x <= 0 (no tolerance gap)."""
    if slope == 0:
        return None if intercept > 0 else (lo, hi)
    bound = -intercept / slope
    lo, hi = (lo, min(hi, bound)) if slope > 0 else (max(lo, bound), hi)
    return None if lo > hi else (lo, hi)


def merge_topological_segments_v1(segments):
    """Union overlapping intervals only; preserve singleton holes and gaps."""
    merged = []
    for raw in sorted(segments, key=lambda z: (z['lower'], not z['lower_closed'], z['upper'])):
        cell = dict(raw)
        if cell['lower'] > cell['upper'] or (cell['lower'] == cell['upper'] and not (cell['lower_closed'] and cell['upper_closed'])):
            continue
        joins = (merged and (cell['lower'] < merged[-1]['upper'] or
                 (cell['lower'] == merged[-1]['upper'] and (cell['lower_closed'] or merged[-1]['upper_closed']))))
        if not joins:
            merged.append(cell)
            continue
        previous = merged[-1]
        if cell['lower'] == previous['lower']:
            previous['lower_closed'] |= cell['lower_closed']
        if cell['upper'] > previous['upper']:
            previous['upper'], previous['upper_closed'] = cell['upper'], cell['upper_closed']
        elif cell['upper'] == previous['upper']:
            previous['upper_closed'] |= cell['upper_closed']
    return merged


def solve_feasible_segments_v2(pre, target, scenarios, mass, rates, budgets, linear_domain,
                               *, numerical_tolerance, support_owner=None, max_cells=None):
    """All VaR/ES feasible cells on a registered action ray, including atoms.

    If w(lambda)=p+lambda*d, fixed-point cost has c=z*lambda/(1+z*lambda)
    where z=c(1)/(1-c(1)). Scenario NET loss is
      [L0_j+(Ld_j+z)*lambda]/[1+z*lambda].
    VaR budget crossings and loss-order crossings partition the ray. Each ES
    inequality has an affine numerator in a cell. max_cells limits computation
    only: an exhausted budget returns PARTIAL_SEARCH, never INFEASIBLE.
    """
    p, t = simplex(pre, 6), simplex(target, 6)
    r = np.asarray(scenarios, float)
    if r.ndim != 2 or r.shape[1] != 6 or not len(r) or not np.isfinite(r).all() or np.any(r <= -1):
        raise ValueError('SCENARIO_DOMAIN')
    if not np.isfinite(numerical_tolerance) or numerical_tolerance <= 0:
        raise ValueError('POSITIVE_NUMERICAL_TOLERANCE_REQUIRED')
    if max_cells is not None and (type(max_cells) is not int or max_cells < 0):
        raise ValueError('NONNEGATIVE_INTEGER_CELL_BUDGET_REQUIRED')
    if budgets.get('risk_basis') != 'NET_AFTER_COST' or set(budgets.get('limits', {})) != set(map(str, ALPHAS)):
        return {'status': 'BLOCKED', 'reason': 'ALL_ALPHA_NET_BUDGETS_REQUIRED'}
    limits = budgets['limits']
    if any(set(limits[str(a)]) != {'VaR', 'ES'} or not all(np.isfinite(x) for x in limits[str(a)].values()) for a in ALPHAS):
        raise ValueError('FINITE_ALL_ALPHA_LIMITS_REQUIRED')
    prob = np.full(len(r), 1/len(r)) if mass is None else np.asarray(mass, float)
    var_es(np.zeros(len(r)), .95, prob)
    if support_owner is not None:
        gate = var_es_actual_tail_v1(np.zeros(len(r)), .95, prob, support_owner=support_owner)
        if gate['support_gate_status'] == 'BLOCKED':
            return {'status': 'BLOCKED', 'reason': gate['support_gate_reason']}
    fee = self_financing_cost(p, t, rates)['cost_fraction']
    z = fee/(1-fee)
    l0, ld = -(r @ p), -(r @ (t-p))
    candidate_cuts = []
    for i, j in combinations(range(len(r)), 2):
        if ld[i] != ld[j]:
            candidate_cuts.append(float(-(l0[i]-l0[j])/(ld[i]-ld[j])))
    for alpha in ALPHAS:
        cap = limits[str(alpha)]['VaR']
        intercept, slope = l0-cap, ld+z*(1-cap)
        keep = slope != 0
        candidate_cuts.extend((-intercept[keep]/slope[keep]).tolist())

    def evaluate(lam):
        loss = (l0+(ld+z)*lam)/(1+z*lam)
        risk = {str(a): var_es_actual_tail_v1(loss, a, prob, support_owner=support_owner) for a in ALPHAS}
        residual = max(risk[str(a)][m]-limits[str(a)][m] for a in ALPHAS for m in ('VaR', 'ES'))
        # Use machine precision for mathematical boundaries; caller tolerance
        # reports numerical precision and never fills a forbidden interval.
        roundoff = 32*np.finfo(float).eps*max(1., float(np.max(np.abs(loss))))
        support_ok = all(support_owner is None or q['support_gate_status'] == 'PASS' for q in risk.values())
        return bool(residual <= roundoff and support_ok), float(residual), risk

    pieces, tested_cells, boundary_residuals = [], 0, []
    for interval in linear_domain:
        if len(interval) != 2 or not all(np.isfinite(v) for v in interval) or not 0 <= interval[0] <= interval[1] <= 1:
            raise ValueError('REGISTERED_LAMBDA_DOMAIN')
        lo, hi = map(float, interval)
        cuts = sorted(set([lo, hi] + [v for v in candidate_cuts if lo < v < hi]))
        for point in cuts:
            good, residual, _ = evaluate(point)
            if good:
                pieces.append({'lower': point, 'upper': point, 'lower_closed': True, 'upper_closed': True})
                boundary_residuals.append(residual)
        for left, right in zip(cuts[:-1], cuts[1:]):
            if max_cells is not None and tested_cells >= max_cells:
                return {'status': 'PARTIAL_SEARCH', 'reason': 'REGISTERED_CELL_BUDGET_EXHAUSTED',
                        'segments': merge_topological_segments_v1(pieces), 'tested_cells': tested_cells,
                        'complete': False, 'version': VERSION}
            tested_cells += 1
            mid = left+(right-left)/2
            mid_loss = l0+ld*mid
            current = (left, right)
            for alpha in ALPHAS:
                tail = tail_mass_decomposition(mid_loss, prob, alpha)
                q = np.asarray(tail['tail_contribution_mass'])/(1-alpha)
                # Quantile scenario is constant in each strict order cell.
                index = np.flatnonzero((mid_loss == tail['VaR']) & (prob > 0))[0]
                for metric, intercept, slope in (
                    ('VaR', l0[index], ld[index]+z), ('ES', float(q@l0), float(q@ld)+z)):
                    cap = limits[str(alpha)][metric]
                    current = _intersect_affine(*current, intercept-cap, slope-cap*z)
                    if current is None:
                        break
                if current is None:
                    break
            if current is None:
                continue
            a, b = current
            interior = a+(b-a)/2
            if not evaluate(interior)[0]:
                continue
            left_ok, left_res, _ = evaluate(a)
            right_ok, right_res, _ = evaluate(b)
            if a == b and not left_ok:
                continue
            pieces.append({'lower': a, 'upper': b, 'lower_closed': left_ok, 'upper_closed': right_ok})
            boundary_residuals.extend([left_res, right_res])
    segments = merge_topological_segments_v1(pieces)
    return {'status': 'PASS' if segments else 'INFEASIBLE',
            'reason': None if segments else 'NO_FEASIBLE_POINT_ON_REGISTERED_RAY',
            'segments': segments, 'complete': True, 'tested_cells': tested_cells,
            'fee_rational_coefficient': z, 'numerical_tolerance': numerical_tolerance,
            'maximum_boundary_risk_residual': max(boundary_residuals, default=None),
            'certificate': 'AFFINE_ORDER_CELLS_EXPLICIT_ENDPOINT_MEMBERSHIP',
            'scope': 'REGISTERED_ONE_DIMENSIONAL_RAY_ONLY', 'version': VERSION}


def optimize_registered_ray_v2(pre, target, scenarios, *, cap6, permission, regime, cost_schedule,
                             utility_owner, risk_owner, numerical_policy, mass=None, current_wealth=1.,
                             normal_floor_activation_approved=False, support_owner=None, utility_hurdle=0.):
    """Topology-aware challenger utility selection, retaining frozen candidates."""
    if not isinstance(current_wealth,(int,float)) or not np.isfinite(current_wealth) or current_wealth<=0:
        return {'status':'BLOCKED','reason':'FINITE_POSITIVE_CURRENT_WEALTH_REQUIRED'}
    if not isinstance(utility_hurdle,(int,float)) or not np.isfinite(utility_hurdle) or utility_hurdle<0:
        return {'status':'BLOCKED','reason':'FINITE_NONNEGATIVE_UTILITY_HURDLE_REQUIRED'}
    gamma=utility_owner.get('gamma')
    if utility_owner.get('utility')!='LOG' and (not isinstance(gamma,(int,float)) or not np.isfinite(gamma) or gamma<=0):
        return {'status':'BLOCKED','reason':'FINITE_POSITIVE_UTILITY_GAMMA_REQUIRED'}
    if set(risk_owner.get('required_horizons', [1])) != {1}:
        return {'status': 'BLOCKED', 'reason': 'MULTIHORIZON_JOIN_REQUIRES_SEPARATE_REGISTERED_FRAMES'}
    fn = utility_fn(utility_owner)
    if fn is None or risk_owner.get('approved') is not True or not risk_owner.get('owner_hash'):
        return {'status': 'BLOCKED', 'reason': 'UTILITY_RISK_OWNER_REQUIRED'}
    tol = numerical_policy['lambda_tolerance']
    legal = legal_partial_lambda_intervals_v2(pre, target, cap6, permission, regime=regime,
        normal_floor_activation_approved=normal_floor_activation_approved, tolerance=tol)
    if legal['status'] != 'PASS':
        return legal
    rates = cost_rates(cost_schedule)
    frontier = solve_feasible_segments_v2(pre,target,scenarios,mass,rates,risk_owner,legal['segments'],
                                          numerical_tolerance=tol,support_owner=support_owner)
    if frontier['status'] != 'PASS':
        return frontier
    p,t,r = simplex(pre,6),simplex(target,6),np.asarray(scenarios,float)
    prob = np.full(len(r),1/len(r)) if mass is None else np.asarray(mass,float)

    def utility(lam):
        w = (1-lam)*p+lam*t
        cost = self_financing_cost(p,w,rates)
        growth = (1-cost['cost_fraction'])*(1+r@w)
        if np.any(growth<=0):
            return -np.inf
        return float(prob@fn(current_wealth*growth))

    candidates = []
    for cell in frontier['segments']:
        lo,hi = cell['lower'],cell['upper']
        left = lo if cell['lower_closed'] else np.nextafter(lo,hi)
        right = hi if cell['upper_closed'] else np.nextafter(hi,lo)
        points = [left,right,(left+right)/2]
        points += [v for v in (0.,.25,.5,.75,1.) if left<=v<=right]
        if right>left:
            opt=minimize_scalar(lambda v:-utility(v),bounds=(left,right),method='bounded',options={'xatol':tol})
            points.append(float(opt.x))
        for lam in points:
            w=(1-lam)*p+lam*t
            cost=self_financing_cost(p,w,rates)
            growth=(1-cost['cost_fraction'])*(1+r@w)
            risk={str(a):var_es_actual_tail_v1(1-growth,a,prob,support_owner=support_owner) for a in ALPHAS}
            valid=all(q[m]<=risk_owner['limits'][alpha][m]+32*np.finfo(float).eps
                      for alpha,q in risk.items() for m in ('VaR','ES'))
            valid &= all(support_owner is None or q['support_gate_status']=='PASS' for q in risk.values())
            eu=utility(lam)
            if valid and np.isfinite(eu):
                candidates.append({'lambda':float(lam),'weights':w.tolist(),'expected_net_utility':eu,
                                   'risk':risk,**cost})
    if not candidates:
        return {'status':'U','reason':'OPEN_CELL_REPRESENTABLE_OPTIMUM_UNRESOLVED','frontier':frontier}
    best=max(candidates,key=lambda z:(z['expected_net_utility'],-z['cost_fraction']))
    hold=next((z for z in candidates if z['lambda']==0),None)
    if hold is not None and best['expected_net_utility']-hold['expected_net_utility']<=utility_hurdle:
        best=hold
    return {'status':'PASS','best':best,'candidates':candidates,'frontier':frontier,'version':VERSION,
            'cost_schedule_sha':canonical_sha256(cost_schedule),'optimization_claim':'BEST_FOUND_ON_EXPLICIT_FEASIBLE_CELLS',
            'production_promoted':False}
