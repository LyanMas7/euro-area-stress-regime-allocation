#!/usr/bin/env python3
"""W05 RQ03 -- NEW FUNCTIONS ONLY REV10, independent add-on, no existing module overwritten.

User-priority scope:
 (1) five benchmark + POLICY VaR/ES and capital-fraction risk frontier;
     NORMAL = [0.50,0.75], STRESS (D/G/BOTH) = [0.0,0.50];
     fractions are risky CAPITAL allocations, NOT probabilities or ES percentages.
 (2) own-geometry benchmark restructuring with cost-once wealth and per-asset caps;
 (3) explicit bottom risk-squeeze and paired most-recent-SHORT -> first-BOTTOM
     avoided-loss counterfactual with investment arm using the *same benchmark*.

This is an optional additive file for v8_w05_rq03; it DOES NOT replace any
existing W05 code, frozen incumbent, issuer state, or owner policy.

Scientific scopes:
 - Source-locked DEV historical tape yields RETROSPECTIVE_RESEARCH_ONLY risk.
 - No RQ01/RQ02 issued signal tape is fabricated from the ESRB ex-post labels.
 - No official NET or production selection when owner signatures, C1 funding,
   B1/B2 investability, threshold, utility, or cost authorization is absent.
 - No DEV fold, OOS observation, date-selected fit or synthetic evidence promotion.
 - REV2 hard gates: risky-share applies to all HOLD/PARTIAL/FULL candidates;
   Bottom flags are NOT sufficient without 3F, risk, funding, eligibility,
   owner lineage and effective-window evidence; frame<=action issue time.
 - REV4: verify all SIX post-trade asset caps including cash; classify owner-risk
   VaR/ES breaches as emergency repair even without geometry drift. Never infer
   full feasibility from risky-sleeve caps alone. Audit matched NET ledger.
 - RESEARCH add-ons: one-/four-week nonoverlapping historical VaR/ES and
   NET risk-budget / exact-cost / owner-utility capital solver.
 - REV7: issued scenario mass is part of the risk-source vintage; exact weighted
   VaR/ES and probability-weighted NET utility are passed into ALL capital/action
   gates. Explicit owner scenario-weight convention and tail-support guards.
 - REV7: capital risk frontier is projected through the asset-specific Bottom
   buy/no-buy permissions BEFORE tail evaluation. Only cap-tightening is legal;
   forbidden risky allocation is returned to C1, never redistributed secretly.
 - REV8: enforce the EXECUTED (post-projection) risky capital lower/upper
   bounds, reject grid artifacts violating Normal >=50%, and require the
   same approved cost schedule for capital optimization and live action.
 - REV8: threshold research grid runs all 5 benchmark-specific geometries
   against canonical DEV returns; POLICY/issued risk remains BLOCKED.
 - REV8: trade threshold numerical domains must be finite with explicit
   units; partial rebalancing requires strict improvement and consent.
 - REV7: finite 5pp/2.5pp capital-grid regret diagnostic is research-only;
   six strategy model-specific geometry retains original reference unless an
   explicit owner selects permission-projected tracking (not a new optimum).

REV7 issued-asof extension schema (untrusted packets are NOT authenticated):
 - asof.frames[t].scenario_mass: optional S-length decimal probability simplex,
   part of scenario_source_id + scenario_issued_at immutable vintage. Absence
   means mathematical uniform distribution, permitted only with explicit owner.
 - owners.scenario_probability_owner: {approved,owner_hash,mode} where mode is
   UNIFORM_ISSUED or WEIGHTED_ISSUED and must match every relevant frame.
 - owners.capital_objective_owner.capital_grid_step: optional approved positive
   fraction step (default 0.05 for backward-compatible research).
 - models[m].tolerance_owner.threshold_reference: ORIGINAL_BENCHMARK (default)
   or PERMISSION_PROJECTED_TARGET; the latter requires owner field
   reference_geometry_contract=PROJECTED_TARGET_NOT_CONSTRAINED_OPTIMUM.
 - risk_owner.min_effective_tail_support applies to both capital selection and
   EVERY post-trade action, not only an ex-ante frontier estimate.
 - REV6: explicit owner objective gate; immutable Short ID/issued-time lineage, EXIT explicitly bound to
   active Short, complete right-censor/supersession/EXIT-before-Bottom ledger,
   separately audited risk basis and weekly Bottom squeeze permissions.
 - REV3: enforce EWI/Country/Risk-Cov and risk-check evidence known AT bottom
   permission issuance, not merely at later action time; date-only permission
   windows are inclusive UTC civil dates; mathematically exact Short->Bottom
   avoided-loss attribution (market difference vs self-financing cost drag).

REV10 adds all-start candidate native-week freeze versus own-benchmark
   restructuring with independent oracle, using real DEV returns, and marks
   overlapping windows NOT historical issued Short/Bottom events.

R23 (opt-in, config['r23_main_only_lock'] is True; default OFF keeps the audited all-strategy freeze):
   the Short->Bottom restructuring lock belongs to POLICY (Main) only.  POLICY is NO_TRADE_LOCKED while the
   issued zone is NO_RISKY_ZONE (passive drift, zero turnover/cost, risk breaches ESCALATED not repaired);
   EW/IVOL/MVP/ERC/MaxDiv are exempt: regime-only zone, no forced cash liquidation, no Bottom overlay.
   The research counterfactual in short_to_bottom_counterfactual (C1 cash-freeze arm vs invested arm) is
   unchanged: it compares hypotheticals, it is not the Main policy.

REV9: Normal 50% is a post-action HARD lower bound and requires explicit
   normal_floor_activation_approved owner field when pre is below 50%.
 - REV9: finite, source-locked DEV VaR/ES budget frontier and 400 x 4
   joint geometry/capital/cost/risk veto test; only retrospective inference,
   no production threshold selection or owner record fabrication.

Integration: copy this file beside benchmarks.py, tail.py, trading.py and run:
  python -m v8_w05_rq03.W05_RQ03_PRIORITY_END_TO_END_AUDITED_NEW_FUNCTIONS_REV10_20261023 \
      --dataset-root /path/to/Euro_Area_Dataset_Mentor_Ready --mode risk
  python -m v8_w05_rq03.W05_RQ03_PRIORITY_END_TO_END_AUDITED_NEW_FUNCTIONS_REV10_20261023 \
      --dataset-root /path/to/Euro_Area_Dataset_Mentor_Ready --mode replay \
      --issued-json /path/to/owner_issued_records.json \
      --owner-json /path/to/owner_execution.json
  PYTHONPATH=/path/REV333/src python W05_RQ03_PRIORITY_END_TO_END_AUDITED_NEW_FUNCTIONS_REV10_20261023.py --self-test
  PYTHONPATH=/path/REV333/src python W05_RQ03_PRIORITY_END_TO_END_AUDITED_NEW_FUNCTIONS_REV10_20261023.py \
    --dataset-root /path/to/Euro_Area_Dataset_Mentor_Ready --mode horizon

CLI writes JSON to STDOUT; never edits source data / checkpoint / OOS.
Python >=3.10, numpy; existing v8_w05_rq03 package installed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

# Import existing W05 mathematics; never clone or overwrite the incumbent solvers.
if __package__:
    from . import benchmarks as B
    from . import tail as T
    from . import trading as TR
    from . import regime_policy as RP
    from .contract import (ASSETS, METHODS, ALPHAS, REGIMES,
                           canonical_sha256, stamp, owner_ok, covariance as _covariance)
else:
    # Also supports direct execution as a separate raw code file when REV333/src
    # (or the package parent folder) is on PYTHONPATH.
    from v8_w05_rq03 import benchmarks as B
    from v8_w05_rq03 import tail as T
    from v8_w05_rq03 import trading as TR
    from v8_w05_rq03 import regime_policy as RP
    from v8_w05_rq03.contract import (ASSETS, METHODS, ALPHAS, REGIMES,
                                     canonical_sha256, stamp, owner_ok,
                                     covariance as _covariance)

ALL_METHODS = ('EW', 'IVOL', 'MVP', 'ERC', 'MaxDiv', 'POLICY')
STRESS = ('D', 'G', 'BOTH')
TAPE_REL = Path('03_ASSET_DATA/05_CANONICAL_MAIN_INPUT/V8_RQ03_SIX_ASSET_DEV_WEEKLY.csv')
LOCK_REL = Path('03_ASSET_DATA/05_CANONICAL_MAIN_INPUT/V8_RQ03_SOURCE_LOCK.json')
RESEARCH = 'RETROSPECTIVE_DEV_RESEARCH_ONLY_NOT_ISSUED_NOT_OFFICIAL_NET'


def outcome(status: str, reason: str | None = None, **fields) -> dict:
    if status not in ('PASS', 'BLOCKED', 'NOT_EVALUATED', 'U', 'FAIL'):
        raise ValueError('BAD_STATUS')
    return {'status': status, 'reason': reason, **fields}


def _arr(x, n=None):
    a = np.asarray(x, dtype=float)
    if a.ndim != 1 or (n is not None and len(a) != n) or not np.isfinite(a).all():
        raise ValueError('INVALID_ARRAY')
    return a


def _simplex(x, n=6):
    a = _arr(x, n)
    if a.min() < -1e-10 or abs(a.sum() - 1.0) > 1e-8:
        raise ValueError('INVALID_SIMPLEX')
    return np.maximum(0., a) / np.maximum(0., a).sum()


def _risk_regime(regime: str) -> str:
    if regime not in REGIMES:
        raise ValueError('STATE_MUST_BE_N_D_G_BOTH')
    return 'NORMAL' if regime == 'N' else 'STRESS'


def capital_domain(regime: str, step: float = 0.05) -> np.ndarray:
    """Normal [50%,75%], Stress <=50%; domain candidates, NOT selected owners."""
    if not isinstance(step,(int,float)) or isinstance(step,bool) or not np.isfinite(step) or step <= 0 or step > 0.5:
        raise ValueError('INVALID_CAPITAL_STEP')
    low, high = (0.5, 0.75) if regime == 'N' else (0.0, 0.5) if regime in STRESS else (None, None)
    if low is None:
        raise ValueError('UNKNOWN_REGIME')
    vals = [low + step * i for i in range(int((high - low) / step) + 1)]
    return np.array(sorted(set([round(min(high, v), 12) for v in vals] + [high])), dtype=float)


def _asof_record(rec: dict, issue_at: str, required: tuple[str, ...] = ()) -> bool:
    if not isinstance(rec, dict) or not all(k in rec for k in required):
        return False
    if not rec.get('issued_at') or not rec.get('source_id'):
        return False
    try:
        return stamp(rec['issued_at']) <= stamp(issue_at)
    except (TypeError, ValueError):
        return False


def load_locked_dev(dataset_root: str | Path) -> dict:
    """Read EXACT canonical DEV weekly tape; verify raw file SHA and declared chronology."""
    root = Path(dataset_root)
    options = [root, root/'data'/'Euro_Area_Dataset_Mentor_Ready', root/'Euro_Area_Dataset_Mentor_Ready']
    found = [p for p in options if (p/TAPE_REL).is_file() and (p/LOCK_REL).is_file()]
    if len(found) != 1:
        return outcome('BLOCKED', 'EXACTLY_ONE_CANONICAL_LOCKED_DEV_ROOT_REQUIRED', candidate_count=len(found))
    root = found[0]
    raw = (root/TAPE_REL).read_bytes()
    lock = json.loads((root/LOCK_REL).read_text(encoding='utf-8-sig'))
    digest = hashlib.sha256(raw).hexdigest()
    if digest != lock.get('sha256_tape') or lock.get('oos') != 'SEALED':
        return outcome('FAIL', 'CANONICAL_TAPE_SHA_OR_OOS_LOCK_MISMATCH')
    if lock.get('status') != 'DEV_RETROSPECTIVE_PROXY_ONLY_NOT_PIT_NOT_TRADABLE':
        return outcome('FAIL', 'UNEXPECTED_TAPE_STATUS')
    import io
    r = csv.DictReader(io.StringIO(raw.decode('utf-8-sig')))
    if tuple(r.fieldnames or ()) != ('week_start', 'week_end', *ASSETS):
        return outcome('FAIL', 'SIX_ASSET_TAPE_COLUMNS_INVALID')
    rows = list(r)
    if len(rows) != int(lock['n_weekly_rows']):
        return outcome('FAIL', 'ROW_COUNT_MISMATCH')
    starts, ends, X = [], [], []
    for row in rows:
        s, e = date.fromisoformat(row['week_start']), date.fromisoformat(row['week_end'])
        if not (date(2005, 1, 1) <= s <= e <= date(2019, 12, 31)):
            return outcome('FAIL', 'DEV_DATE_RANGE_BREACH')
        if ends and (s <= ends[-1] or e <= ends[-1]):
            return outcome('FAIL', 'NONMONOTONE_OR_OVERLAPPING_RETURN_WINDOWS')
        starts.append(s)
        ends.append(e)
        X.append([float(row[a]) for a in ASSETS])
    x = np.asarray(X, dtype=float)
    if not np.isfinite(x).all() or np.any(x <= -1):
        return outcome('FAIL', 'INVALID_WEEKLY_RETURN')
    if ends[0].isoformat() != lock['first_week_end'] or ends[-1].isoformat() != lock['last_week_end']:
        return outcome('FAIL', 'WEEK_END_BOUNDARY_MISMATCH')
    return outcome('PASS', starts=starts, ends=ends, X=x, tape_sha=digest,
                   n=len(x), lock=lock, scope=RESEARCH)


def benchmark_vectors(cov5, *, policy_shape: dict | None = None, at: str | None = None) -> dict:
    """Reuse the five native W05 solvers. POLICY needs its own issued shape."""
    out = {}
    for m in METHODS:
        d = B.benchmark_shape(m, cov5, sigma_estimator='SAMPLE_DEV')
        if d['status'] != 'PASS':
            out[m] = outcome('U', 'BENCHMARK_SOLVER_UNAVAILABLE')
        else:
            out[m] = outcome('PASS', shape=_simplex(d['shape'], 5),
                             geometry=B.GEOMETRY[m], covariance_scope='FULL_DEV_FIT_RESEARCH_ONLY')
    if at and _asof_record(policy_shape, at, ('shape',)):
        out['POLICY'] = outcome('PASS', shape=_simplex(policy_shape['shape'], 5),
                                source_id=policy_shape['source_id'], issued_at=policy_shape['issued_at'])
    else:
        out['POLICY'] = outcome('BLOCKED', 'ISSUED_POLICY_SHAPE_REQUIRED')
    return out


def _portfolio(shape5, a: float):
    v = _simplex(shape5, 5)
    if not np.isfinite(a) or a < 0 or a > 1:
        raise ValueError('CAPITAL_FRACTION_OUTSIDE_0_1')
    return np.r_[v*a, 1.-a]


def _scenario_mass(mass, n: int) -> np.ndarray:
    """Exact discrete probability simplex; no silent renormalisation of invalid mass.

    Uniform is only a local mathematical default. Issued replay separately
    REQUIRES an approved owner for its weight convention and binds the weights
    to the source-vintage hash. Zero-mass scenarios are allowed and ignored by
    the exact atom-splitting ES; at least two strictly positive observations
    are needed for a well-defined risk pool.
    """
    if mass is None:
        return np.full(n, 1. / n, dtype=float)
    p = np.asarray(mass, dtype=float)
    if (p.shape != (n,) or not np.isfinite(p).all() or
            np.any(p < 0) or abs(float(p.sum())-1.)>1e-10 or
            np.count_nonzero(p > 0) < 2):
        raise ValueError('INVALID_ISSUED_SCENARIO_MASS_SIMPLEX')
    return p


def effective_trade_caps(pre_weights6, squeeze: dict, cap_owner: dict) -> dict:
    """Tighten, never relax, issuer-approved caps by current Bottom/freeze permissions.

    In each risky asset: upper_i=min(owner_cap_i, squeeze_cap_i,
    pre_i if purchases are forbidden). The resulting feasible simplex is an
    owner-cap intersection, NOT a fabricated new approval. A short freeze may
    still permit a sell toward cash. Missing cash funding or caps fail closed.
    This supports model-specific projection WITHOUT covert risky redistribution.
    """
    if not owner_ok(cap_owner) or cap_owner.get('unit')!='FRACTION_OF_NAV':
        return outcome('BLOCKED','GLOBAL_OWNER_ASSET_CAPS_REQUIRED')
    caps=cap_owner.get('caps')
    if not isinstance(caps,dict) or set(caps)!=set(ASSETS):
        return outcome('BLOCKED','SIX_GLOBAL_CAPS_REQUIRED')
    if not isinstance(squeeze,dict) or squeeze.get('status')!='PASS':
        return outcome('BLOCKED','ISSUED_SQUEEZE_REQUIRED')
    try:
        pre=_simplex(pre_weights6)
        owner=np.asarray([caps[a] for a in ASSETS],float)
        if not np.isfinite(owner).all() or (owner<0).any() or (owner>1).any():
            raise ValueError('OWNER_CAP_DOMAIN')
        eff={}
        for i,a in enumerate(ASSETS[:5]):
            upper=float(squeeze['asset_caps'][a])
            permission=squeeze['new_buy'][a]
            if not isinstance(permission,bool) or not np.isfinite(upper) or not 0<=upper<=1:
                raise ValueError('SQUEEZE_CAP_OR_PERMISSION_DOMAIN')
            eff[a]=float(min(owner[i],upper,1. if permission else pre[i]))
        eff['C1']=float(owner[5])
    except (KeyError,ValueError,TypeError) as exc:
        return outcome('BLOCKED','CAP_INTERSECTION_INVALID',detail=str(exc))
    # A valid simplex requires a total combined upper bound >= 1.
    if sum(eff.values()) < 1.-1e-10:
        return outcome('BLOCKED','SQUEEZE_CASH_COMBINED_CAP_INFEASIBLE',
                       total_upper=float(sum(eff.values())))
    return outcome('PASS',caps=eff,owner_hash=cap_owner['owner_hash'],
                   cash_cap=eff['C1'],
                   buy_allowed={a:squeeze['new_buy'][a] for a in ASSETS[:5]},
                   max_risky_capacity=float(min(1.,sum(eff[a] for a in ASSETS[:5]))),
                   scope='DETERMINISTIC_TIGHTENING_OF_VERIFIED_OWNER_CAPS')


def risk_frontier(scenarios6, shapes: dict, regime: str,
                  *, alphas=ALPHAS, step: float = .05, mass=None,
                  projected_caps5: dict | None = None) -> dict:
    """VaR/ES across candidate risky allocations; no regime labels are invented."""
    r = np.asarray(scenarios6, dtype=float)
    if r.ndim != 2 or r.shape[1] != 6 or len(r) < 2 or not np.isfinite(r).all():
        raise ValueError('SIX_ASSET_SCENARIO_MATRIX_REQUIRED')
    probs = _scenario_mass(mass, len(r))
    results = {}
    for m in ALL_METHODS:
        sh = shapes.get(m, {'status': 'BLOCKED', 'reason': 'MISSING_SHAPE'})
        if sh['status'] != 'PASS':
            results[m] = outcome('BLOCKED', sh.get('reason', 'SHAPE_UNAVAILABLE'))
            continue
        entries = []
        for a in capital_domain(regime, step):
            w = _portfolio(sh['shape'], float(a))
            if projected_caps5 is not None:
                if set(projected_caps5)!=set(ASSETS[:5]):
                    raise ValueError('FIVE_RISKY_PROJECTION_CAPS_REQUIRED')
                w = _asset_projection(w,projected_caps5)
            executed_share=float(w[:5].sum())
            if regime == 'N' and executed_share < 0.5 - 1e-10:
                continue  # A permission-projected 50% target is NOT 50% deployable capital.
            if executed_share > (0.75 if regime == 'N' else 0.50) + 1e-10:
                continue
            metrics = T.portfolio_tail(r, w, alphas=alphas, mass=probs)
            entries.append({'a': float(a), 'executed_risky_fraction':float(w[:5].sum()),
                            'weights': w.tolist(),
                            'risk': {str(alpha): {'VaR': float(v['VaR']), 'ES': float(v['ES']),
                                                  'effective_tail_support': float(v['effective_tail_support'])}
                                     for alpha, v in metrics.items()}})
        results[m] = outcome('PASS' if entries else 'BLOCKED',
                             None if entries else 'NO_EXECUTABLE_CAPITAL_GRID_UNDER_PERMISSIONS',
                             regime=regime, frontier=entries,
                             capital_unit='FRACTION_OF_NAV_IN_FIVE_RISKY_ASSETS',
                             claim=RESEARCH)
    return outcome('PASS', models=results, regime=regime, scenario_count=len(r),
                   scenario_effective_count=float(1./np.sum(probs**2)), claim=RESEARCH)


def choose_capital_by_risk(frontier: dict, model: str, regime: str,
                           limits_owner: dict | None, *, required_alphas=ALPHAS,
                           asset_cap_owner: dict | None = None) -> dict:
    """Select highest feasible allocation, NOT a return/utility optimum.

    limits_owner schema: {approved:true, owner_hash:'..',
      caps: {model: {regime: {'0.95':{'VaR':decimal,'ES':decimal}, ...}}}}
    Caps are losses per unit NAV (e.g. .03 means 3%), not percent strings.
    """
    if not owner_ok(limits_owner):
        return outcome('BLOCKED', 'OWNER_APPROVED_MODEL_REGIME_VAR_ES_BUDGET_REQUIRED')
    if limits_owner.get('risk_unit') != 'DECIMAL_LOSS_FRACTION_OF_NAV':
        return outcome('BLOCKED','VAR_ES_OWNER_DECIMAL_NAV_LOSS_UNIT_REQUIRED')
    if frontier.get('regime') != regime:
        return outcome('BLOCKED', 'RISK_FRONTIER_REGIME_MISMATCH', requested=regime,
                       supplied=frontier.get('regime'))
    data = frontier.get('models', {}).get(model)
    if not data or data['status'] != 'PASS':
        return outcome('BLOCKED', 'BENCHMARK_RISK_FRONTIER_UNAVAILABLE')
    lim = limits_owner.get('caps', {}).get(model, {}).get(regime)
    if not isinstance(lim, dict):
        return outcome('BLOCKED', 'MODEL_REGIME_RISK_LIMITS_MISSING')
    if not owner_ok(asset_cap_owner):
        return outcome('BLOCKED', 'OWNER_ASSET_CAPS_REQUIRED')
    if asset_cap_owner.get('unit') != 'FRACTION_OF_NAV':
        return outcome('BLOCKED','OWNER_ASSET_CAP_FRACTION_UNIT_REQUIRED')
    caps = asset_cap_owner.get('caps', {})
    if set(caps) != set(ASSETS) or any(not isinstance(caps[k], (int, float)) or not 0 <= caps[k] <= 1 for k in ASSETS):
        return outcome('BLOCKED', 'SIX_VALID_ASSET_CAPS_REQUIRED')
    for alpha in required_alphas:
        q = lim.get(str(alpha))
        if not isinstance(q, dict) or any(k not in q or not isinstance(q[k], (int, float)) or
                                          not np.isfinite(q[k]) or q[k] < 0 for k in ('VaR', 'ES')):
            return outcome('BLOCKED', 'VAR_ES_ALPHA_LIMITS_MISSING_OR_INVALID', alpha=alpha)
    minimum_tail_support = limits_owner.get('min_effective_tail_support')
    if minimum_tail_support is not None and (not isinstance(minimum_tail_support, (int,float))
            or not np.isfinite(minimum_tail_support) or minimum_tail_support <= 0):
        return outcome('BLOCKED', 'OWNER_MIN_TAIL_SUPPORT_INVALID')
    feasible = []
    for row in data['frontier']:
        if minimum_tail_support is not None and any(
            row['risk'][str(alpha)]['effective_tail_support'] < minimum_tail_support
            for alpha in required_alphas):
            continue
        if any(w > caps[a] + 1e-10 for a, w in zip(ASSETS, row['weights'])):
            continue
        if all(row['risk'][str(alpha)]['VaR'] <= lim[str(alpha)]['VaR']+1e-12 and
               row['risk'][str(alpha)]['ES'] <= lim[str(alpha)]['ES']+1e-12 for alpha in required_alphas):
            feasible.append(row)
    if not feasible:
        return outcome('BLOCKED', 'NO_FEASIBLE_NORMAL_LOWER_BOUND_OR_STRESS_CANDIDATE',
                       model=model, regime=regime, risk_owner_hash=limits_owner['owner_hash'])
    chosen = max(feasible, key=lambda z: z['a'])
    return outcome('PASS', model=model, regime=regime, risky_share=chosen['a'],
                   risk=chosen['risk'], weights=chosen['weights'],
                   feasible_executed_risky_fraction=chosen['executed_risky_fraction'],
                   objective='MAX_FEASIBLE_RISKY_SHARE_NOT_NET_UTILITY_OPTIMUM',
                   risk_owner_hash=limits_owner['owner_hash'], cap_owner_hash=asset_cap_owner['owner_hash'])



def net_capital_utility_choice(scenarios6, shape5, pre_weights6, model: str, regime: str,
                               *, nav: float, risk_owner: dict | None,
                               asset_cap_owner: dict | None,
                               cost_owner: dict | None,
                               utility_owner: dict | None,
                               step: float = 0.05,
                               alphas=ALPHAS, mass=None,
                               projected_caps5: dict | None = None) -> dict:
    """Cost-once NET risk/utility capital solver with explicit owner gates.

    Selects best risk-feasible candidate in Normal 50%-75% or Stress <=50%.
    Scenario outcomes are as-of forecasts supplied by caller, NOT fitted here;
    costs use exact self-financing equation. Risk budgets apply to NET loss
    per NAV (not to gross sleeve VaR), under owner's explicit `risk_basis=NET`.
    Unlike `choose_capital_by_risk`, objective is owner utility after costs.
    Neither optimizer is permission to buy in NO_RISKY_ZONE or at Bottom;
    benchmark_restructure retains those independent hard gates.
    """
    if model not in ALL_METHODS or regime not in REGIMES:
        raise ValueError('UNKNOWN_METHOD_OR_REGIME')
    if not all(owner_ok(o) for o in (risk_owner,asset_cap_owner,cost_owner,utility_owner)):
        return outcome('BLOCKED','NET_RISK_CAP_COST_UTILITY_OWNER_REQUIRED')
    if (risk_owner.get('risk_unit') != 'DECIMAL_LOSS_FRACTION_OF_NAV'
            or asset_cap_owner.get('unit') != 'FRACTION_OF_NAV'
            or cost_owner.get('unit') != 'PROPORTIONAL_TRADE_VALUE'):
        return outcome('BLOCKED','NET_RISK_CAP_COST_OWNER_UNIT_CONTRACT_UNVERIFIED')
    if risk_owner.get('risk_basis')!='NET_AFTER_SELF_FINANCING_COST':
        return outcome('BLOCKED','NET_RISK_BUDGET_SEMANTICS_OWNER_REQUIRED')
    if not np.isfinite(nav) or nav<=0:
        return outcome('BLOCKED','NAV_REQUIRED')
    risklimits=(risk_owner.get('caps') or {}).get(model,{}).get(regime)
    if not isinstance(risklimits,dict):
        return outcome('BLOCKED','MODEL_REGIME_NET_VAR_ES_LIMIT_MISSING')
    assetcaps=asset_cap_owner.get('caps') or {}
    costrates=cost_owner.get('rates') or {}
    if set(assetcaps)!=set(ASSETS) or set(costrates)!=set(ASSETS):
        return outcome('BLOCKED','SIX_ASSET_CAP_COST_SCHEDULE_REQUIRED')
    try:
        caps=np.asarray([assetcaps[a] for a in ASSETS],float)
        cost=np.asarray([costrates[a] for a in ASSETS],float)
        if not (np.isfinite(caps).all() and (caps>=0).all() and (caps<=1).all()
                and np.isfinite(cost).all() and (cost>=0).all() and (cost<1).all()):
            raise ValueError('CAP_COST_DOMAIN')
        pre=_simplex(pre_weights6)
        shape=_simplex(shape5,5)
        sc=np.asarray(scenarios6,float)
        if sc.ndim!=2 or sc.shape[1]!=6 or len(sc)<2 or not np.isfinite(sc).all() or (sc<=-1).any():
            raise ValueError('INVALID_ASOF_SCENARIO_MATRIX')
        probs=_scenario_mass(mass,len(sc))
    except (ValueError,TypeError,KeyError) as exc:
        return outcome('BLOCKED','NET_RISK_INPUT_DOMAIN', detail=str(exc))
    min_tail=risk_owner.get('min_effective_tail_support',None)
    if min_tail is not None and (not isinstance(min_tail,(int,float)) or isinstance(min_tail,bool)
                                 or not np.isfinite(min_tail) or min_tail<=0):
        return outcome('BLOCKED','MIN_TAIL_SUPPORT_OWNER_INVALID')
    for alpha in alphas:
        lim=risklimits.get(str(alpha))
        if (not isinstance(lim,dict) or any(x not in lim for x in ('VaR','ES'))
                or not all(isinstance(lim[x],(int,float)) and np.isfinite(lim[x])
                           and lim[x]>=0 for x in ('VaR','ES'))):
            return outcome('BLOCKED','NET_RISK_ALPHA_LIMITS_INCOMPLETE',alpha=alpha)
    result=[]
    for a in capital_domain(regime,step):
        w=_portfolio(shape,float(a))
        if projected_caps5 is not None:
            if set(projected_caps5)!=set(ASSETS[:5]):
                return outcome('BLOCKED','FIVE_RISKY_PROJECTION_CAPS_REQUIRED')
            w=_asset_projection(w,projected_caps5)
        executed_share=float(w[:5].sum())
        if regime == 'N' and executed_share < 0.5-1e-10:
            continue
        if executed_share > (0.75 if regime == 'N' else 0.50)+1e-10:
            continue
        if (w>caps+1e-10).any():
            continue
        ch=TR.self_financing_cost(pre,w,cost)
        c=float(ch['cost_fraction'])
        net_return=(1-c)*(1+sc @ w)-1
        metrics={float(q):T.var_es(-net_return,float(q),probs) for q in alphas}
        if min_tail is not None and any(x['effective_tail_support']<min_tail for x in metrics.values()):
            continue
        if any(v['VaR']>risklimits[str(q)]['VaR']+1e-12 or
               v['ES']>risklimits[str(q)]['ES']+1e-12 for q,v in metrics.items()):
            continue
        try:
            util=_utility_scores(w,sc,nav*(1-c),utility_owner,mass=probs)
        except (ValueError,OverflowError) as ex:
            return outcome('BLOCKED','NET_UTILITY_DOMAIN',detail=str(ex))
        if not np.isfinite(util):
            return outcome('BLOCKED','NONFINITE_NET_UTILITY')
        result.append({'a':float(a),'executed_risky_fraction':float(w[:5].sum()),
                       'weights':w.tolist(),
                       'risk':{str(k):{'VaR':v['VaR'],'ES':v['ES'],
                                       'effective_tail_support':v['effective_tail_support']} for k,v in metrics.items()},
                       'cost_fraction':c,'turnover':float(ch['turnover']),
                       'expected_net_utility':float(util),
                       'expected_net_return':float(probs @ net_return)})
    if not result:
        return outcome('BLOCKED','NO_FEASIBLE_NET_RISK_UTILITY_CAPITAL_ALLOCATION',
                       model=model,regime=regime)
    selected=max(result,key=lambda d:(d['expected_net_utility'],-d['cost_fraction']))
    return outcome('PASS',model=model,regime=regime,risky_share=selected['a'],
                   selected=selected,feasible_frontier=result,
                   claim='CONDITIONAL_TO_ISSUED_SCENARIOS_AND_VERIFIED_EXTERNAL_OWNERS',
                   objective='MAX_OWNER_EXPECTED_UTILITY_SUBJECT_TO_NET_VAR_ES_AND_CAPS')


def capital_grid_refinement_audit(scenarios6, shape5, pre_weights6,
                                  model: str, regime: str, *, nav: float,
                                  risk_owner: dict | None,
                                  asset_cap_owner: dict | None,
                                  cost_owner: dict | None,
                                  utility_owner: dict | None,
                                  scenario_mass=None,
                                  projected_caps5: dict | None = None,
                                  coarse_step: float = 0.05,
                                  fine_step: float = 0.025,
                                  pre_registered_utility_gap: float | None = None) -> dict:
    """Finite, explicit threshold-grid diagnostic; NO in-sample promotion.

    H0: the fine admissible capital grid has no material expected NET utility
    gain over the coarse grid under identical externally issued scenarios.
    H1: coarse grid incurs material discretisation regret.
    Pre-registered falsifier: gain > owner/research gap; do not infer
    independent validation from the same DEV scenario pool.

    The threshold step is in risky NAV FRACTIONS (e.g. 0.025 = 2.5pp).
    It is never a structural EW/MVP/ERC/MaxDiv geometry tolerance and never a
    VaR/ES budget. Each candidate is re-evaluated with the SAME self-financing
    cost, scenario mass, owner limits, and permission-projected weights.
    """
    if (not isinstance(coarse_step,(int,float)) or
            not isinstance(fine_step,(int,float)) or
            isinstance(coarse_step,bool) or isinstance(fine_step,bool) or
            not np.isfinite(coarse_step) or not np.isfinite(fine_step) or
            not 0<fine_step<coarse_step<=0.5):
        return outcome('BLOCKED','PREREGISTERED_FINITE_REFINEMENT_STEPS_REQUIRED')
    if (pre_registered_utility_gap is not None and (
            not isinstance(pre_registered_utility_gap,(float,int)) or
            not np.isfinite(pre_registered_utility_gap) or
            pre_registered_utility_gap<0)):
        return outcome('BLOCKED','PREREGISTERED_UTILITY_GAP_INVALID')
    grids={}
    for label,step in [('coarse',float(coarse_step)),('fine',float(fine_step))]:
        r=net_capital_utility_choice(scenarios6,shape5,pre_weights6,model,regime,
                                     nav=nav,risk_owner=risk_owner,
                                     asset_cap_owner=asset_cap_owner,
                                     cost_owner=cost_owner,
                                     utility_owner=utility_owner,
                                     step=step,mass=scenario_mass,
                                     projected_caps5=projected_caps5)
        if r['status']!='PASS':
            return outcome('NOT_EVALUATED','GRID_COMPARISON_NOT_FEASIBLE',
                           first_failed_grid=label,reason_detail=r.get('reason'))
        grids[label]=r
    coarse=grids['coarse']['selected']
    fine=grids['fine']['selected']
    gain=float(fine['expected_net_utility']-coarse['expected_net_utility'])
    coarse_fractions={round(a,10) for a in capital_domain(regime,coarse_step)}
    fine_fractions={round(a,10) for a in capital_domain(regime,fine_step)}
    nested=coarse_fractions<=fine_fractions
    if nested and gain < -1e-10:
        return outcome('FAIL','REFINED_NESTED_GRID_WORSE_THAN_COARSE')
    return outcome('PASS',model=model,regime=regime,
                   coarse_step=float(coarse_step),fine_step=float(fine_step),
                   grids_nested=nested,
                   coarse_requested_share=coarse['a'],
                   fine_requested_share=fine['a'],
                   coarse_projected_risky_share=coarse['executed_risky_fraction'],
                   fine_projected_risky_share=fine['executed_risky_fraction'],
                   coarse_feasible_candidates=len(grids['coarse']['feasible_frontier']),
                   fine_feasible_candidates=len(grids['fine']['feasible_frontier']),
                   coarse_expected_net_utility=coarse['expected_net_utility'],
                   fine_expected_net_utility=fine['expected_net_utility'],
                   fine_minus_coarse_utility=gain,
                   materiality=('NOT_IDENTIFIABLE_NO_PREREGISTERED_GAP'
                                if pre_registered_utility_gap is None else
                                'MATERIAL_IN_SAMPLE' if gain>pre_registered_utility_gap else
                                'NOT_MATERIAL_IN_SAMPLE'),
                   claim='SAME_SCENARIO_IN_SAMPLE_SENSITIVITY_NO_THRESHOLD_PROMOTION')


def research_multiweek_risk(panel: dict, *, horizons=(1,4), alphas=ALPHAS,
                            policy_shape: dict | None = None) -> dict:
    """Native 1/4-week empirical horizon diagnostics, without sqrt-time shortcut.

    Separate NONOVERLAPPING joint paths for each horizon; no future input is
    allowed in issued/trading logic. This is a retrospective full-DEV study;
    no four-regime conditional forecast is inferred from missing D/G labels.
    All weights held through H weeks, as in incumbent tail.horizon_path_risk.
    """
    if panel.get('status')!='PASS':
        return outcome('BLOCKED','LOCKED_DEV_PANEL_REQUIRED')
    r=np.asarray(panel['X'],float)
    if r.ndim!=2 or r.shape[1]!=6 or len(r)<4 or not np.isfinite(r).all():
        return outcome('BLOCKED','NATIVE_SIX_ASSET_WEEKLY_RETURNS_REQUIRED')
    s=np.cov(r[:,:5],rowvar=False)
    sh=benchmark_vectors(s,policy_shape=policy_shape)
    by_method={}
    for m in ALL_METHODS:
        if sh[m]['status']!='PASS':
            by_method[m]=outcome('BLOCKED',sh[m].get('reason','SHAPE_UNAVAILABLE'))
            continue
        # Risk of the pure risky sleeve. Capital-domain risk remains a separate
        # call; never imply full investment is Normal/Stress-approved.
        w=np.r_[_simplex(sh[m]['shape'],5),0.]
        h_data={}
        for h in horizons:
            if not isinstance(h,int) or h<1 or len(r)//h<2:
                return outcome('BLOCKED','HORIZON_WITH_TOO_FEW_NONOVERLAP_PATHS',horizon=h)
            n=len(r)//h
            paths=r[:n*h].reshape(n,h,6)
            cells={}
            for alpha in alphas:
                met=T.horizon_path_risk(paths,w,float(alpha))
                cells[str(alpha)]={'VaR':float(met['horizon_VaR']),
                                   'ES':float(met['horizon_ES']),
                                   'effective_tail_support':float(met['effective_tail_support'])}
            h_data[str(h)]={'native_paths':n,'path_length_weeks':h,'tail':cells,
                            'sampling':'NONOVERLAPPING_FROM_FIRST_DEV_WEEK',
                            'static_weights':True}
        by_method[m]=outcome('PASS',horizons=h_data)
    return outcome('PASS',models=by_method,weekly_rows=len(r),
                   issue_valid=False,regime_conditional=False,
                   OOS='SEALED',risk_scope='FULL_DEV_RETROSPECTIVE_NONISSUED')

def _bottom_permission_proof(asset: str, perm: dict, decision_at: str | None,
                             active_risk_cap_hash: str | None,
                             trade_at: str | None = None) -> bool:
    """Re-verify evidence, not a self-attested three_factor_structurally_validated bool.

    Source-clock checks are structural, never issuer cryptographic authentication.
    A permission requires issuer/committee/risk/funding/eligibility and a bounded
    effective window; no boolean sentinel independently grants trade authority.
    """
    if not (isinstance(perm,dict) and perm.get('authorized') is True
            and perm.get('asset') == asset and decision_at and active_risk_cap_hash
            and perm.get('risk_caps_owner_hash') == active_risk_cap_hash
            and _asof_record(perm,decision_at,('committee_id','zone_start','zone_end'))):
        return False
    try:
        # Trade date-only represents the START of the trading date, not the
        # end-of-day meaning used by contract.stamp for data publication.
        if trade_at is not None and isinstance(trade_at,str) and len(trade_at)==10:
            point=datetime.combine(date.fromisoformat(trade_at),datetime.min.time(),tzinfo=timezone.utc)
        else:
            point=stamp(trade_at or decision_at)
        left=stamp(perm['zone_start'])
        right=stamp(perm['zone_end'])
        # Civil-date permission window is inclusive: [00:00 of start,
        # 23:59:59.999999 of end] UTC. Otherwise use explicit timestamp.
        if isinstance(perm['zone_start'], str) and len(perm['zone_start'])==10:
            left=datetime.combine(date.fromisoformat(perm['zone_start']),datetime.min.time(),tzinfo=timezone.utc)
        if not left <= point <= right:
            return False
        # Evidence that justifies an issued permission cannot be published
        # after that permission was issued (retroactive approval/leakage).
        permission_issue=stamp(perm['issued_at'])
        if permission_issue>stamp(decision_at) or permission_issue>point:
            return False
    except (ValueError,TypeError):
        return False
    funding=perm.get('funding_owner')
    eligibility=perm.get('eligibility_owner')
    policy=perm.get('reentry_policy_owner')
    if not (owner_ok(funding) and owner_ok(eligibility) and owner_ok(policy)
            and asset in eligibility.get('assets', ())):
        return False
    risk_check=perm.get('risk_check')
    if not (isinstance(risk_check,dict) and risk_check.get('status')=='PASS'
            and risk_check.get('within_caps') is True
            and risk_check.get('risk_caps_owner_hash')==active_risk_cap_hash):
        return False
    if not _asof_record(risk_check,decision_at,('within_caps',)):
        return False
    if not _issued_no_later(risk_check['issued_at'],perm['issued_at']):
        return False
    families=('EWI','COUNTRY','RISK_COV')
    factors=perm.get('factor_evidence')
    rule=perm.get('committee_rule_owner')
    if not isinstance(factors,dict) or set(factors)!=set(families) or not owner_ok(rule):
        return False
    if perm.get('committee_rule_owner_hash')!=rule['owner_hash']:
        return False
    votes, inputs={},{}
    for family in families:
        f=factors[family]
        if (not _asof_record(f,decision_at,('vote','input_ids'))
                or not _issued_no_later(f['issued_at'],perm['issued_at'])):
            return False
        if type(f['vote']) is not bool or not isinstance(f['input_ids'],list) or not f['input_ids']:
            return False
        votes[family],inputs[family]=f['vote'],f['input_ids']
    committee=RP.bottom_committee(votes,inputs,rule)
    return committee['status']=='PASS' and committee.get('decision') is True


def bottom_risk_squeeze(regime: str, zone: str, *, base_caps: dict | None,
                         defense: dict | None, allowance: dict | None,
                         permission: dict | None, owner_caps: dict | None,
                         decision_at: str | None = None,
                         trade_at: str | None = None,
                         active_risk_cap_hash: str | None = None,
                         weeks_after_bottom: float = 0.0,
                         decay_tau: float | None = None,
                         weeks_after_exit: float = 0.0,
                         defense_decay_tau: float | None = None) -> dict:
    """Per-asset cap = clip(base - defense + eligible_allowance*exp(-age/tau),0, owner).

    Bottom predictions alone grant NO buy permission. Same for POST_BOTTOM:
    permissions remain asset-specific and must carry issuer provenance.
    Inputs are decimal fractions, not percentages; function never increases
    capital to overcome missing authorization.
    """
    if regime not in REGIMES:
        raise ValueError('INVALID_REGIME')
    if zone not in ('NORMAL', 'STRESS_PRE_CONFIRMATION', 'NO_RISKY_ZONE',
                    'BOTTOM_REENTRY_ZONE', 'POST_BOTTOM_STRESS', 'EXIT', 'EXIT_REGIME_CONFLICT'):
        raise ValueError('INVALID_ZONE')
    if (not all(isinstance(o, dict) and set(o) == set(ASSETS[:5]) for o in
                (base_caps, defense, allowance, owner_caps))
            or not isinstance(permission, dict) or bool(set(permission)-set(ASSETS[:5]))):
        return outcome('BLOCKED', 'FIVE_ASSET_CAP_DEFENSE_ALLOWANCE_AND_VALID_PERMISSION_MAP_REQUIRED')
    if not np.isfinite(weeks_after_bottom) or weeks_after_bottom < 0 or not np.isfinite(weeks_after_exit) or weeks_after_exit < 0:
        return outcome('BLOCKED', 'FINITE_NONNEGATIVE_AGE_REQUIRED')
    if zone == 'POST_BOTTOM_STRESS' and (decay_tau is None or not np.isfinite(decay_tau) or decay_tau <= 0):
        return outcome('BLOCKED', 'POSITIVE_OWNER_ALLOWANCE_DECAY_TAU_REQUIRED')
    if zone == 'EXIT' and defense_decay_tau is not None and (
            not np.isfinite(defense_decay_tau) or defense_decay_tau <= 0):
        return outcome('BLOCKED', 'POSITIVE_OWNER_DEFENSE_DECAY_TAU_REQUIRED')
    caps, restrictions, report = {}, {}, {}
    for asset in ASSETS[:5]:
        b, d, q, maximum = base_caps[asset], defense[asset], allowance[asset], owner_caps[asset]
        if any(not isinstance(z, (float, int)) or not np.isfinite(z) or z < 0 for z in (b,d,q,maximum)) or maximum > 1:
            return outcome('FAIL', 'INVALID_CAP_LEVELS', asset=asset)
        issued_permission = _bottom_permission_proof(asset, permission.get(asset), decision_at,
                                                     active_risk_cap_hash, trade_at)
        if zone in ('BOTTOM_REENTRY_ZONE','POST_BOTTOM_STRESS'):
            entry_allowed = issued_permission
            value = q * (np.exp(-weeks_after_bottom/decay_tau) if zone == 'POST_BOTTOM_STRESS' else 1.0) if entry_allowed else 0.0
        else:
            entry_allowed = zone in ('NORMAL', 'EXIT')
            value = 0.0
        effective_defense = (d * float(np.exp(-weeks_after_exit / defense_decay_tau))
                             if zone == 'EXIT' and defense_decay_tau is not None else d)
        cap = max(0., min(maximum, b - effective_defense + value))
        # Under Short freeze, new buying is forbidden, regardless of base cap.
        can_increase = entry_allowed and zone not in ('NO_RISKY_ZONE','STRESS_PRE_CONFIRMATION','EXIT_REGIME_CONFLICT')
        caps[asset] = cap
        restrictions[asset] = bool(can_increase)
        report[asset] = {'base': b, 'defense': effective_defense, 'allowance_effective': value,
                         'risk_cap': cap, 'new_buy_permission': bool(can_increase)}
    return outcome('PASS', asset_caps=caps, new_buy=restrictions, by_asset=report,
                   formula='clip(base_cap-defense+eligible_bottom_allowance*exp(-age/tau),0,owner_cap)',
                   claim='CONDITIONAL_GIVEN_EXTERNALLY_VERIFIED_PERMISSION_AND_CAPS')


def _asset_projection(target, asset_caps: dict):
    x = _simplex(target)
    y = x.copy()
    for i,a in enumerate(ASSETS[:5]):
        y[i] = min(y[i], asset_caps[a])
    y[5] = 1.0 - float(np.sum(y[:5]))
    return _simplex(y)


def _utility_scores(w, scenario6, nav: float, utility_owner: dict, *, mass=None) -> float:
    if not owner_ok(utility_owner):
        raise ValueError('APPROVED_UTILITY_OWNER_REQUIRED')
    outcome_nav = nav * (1 + scenario6 @ w)
    probs = _scenario_mass(mass, len(outcome_nav))
    if min(outcome_nav) <= 0:
        raise ValueError('NONPOSITIVE_SCENARIO_WEALTH')
    mode = utility_owner.get('mode')
    if mode == 'LOG':
        return float(probs @ np.log(outcome_nav))
    if mode == 'CARA':
        gamma = utility_owner.get('gamma')
        if not isinstance(gamma,(int,float)) or gamma <= 0:
            raise ValueError('CARA_GAMMA_OWNER_REQUIRED')
        return float(probs @ (-np.exp(-gamma * outcome_nav)))
    if mode == 'CRRA':
        gamma = utility_owner.get('gamma')
        if not isinstance(gamma,(int,float)) or gamma <= 0:
            raise ValueError('CRRA_GAMMA_OWNER_REQUIRED')
        return float(probs @ (np.log(outcome_nav) if abs(gamma-1) < 1e-10 else
                           outcome_nav ** (1-gamma) / (1-gamma)))
    raise ValueError('UNKNOWN_OWNER_UTILITY_MODE')


def projected_reference_deviation(model: str, current5, target5, covariance5) -> dict:
    """Own-unit deviation from a permission-projected, executable risky target.

    Unlike ERC=equal 1/5 RC or unconstrained MVP optimum, this is a TRACKING
    deviation to the approved *projected reference*, NOT a claim that the
    restricted portfolio itself solves the original unconstrained optimisation.
    That separation prevents threshold churn when Bottom 3F only permits E1.
    """
    if model not in METHODS:
        return outcome('BLOCKED','FIVE_BENCHMARK_GEOMETRY_REQUIRED')
    try:
        p=_simplex(current5,5);t=_simplex(target5,5)
        s=_covariance(covariance5,5,positive=True)
        if model in ('EW','IVOL'):
            d=float(np.sum(np.abs(p-t)))
        elif model=='MVP':
            d=float(max(0.,p@s@p-t@s@t))
        elif model=='ERC':
            rc_p=B.risk_contributions(p,s);rc_t=B.risk_contributions(t,s)
            if min(rc_p.min(),rc_t.min()) < -1e-12:
                return outcome('NOT_EVALUATED','NEGATIVE_RISK_CONTRIBUTION_IN_PROJECTED_ERC')
            d=float(np.max(np.abs(rc_p-rc_t)))
        else:
            d=float(max(0.,B.diversification_ratio(t,s)-B.diversification_ratio(p,s)))
    except (ValueError,RuntimeError) as ex:
        return outcome('NOT_EVALUATED','PROJECTED_GEOMETRY_NOT_IDENTIFIABLE',detail=str(ex))
    return outcome('PASS',model=model,geometry=B.GEOMETRY[model],deviation=d,
                   interpretation='REFERENCE_TRACKING_IN_OWN_GEOMETRY_NOT_CONSTRAINED_OPTIMALITY')


def benchmark_restructure(model: str, pre, shape5, covariance5, *, regime: str,
                            zone: str, capital_share: float, squeeze: dict,
                            threshold_owner: dict | None, utility_owner: dict | None,
                            scenarios6, cost_rates6, nav: float,
                            risk_limits: dict | None = None,
                            risk_basis: str = 'GROSS_WEEKLY_LOSS',
                            decision_at: str | None = None,
                            cash_cap: float = 1.0, scenario_mass=None,
                            min_effective_tail_support: float | None = None) -> dict:
    """HOLD/PARTIAL/FULL/CASH selection after cost, caps, model-specific geometry.

    Must use the same function on each week of the 'stay invested' counterfactual.
    Expected scenarios are externally supplied as-of; caller must validate lineage.
    This function never silently fills missing owners or overrides active freeze.
    """
    if model not in ALL_METHODS:
        raise ValueError('INVALID_MODEL')
    if model == 'POLICY' and threshold_owner is not None and threshold_owner.get('reference_geometry_model') not in METHODS:
        return outcome('BLOCKED', 'POLICY_REFERENCE_GEOMETRY_OWNER_REQUIRED')
    p = _simplex(pre)
    if not isinstance(squeeze,dict) or squeeze.get('status')!='PASS':
        return outcome('BLOCKED', 'RISK_SQUEEZE_OWNER_CAPS_REQUIRED')
    if not owner_ok(threshold_owner) or not owner_ok(utility_owner):
        return outcome('BLOCKED', 'STRUCTURE_TOLERANCE_AND_UTILITY_OWNER_REQUIRED')
    if threshold_owner.get('model') != model:
        return outcome('BLOCKED','MODEL_SPECIFIC_THRESHOLD_OWNER_REQUIRED')
    geom_method = (model if model != 'POLICY'
                   else threshold_owner.get('reference_geometry_model'))
    expected_geometry = B.GEOMETRY.get(geom_method)
    if not expected_geometry or threshold_owner.get('geometry') != expected_geometry:
        return outcome('BLOCKED','MODEL_OWN_GEOMETRY_TOLERANCE_UNITS_MISMATCH',
                       expected_geometry=expected_geometry)
    reference_kind=threshold_owner.get('threshold_reference','ORIGINAL_BENCHMARK')
    if reference_kind not in ('ORIGINAL_BENCHMARK','PERMISSION_PROJECTED_TARGET'):
        return outcome('BLOCKED','OWNER_THRESHOLD_REFERENCE_NOT_RECOGNIZED')
    if (reference_kind=='PERMISSION_PROJECTED_TARGET' and
            threshold_owner.get('reference_geometry_contract') !=
            'PROJECTED_TARGET_NOT_CONSTRAINED_OPTIMUM'):
        return outcome('BLOCKED','PROJECTED_GEOMETRY_OWNER_INTERPRETATION_REQUIRED')
    threshold = threshold_owner.get('value')
    hurdle = utility_owner.get('hurdle')
    if (not isinstance(threshold,(int,float)) or isinstance(threshold,bool)
            or not np.isfinite(threshold) or threshold<0
            or not isinstance(hurdle,(int,float)) or isinstance(hurdle,bool)
            or not np.isfinite(hurdle) or hurdle<0):
        return outcome('BLOCKED','FINITE_NONNEGATIVE_THRESHOLD_AND_UTILITY_HURDLE_REQUIRED')
    if not np.isfinite(nav) or nav <= 0:
        raise ValueError('INVALID_NAV')
    if (not isinstance(cash_cap,(int,float)) or isinstance(cash_cap,bool)
            or not np.isfinite(cash_cap) or not 0 <= cash_cap <= 1):
        return outcome('BLOCKED', 'CASH_CAP_OWNER_FRACTION_DOMAIN_INVALID')
    if (not isinstance(squeeze.get('asset_caps'),dict) or
            set(squeeze['asset_caps']) != set(ASSETS[:5]) or
            not isinstance(squeeze.get('new_buy'),dict) or
            set(squeeze['new_buy']) != set(ASSETS[:5])):
        return outcome('BLOCKED', 'SIX_ASSET_POST_TRADE_CAP_INPUT_INCOMPLETE')
    caps6=np.array([squeeze['asset_caps'][a] for a in ASSETS[:5]]+[cash_cap],dtype=float)
    if (not np.isfinite(caps6).all() or (caps6<0).any() or (caps6>1).any()):
        return outcome('BLOCKED', 'INVALID_ASSET_OR_CASH_CAP_VALUES')
    if caps6.sum() < 1.-1e-10:
        return outcome('BLOCKED', 'ASSET_AND_CASH_CAP_SIMPLEX_INFEASIBLE')
    if risk_limits is not None:
        if risk_basis not in ('GROSS_WEEKLY_LOSS','NET_AFTER_SELF_FINANCING_COST'):
            return outcome('BLOCKED', 'UNKNOWN_OWNER_RISK_BASIS')
        if not isinstance(risk_limits,dict) or set(map(str,risk_limits)) != set(map(str,ALPHAS)):
            return outcome('BLOCKED', 'ALL_THREE_OWNER_VAR_ES_ALPHA_LIMITS_REQUIRED')
        for a, lim in risk_limits.items():
            if (not isinstance(lim,dict) or not all(
                isinstance(lim.get(key),(int,float)) and not isinstance(lim[key],bool)
                and np.isfinite(lim[key]) and lim[key]>=0 for key in ('VaR','ES'))):
                return outcome('BLOCKED','INVALID_OWNER_VAR_ES_LIMIT',alpha=a)
    scen = np.asarray(scenarios6,dtype=float)
    if scen.ndim != 2 or scen.shape[1] != 6 or not np.isfinite(scen).all():
        raise ValueError('SCENARIO_SHAPE')
    costs = _arr(cost_rates6,6)
    if np.any(costs < 0) or np.any(costs >= 1):
        raise ValueError('COST_RATES_OUTSIDE_0_1')
    s = np.asarray(covariance5,dtype=float)
    if scen.min() <= -1:
        return outcome('FAIL', 'IMPOSSIBLE_RETURN_AT_OR_BELOW_MINUS_ONE')
    try:
        probs=_scenario_mass(scenario_mass,len(scen))
    except (ValueError,TypeError) as e:
        return outcome('BLOCKED','ISSUED_SCENARIO_PROBABILITY_INVALID',detail=str(e))
    if min_effective_tail_support is not None and (
            not isinstance(min_effective_tail_support,(float,int)) or
            not np.isfinite(min_effective_tail_support) or min_effective_tail_support<=0):
        return outcome('BLOCKED','OWNER_MIN_TAIL_SUPPORT_INVALID')
    if s.shape != (5,5) or not np.isfinite(s).all():
        raise ValueError('COVARIANCE_SHAPE')
    if not np.isfinite(capital_share) or capital_share < 0 or capital_share > 1:
        return outcome('BLOCKED', 'RISKY_CAPITAL_SHARE_OUTSIDE_FRACTION_DOMAIN')
    # A selected VaR/ES-controlled capital share is a HARD upper bound on
    # every post-trade candidate, not just the ideal target vector.
    allowed_band = capital_domain(regime)
    if capital_share > float(allowed_band.max()) + 1e-12 or (
            regime == 'N' and capital_share < float(allowed_band.min()) - 1e-12):
        return outcome('BLOCKED', 'RISKY_SHARE_OUTSIDE_REGIME_CONTRACT')
    target = _asset_projection(_portfolio(shape5, capital_share), squeeze['asset_caps'])
    # Asset-specific Bottom 3F re-entry: preserve the allowed E1 purchase
    # even if E2--B2 are frozen. Forbidden proposed purchases go to cash;
    # NEVER redistribute them to another risky asset.
    for i, asset in enumerate(ASSETS[:5]):
        if not squeeze['new_buy'][asset] and target[i] > p[i]:
            target[i] = p[i]
    target[5] = 1. - float(target[:5].sum())
    target = _simplex(target)
    p_risky, target_risky = p[:5].sum(), target[:5].sum()
    if p_risky > 1e-12:
        g = B.structural_deviation(geom_method,p[:5]/p_risky,s,target=shape5)
        if g['status'] != 'PASS':
            return outcome('BLOCKED', 'STRUCTURAL_GEOMETRY_UNDEFINED', detail=g.get('reason'))
        original_deviation=float(g['deviation'])
    else:
        original_deviation=0.
    projected_deviation=None
    if p_risky > 1e-12 and target_risky > 1e-12:
        projected=projected_reference_deviation(geom_method,p[:5]/p_risky,
                                                 target[:5]/target_risky,s)
        if projected['status']=='PASS':
            projected_deviation=float(projected['deviation'])
        elif reference_kind=='PERMISSION_PROJECTED_TARGET':
            return outcome('NOT_EVALUATED','FEASIBLE_REFERENCE_GEOMETRY_UNDEFINED',
                           detail=projected.get('reason'))
    elif p_risky<=1e-12 or target_risky<=1e-12:
        projected_deviation=0.
    deviation=(original_deviation if reference_kind=='ORIGINAL_BENCHMARK'
               else projected_deviation)
    # The risky NAV fraction is a separate geometry and must not be ignored.
    fraction_drift = abs(float(p_risky - target_risky))
    tol_fraction = threshold_owner.get('capital_tolerance')
    if (not isinstance(tol_fraction,(float,int)) or isinstance(tol_fraction,bool)
            or not np.isfinite(tol_fraction) or tol_fraction<0):
        return outcome('BLOCKED','FINITE_CAPITAL_FRACTION_THRESHOLD_OWNER_REQUIRED')
    triggered = deviation > threshold or fraction_drift > tol_fraction
    violation_cap = (bool(np.any(p > caps6 + 1e-10))
                     or float(p[:5].sum()) > capital_share + 1e-10)
    candidates = [('HOLD',0.,p)]
    if zone in ('NO_RISKY_ZONE','STRESS_PRE_CONFIRMATION','EXIT_REGIME_CONFLICT'):
        # Freeze cannot buy, but may sell risky to cash; keep all approvals explicit.
        pass
    for lam in (.25,.5,.75,1.):
        c = TR.rebalance(p,target,lam)
        candidates.append(('FULL' if lam == 1. else 'PARTIAL',lam,c))
    candidates.append(('CASH',1.,np.r_[np.zeros(5),1.]))
    def _risk_compliant(weights: np.ndarray, trade_cost: float) -> bool:
        if risk_limits is None:
            return True
        if risk_basis == 'NET_AFTER_SELF_FINANCING_COST':
            net_return=(1.-trade_cost)*(1.+sc@weights)-1.
            metrics={float(a):T.var_es(-net_return,float(a),probs) for a in risk_limits}
        else:
            metrics=T.portfolio_tail(scen,weights,
                                     alphas=tuple(float(a) for a in risk_limits),mass=probs)
        return all(metrics[float(a)]['VaR'] <= lim['VaR']+1e-12 and
                   metrics[float(a)]['ES'] <= lim['ES']+1e-12 and
                   (min_effective_tail_support is None or
                    metrics[float(a)]['effective_tail_support']>=min_effective_tail_support)
                   for a,lim in risk_limits.items())

    # REV9: a NORMAL 50% capital FLOOR constrains all executed candidates,
    # not merely the proposed target. Stress has only an upper bound (50%).
    # A blocked purchase/cash cap can make the Normal floor unreachable; in that
    # case return BLOCKED instead of claiming HOLD satisfies the capital rule.
    regime_floor = 0.50 if regime == 'N' else 0.0
    violation_floor_pre = float(p[:5].sum()) < regime_floor - 1e-10
    # HOLD's tail risk may breach even when its geometry and weight caps do
    # not. That is a mandatory risk repair, not an optional threshold trade.
    violation_risk=not _risk_compliant(p,0.)
    violation=violation_cap or violation_risk or violation_floor_pre
    valid = []
    for action,lam,c in candidates:
        if (bool(np.any(c > caps6+1e-10)) or
                float(c[:5].sum()) > capital_share+1e-10 or
                float(c[:5].sum()) < regime_floor-1e-10):
            continue
        if any(c[i] > p[i]+1e-10 and not squeeze['new_buy'][a] for i,a in enumerate(ASSETS[:5])):
            continue
        charge=TR.self_financing_cost(p,c,costs)
        cost=float(charge['cost_fraction'])
        if not _risk_compliant(c,cost):
            continue
        score=_utility_scores(c,scen,nav*(1-cost),utility_owner,mass=probs)
        valid.append({'action':action,'lambda':lam,'weights':c, 'cost_fraction':cost,
                      'turnover':charge['turnover'],'expected_utility':score})
    if not valid:
        return outcome('BLOCKED','NO_FEASIBLE_HOLD_OR_RESTRUCTURE_ACTION',
                       model=model,regime=regime,zone=zone,
                       cap_violation=violation_cap,risk_violation=violation_risk,
                       normal_floor_violation_pre=violation_floor_pre,
                       normal_floor=regime_floor)
    hold = next((v for v in valid if v['action']=='HOLD'),None)
    if zone == 'NO_RISKY_ZONE':
        if not utility_owner.get('short_cash_liquidation_approved', False):
            return outcome('BLOCKED', 'SHORT_CASH_LIQUIDATION_OWNER_REQUIRED')
        chosen_cash = next((a for a in valid if a['action'] == 'CASH'), None)
        if chosen_cash is None:
            return outcome('BLOCKED', 'SHORT_CASH_ACTION_NOT_FEASIBLE')
        selected = chosen_cash
    elif not triggered and not violation:
        if hold is None:
            return outcome('BLOCKED','INFEASIBLE_HOLD_REQUIRES_OWNER_RISK_REPAIR')
        selected = hold
    else:
        if violation_floor_pre and not utility_owner.get('normal_floor_activation_approved',False):
            return outcome('BLOCKED','NORMAL_FLOOR_CAPITAL_ACTIVATION_OWNER_REQUIRED')
        if (violation_cap or violation_risk) and not utility_owner.get('emergency_repair_approved',False):
            return outcome('BLOCKED','CAP_BREACH_EMERGENCY_REPAIR_OWNER_REQUIRED')
        selected = max(valid,key=lambda z:z['expected_utility'])
        if not violation:
            if hold is None:
                return outcome('BLOCKED','HOLD_INFEASIBLE_OWNER_REVIEW_REQUIRED')
            if selected['expected_utility']-hold['expected_utility'] <= hurdle:
                selected = hold
    if (float(np.sum(selected['weights'][:5])) > capital_share + 1e-10
            or float(np.sum(selected['weights'][:5])) < regime_floor-1e-10
            or bool(np.any(selected['weights'] > caps6+1e-10))):
        return outcome('FAIL', 'POST_TRADE_SIX_ASSET_OR_RISKY_CAPITAL_BAND_BREACH')
    return outcome('PASS',method=model,regime=regime,zone=zone,
                   cap_violation_pre=violation_cap,risk_violation_pre=violation_risk,
                   normal_floor_violation_pre=violation_floor_pre,
                   executed_risky_share=float(np.sum(selected['weights'][:5])),
                   cash_cap=float(cash_cap),
                   action=selected['action'],post_weights=selected['weights'].tolist(),
                   target_weights=target.tolist(),gross_structural_deviation=deviation,
                   original_benchmark_geometry_deviation=original_deviation,
                   projected_reference_geometry_deviation=projected_deviation,
                   threshold_reference=reference_kind,
                   capital_drift=fraction_drift,own_geometry=threshold_owner.get('geometry',model),
                   cost_fraction=float(selected['cost_fraction']),
                   turnover=float(selected['turnover']),
                   expected_utility=float(selected['expected_utility']),
                   wealth_pre=float(nav),decision_at=decision_at,
                   mode='CONDITIONAL_EXECUTION_NOT_CERTIFIED_UNLESS_EXTERNAL_OWNERS_VERIFIED')


def _issue_before_window(packet: dict, start: date, prev_end: date | None) -> bool:
    """Records effective for week's Mon must have been issued after previous Fri and before Mon."""
    if not isinstance(packet,dict) or not packet.get('issue_at'):
        return False
    try:
        issued = stamp(packet['issue_at'])
    except (ValueError, TypeError):
        return False
    start_dt = datetime.combine(start,datetime.min.time(),tzinfo=timezone.utc)
    if issued >= start_dt:
        return False
    if prev_end is not None:
        # Do not claim that the prior week's Friday return was fully
        # available at Friday 00:01. It closes at the END of that UTC day.
        prev_end_dt = datetime.combine(prev_end,datetime.max.time(),tzinfo=timezone.utc)
        if issued <= prev_end_dt:
            return False
    return True



def _issued_no_later(source, decision) -> bool:
    try:
        return bool(source) and stamp(source) <= stamp(decision)
    except (TypeError,ValueError):
        return False


def parse_issued_path(records: list[dict], panel: dict, *, active_cap_owner_hash: str | None = None) -> dict:
    """Verify chronological weekly issued input; never derive Short/Bottom from labels.

    Each observation matching week_start/week_end must supply as-of fields:
       issue_at, regime N|D|G|BOTH, regime_issued_at, regime_source_id,
       short_confirmed bool, short_id if true, short_issued_at if true,
       bottom_permissions {E1..B2: issued permission records},
       bottom_start (week_start string) iff at least one authorized permission,
       first_n_confirmed bool, exit_id, exit_issued_at and exit_for_short_id
       if true (EXIT binds the active issued Short, not an older Short).
    Absence of issuer owner proof is indicated in the result; no auto-certification.
    """
    if panel.get('status')!='PASS':
        return outcome('BLOCKED','CANONICAL_PANEL_UNAVAILABLE')
    if not isinstance(records,list) or len(records)!=panel['n']:
        return outcome('BLOCKED','ISSUED_WEEKLY_RECORDS_REQUIRED_MATCH_ALL_781_WEEKS')
    seen_short, pending_short, prior, episodes = set(),None,None,[]
    superseded_shorts = []
    censored_episodes = []
    short_issued_clocks = {}
    active_short, active_short_id, bottom_passed = False, None, False
    def censor_pending(reason: str, terminal_index: int, terminal_issue: str | None = None):
        """Censor pre-Bottom events, never silently turn them into successes."""
        if pending_short is not None:
            censored_episodes.append({
                'status': 'NOT_EVALUATED', 'reason': reason,
                'short_id': pending_short['id'],
                'short_index': pending_short['index'],
                'last_observed_index': terminal_index,
                'short_issued_at': pending_short['issued_at'],
                'terminal_issue_at': terminal_issue,
                'bottom_observed': False,
                'avoided_loss_fraction': None,
                'scope': 'CENSORED_EVENT_NOT_A_ZERO_OR_A_POSITIVE_AVOIDED_LOSS'})
    validated = []
    for i,(raw_rec,start,end) in enumerate(zip(records,panel['starts'],panel['ends'])):
        if not isinstance(raw_rec, dict):
            return outcome('FAIL','ISSUED_WEEKLY_PACKET_NOT_MAPPING',index=i)
        rec=dict(raw_rec)
        if type(rec.get('short_confirmed',False)) is not bool or type(rec.get('first_n_confirmed',False)) is not bool:
            return outcome('FAIL','ISSUED_STATE_FLAGS_MUST_BE_BOOLEAN',index=i)
        if rec.get('week_start')!=start.isoformat() or rec.get('week_end')!=end.isoformat():
            return outcome('FAIL','ISSUED_WEEK_KEY_MISMATCH',index=i)
        if not _issue_before_window(rec,start, prior):
            # For the very first 2005 week a prior-week clock is not available.
            return outcome('FAIL','ISSUE_AFTER_WINDOW_START_OR_PRIOR_PERIOD_END',index=i)
        if rec.get('regime') not in REGIMES or not rec.get('regime_source_id') or not rec.get('regime_issued_at'):
            return outcome('BLOCKED','ISSUED_RQ01_STATE_AND_SOURCE_REQUIRED',index=i)
        if not _issued_no_later(rec['regime_issued_at'],rec['issue_at']):
            return outcome('FAIL','FUTURE_OR_INVALID_REGIME_ISSUE',index=i)
        if rec.get('short_confirmed') is True:
            if rec['regime'] not in STRESS:
                return outcome('U','SHORT_CONFIRMED_WITH_NONSTRESS_RQ01',index=i)
            sid=rec.get('short_id')
            short_at=rec.get('short_issued_at')
            if (not isinstance(sid,str) or not sid.strip() or
                    not short_at or not _issued_no_later(short_at,rec['issue_at'])):
                return outcome('FAIL','SHORT_ISSUANCE_INVALID',index=i)
            if sid in seen_short:
                if not active_short or active_short_id != sid:
                    return outcome('FAIL','OLD_SHORT_ID_REUSED_AS_NEW_SHORT',index=i,short_id=sid)
                if short_issued_clocks[sid] != stamp(short_at):
                    return outcome('FAIL','SAME_SHORT_ID_CHANGED_ISSUE_VINTAGE',index=i,short_id=sid)
            else:
                # Distinct newer Short may supersede an unresolved prior Short.
                # Keep it as a censored event, not a zero-loss observation.
                if pending_short is not None:
                    censor_pending('SUPERSEDED_BY_NEWER_SHORT',i,short_at)
                    superseded_shorts.append(pending_short['id'])
                seen_short.add(sid)
                short_issued_clocks[sid]=stamp(short_at)
                pending_short={'id':sid,'index':i,'issued_at':short_at}
                active_short,active_short_id,bottom_passed=True,sid,False
        raw_bottom=rec.get('bottom_permissions',{})
        if not isinstance(raw_bottom,dict):
            return outcome('FAIL','BOTTOM_PERMISSION_MAP_REQUIRED',index=i)
        bottom=dict(raw_bottom)
        if bottom and set(bottom)-set(ASSETS[:5]):
            return outcome('FAIL','UNKNOWN_BOTTOM_ASSET',index=i)
        valid = []
        for asset,perm in bottom.items():
            if not isinstance(perm, dict) or perm.get('authorized') is not True:
                continue
            if not _asof_record(perm,rec['issue_at'],('asset','committee_id','risk_caps_owner_hash')) or perm['asset']!=asset:
                return outcome('FAIL','BOTTOM_PERMISSION_EVIDENCE_INVALID',index=i,asset=asset)
            if active_cap_owner_hash and perm['risk_caps_owner_hash']!=active_cap_owner_hash:
                return outcome('FAIL','BOTTOM_CAP_OWNER_HASH_MISMATCH',index=i,asset=asset)
            factors=perm.get('factor_evidence')
            families=('EWI','COUNTRY','RISK_COV')
            if not isinstance(factors,dict) or set(factors)!=set(families):
                return outcome('BLOCKED','BOTTOM_3F_EWI_COUNTRY_RISK_COV_EVIDENCE_REQUIRED',index=i,asset=asset)
            votes,input_sets={},{}
            for family in families:
                info=factors[family]
                if not _asof_record(info,rec['issue_at'],('vote','input_ids')):
                    return outcome('FAIL','BOTTOM_FACTOR_VOTE_CLOCK_OR_SOURCE_INVALID',index=i,asset=asset,family=family)
                if type(info['vote']) is not bool or not isinstance(info['input_ids'],list) or not info['input_ids']:
                    return outcome('FAIL','BOTTOM_FACTOR_VOTE_INPUT_DOMAIN',index=i,asset=asset,family=family)
                votes[family]=info['vote'];input_sets[family]=info['input_ids']
            rule=perm.get('committee_rule_owner')
            if not owner_ok(rule) or perm.get('committee_rule_owner_hash') != rule['owner_hash']:
                return outcome('BLOCKED','BOTTOM_3F_COMMITTEE_OWNER_REQUIRED',index=i,asset=asset)
            committee=RP.bottom_committee(votes,input_sets,rule)
            if committee['status']!='PASS' or committee.get('decision') is not True:
                return outcome('BLOCKED','BOTTOM_3F_DEPENDENCE_ADJUSTED_COMMITTEE_NOT_APPROVED',
                               index=i,asset=asset,committee_status=committee['status'])
            risk_check=perm.get('risk_check')
            if not (isinstance(risk_check,dict) and risk_check.get('status')=='PASS' and
                    risk_check.get('within_caps') is True and
                    risk_check.get('risk_caps_owner_hash')==perm['risk_caps_owner_hash']):
                return outcome('BLOCKED','BOTTOM_ASSET_RISK_CHECK_OWNER_NOT_MATCHED',index=i,asset=asset)
            perm=dict(perm)
            perm['three_factor_structurally_validated']=True
            if not _bottom_permission_proof(asset,perm,rec['issue_at'],
                                            active_cap_owner_hash,start.isoformat()):
                return outcome('BLOCKED','BOTTOM_FUNDING_ELIGIBILITY_RISK_WINDOW_OR_PROVENANCE_NOT_VERIFIED',
                               index=i,asset=asset)
            bottom[asset]=perm
            valid.append(asset)
        rec['bottom_permissions']=bottom
        if valid and rec['regime'] not in STRESS:
            return outcome('U','BOTTOM_PERMISSION_WITH_NONSTRESS_REGIME',index=i)
        if valid and not active_short:
            return outcome('BLOCKED','BOTTOM_WITHOUT_ACTIVE_ISSUED_SHORT',index=i)
        if valid and pending_short is not None:
            try:
                bottom_eligible = bool(rec.get('bottom_start')) and date.fromisoformat(rec['bottom_start']) <= start
            except (TypeError,ValueError):
                bottom_eligible = False
            if not bottom_eligible:
                return outcome('FAIL','BOTTOM_START_NOT_EFFECTIVE',index=i)
            if pending_short['index']==i:
                return outcome('BLOCKED','SAME_WEEK_SHORT_TO_BOTTOM_INTERVAL_NOT_IDENTIFIABLE',index=i)
            if pending_short['index'] < i:
                episodes.append({'short_id':pending_short['id'], 'short_index':pending_short['index'],
                                 'bottom_index':i,'assets':valid})
                pending_short = None
                bottom_passed = True
        if rec.get('first_n_confirmed') is True:
            if not active_short:
                return outcome('BLOCKED','EXIT_WITHOUT_ACTIVE_ISSUED_SHORT',index=i)
            if rec.get('regime')!='N' or not rec.get('exit_id') or not rec.get('exit_issued_at') or not _issued_no_later(rec['exit_issued_at'],rec['issue_at']):
                return outcome('FAIL','INCOHERENT_EXIT',index=i)
            if rec.get('exit_for_short_id') != active_short_id:
                return outcome('FAIL','EXIT_DOES_NOT_BIND_ACTIVE_SHORT',index=i,
                               expected_active_short_id=active_short_id)
            if stamp(rec['exit_issued_at']) < short_issued_clocks[active_short_id]:
                return outcome('FAIL','EXIT_ISSUED_BEFORE_REFERENCED_SHORT',index=i)
            if pending_short is not None:
                censor_pending('EXIT_BEFORE_VALID_BOTTOM',i,rec['exit_issued_at'])
            pending_short = None
            active_short,active_short_id,bottom_passed = False,None,False
            expected_zone='EXIT'
        elif active_short and valid:
            expected_zone='BOTTOM_REENTRY_ZONE'
        elif active_short and bottom_passed:
            expected_zone='POST_BOTTOM_STRESS'
        elif active_short:
            expected_zone='NO_RISKY_ZONE'
        else:
            expected_zone='NORMAL' if rec['regime']=='N' else 'STRESS_PRE_CONFIRMATION'
        if rec.get('zone') not in (None,expected_zone):
            return outcome('FAIL','ISSUED_ZONE_INCONSISTENT_WITH_SHORT_EXIT_BOTTOM',index=i,expected=expected_zone)
        rec['zone']=expected_zone
        if active_short and rec['regime']=='N':
            return outcome('U','UNRESOLVED_SHORT_WITH_NORMAL_REGIME',index=i)
        validated.append(rec)
        prior=end
    if pending_short is not None:
        censor_pending('RIGHT_CENSORED_AT_DEV_END',len(validated)-1,None)
    return outcome('PASS',records=validated,episodes=episodes,
                   censored_episodes=censored_episodes,
                   event_support={'paired_short_bottom':len(episodes),
                                  'censored_without_bottom':len(censored_episodes),
                                  'superseded_without_bottom':sum(
                                      z['reason']=='SUPERSEDED_BY_NEWER_SHORT' for z in censored_episodes),
                                  'exited_without_bottom':sum(
                                      z['reason']=='EXIT_BEFORE_VALID_BOTTOM' for z in censored_episodes),
                                  'right_censored_without_bottom':sum(
                                      z['reason']=='RIGHT_CENSORED_AT_DEV_END' for z in censored_episodes)},
                   superseded_short_ids=superseded_shorts,
                   unmatched_latest_short_id=None if pending_short is None else pending_short['id'],
                   claim='ISSUED_FIELDS_STRUCTURALLY_VALIDATED_NOT_EXTERNAL_ISSUER_AUTHENTICATION')



def authorized_execution_cost_rates(config: dict) -> dict:
    """Cost identity gate for optimizer / executor / wealth ledger.

    Never optimize at rates k and execute at rates k' under the same owner ID.
    No owner source, mixed currency units, stale schedule or invented free
    transaction costs can be silently accepted.
    """
    own=config.get('cost_owner')
    if not owner_ok(own) or own.get('unit')!='PROPORTIONAL_TRADE_VALUE':
        return outcome('BLOCKED','EXACT_EXECUTION_COST_OWNER_REQUIRED')
    try:
        owner_rates=np.asarray([own['rates'][a] for a in ASSETS],float)
        executed=_arr(config.get('cost_rates6'),6)
    except (KeyError,TypeError,ValueError) as exc:
        return outcome('BLOCKED','EXECUTION_COST_VECTOR_MISSING',detail=str(exc))
    if (not np.isfinite(owner_rates).all() or (owner_rates<0).any()
        or (owner_rates>=1).any() or not np.array_equal(owner_rates,executed)):
        return outcome('BLOCKED','OWNER_OPTIMIZER_EXECUTOR_COST_SCHEDULE_MISMATCH')
    return outcome('PASS',rates=executed.tolist(),owner_hash=own['owner_hash'])


def _r23_locked_policy_week(model: str, t: int, start_w: np.ndarray, nav: float, panel: dict,
                            config: dict, rec: dict) -> dict:
    """R23 NO_TRADE_LOCKED week for the Main model (opt-in).  No optimiser, no candidate, no fill.

    post == pre, cost = turnover = 0, NAV and weights drift passively; the hard-risk monitor still runs
    and a breach is only recorded (locked_risk_monitor / cap_violation_pre / risk_violation_pre), never
    repaired by a trade.  Same as-of scenario-mass owner gates as the normal path."""
    if __package__:
        from . import main_lock as ML
    else:
        from v8_w05_rq03 import main_lock as ML
    verified_cost=authorized_execution_cost_rates(config)
    if verified_cost['status']!='PASS':
        return verified_cost
    scen = np.asarray(config['scenario_by_week'][t],dtype=float)
    mass_owner=config.get('scenario_probability_owner')
    mass_modes=config.get('scenario_weight_mode_by_week')
    mass_tape=config.get('scenario_mass_by_week')
    if not owner_ok(mass_owner) or mass_owner.get('mode') not in ('UNIFORM_ISSUED','WEIGHTED_ISSUED'):
        return outcome('BLOCKED','EXPLICIT_OWNER_SCENARIO_PROBABILITY_CONVENTION_REQUIRED')
    if not isinstance(mass_modes,list) or not isinstance(mass_tape,list) or (
            len(mass_modes)!=len(config['scenario_by_week']) or
            len(mass_tape)!=len(config['scenario_by_week'])):
        return outcome('BLOCKED','SCENARIO_MASS_ASOF_CHAIN_REQUIRED')
    if mass_modes[t] != mass_owner['mode']:
        return outcome('BLOCKED','ISSUED_SCENARIO_WEIGHT_MODE_OWNER_MISMATCH')
    try:
        mass=_scenario_mass(mass_tape[t],len(scen))
    except (ValueError,TypeError) as ex:
        return outcome('BLOCKED','INVALID_ASOF_SCENARIO_PROBABILITIES',detail=str(ex))
    global_caps=config['asset_cap_owner'].get('caps') or {}
    if set(global_caps)!=set(ASSETS):
        return outcome('BLOCKED','SIX_GLOBAL_CAPS_REQUIRED')
    _simplex(start_w)                      # validation only: the held vector is kept bit-identical (post == pre)
    post = np.asarray(start_w,dtype=float).copy()
    x = panel['X'][t]
    period_return = float(post @ x)
    new_nav = nav*(1+period_return)
    next_pre = TR.drift(post,x)
    tails = T.portfolio_tail(scen,post,alphas=ALPHAS,mass=mass)
    table={str(alpha):{'gross_VaR':float(tails[alpha]['VaR']),'gross_ES':float(tails[alpha]['ES']),
                       'net_VaR':float(tails[alpha]['VaR']),'net_ES':float(tails[alpha]['ES']),
                       'effective_tail_support':float(tails[alpha]['effective_tail_support'])}
           for alpha in ALPHAS}
    limits = (config['risk_owner'].get('caps') or {}).get(model,{}).get(rec['regime'])
    measures, risk_caps = ({}, {})
    if isinstance(limits,dict) and all(str(a) in limits for a in ALPHAS):
        measures, risk_caps = ML.locked_tail_check(scen,post,{a:limits[str(a)] for a in ALPHAS},mass=mass)
    monitor = ML.locked_risk_monitor(post,rec['regime'],
                                     asset_caps={a:float(global_caps[a]) for a in ASSETS[:5]},
                                     risk_measures=measures,risk_caps=risk_caps)
    return outcome('PASS',pre=start_w.tolist(),post=post.tolist(),target=post.tolist(),
                   nav_pre=float(nav),nav=float(new_nav),next_pre=next_pre,
                   action=ML.NO_TRADE_LOCKED,regime=rec['regime'],zone=rec['zone'],
                   risky_share_selected=float(post[:5].sum()),risky_share_post=float(post[:5].sum()),
                   structural_deviation=None,capital_drift=None,
                   cost_fraction=0.0,cost=0.0,turnover=0.0,risk=table,
                   scenario_effective_count=float(1./np.sum(mass**2)),
                   scenario_mass_owner_hash=mass_owner['owner_hash'],
                   gross_return=period_return,ret=float(new_nav/nav-1),
                   issued_action_at=rec['issue_at'],source_tape_sha=panel['tape_sha'],
                   owner_risk_hash=config['risk_owner']['owner_hash'],
                   owner_cap_hash=config['asset_cap_owner']['owner_hash'],
                   owner_cost_hash=config['cost_owner']['owner_hash'],
                   cap_violation_pre=any(b.startswith('ASSET_CAP') or b in ('STRESS_RISKY_SHARE','NORMAL_RISKY_BAND')
                                         for b in monitor['breaches']),
                   risk_violation_pre=any('@' in b for b in monitor['breaches']),
                   post_cash_cap=float(global_caps['C1']),bottom_squeeze_by_asset={},
                   new_buy_permission={a:False for a in ASSETS[:5]},
                   selected_risky_cap_headroom=0.0,projected_risky_capacity=None,
                   selected_frontier_risky_fraction=float(post[:5].sum()),
                   post_asset_cap_headroom={a:float(global_caps[a])-float(post[j]) for j,a in enumerate(ASSETS)},
                   locked_week=True,locked_risk_monitor=monitor)


def _continuation_weights(model: str, t: int, start_w: np.ndarray, nav: float,
                          panel: dict, shape_by_week: dict, cov_by_week,
                          config: dict, *, ignore_short_freeze: bool = False):
    """One trade decision at start of week t, then period t realised return and drift.
    This is only invoked with structurally complete external week-specific inputs.
    """
    rec = config['issued_records'][t]
    if model == 'EW':
        ew_owner=config.get('ew_definition_owner')
        if not owner_ok(ew_owner) or ew_owner.get('definition') != 'EW5_RISKY_SLEEVE':
            return outcome('BLOCKED','EW5_VERSUS_EW6_DEFINITION_OWNER_REQUIRED')
    zone = rec['zone']
    if ignore_short_freeze and zone in ('NO_RISKY_ZONE','POST_BOTTOM_STRESS'):
        # Research counterfactual: invest under stress <=50%, not an official permission.
        zone = 'NORMAL'
    if config.get('r23_main_only_lock') is True:
        # R23: the Short->Bottom lock is Main-only.  Benchmarks are exempt (regime-only zone, no cash
        # liquidation, no Bottom overlay); the Main model is NO_TRADE_LOCKED while the zone is NO_RISKY_ZONE
        # and is Common-ON (Normal/Stress) before its first Short.
        if model != 'POLICY':
            zone = 'NORMAL'
        elif zone == 'STRESS_PRE_CONFIRMATION':
            zone = 'NORMAL'
        elif zone == 'NO_RISKY_ZONE':
            return _r23_locked_policy_week(model,t,start_w,nav,panel,config,rec)
    model_config = config['models'][model]
    global_caps=config['asset_cap_owner'].get('caps') or {}
    if set(global_caps)!=set(ASSETS) or any(model_config['asset_caps'].get(a,1.1) > global_caps[a]+1e-10
                                            for a in ASSETS[:5]):
        return outcome('BLOCKED','MODEL_ASSET_CAP_EXCEEDS_OWNER_GLOBAL_CAP')
    sq = bottom_risk_squeeze(rec['regime'],zone,
                             base_caps=model_config['base_caps'],defense=model_config['defense'],
                             allowance=model_config['allowance'],permission=rec.get('bottom_permissions',{}),
                             owner_caps=model_config['asset_caps'],decision_at=rec['issue_at'],
                             trade_at=rec['week_start'],
                             active_risk_cap_hash=config['asset_cap_owner']['owner_hash'],
                             weeks_after_bottom=float(rec.get('weeks_after_bottom',0)),
                             decay_tau=model_config.get('decay_tau'),
                             weeks_after_exit=float(rec.get('weeks_after_exit',0)),
                             defense_decay_tau=model_config.get('defense_decay_tau'))
    if sq['status']!='PASS':
        return sq
    eff=effective_trade_caps(start_w,sq,config['asset_cap_owner'])
    if eff['status']!='PASS':
        return eff
    # These tightened caps are not new owner grants; they only restrict a
    # previously approved owner cap and are auditable to its original hash.
    tightened_owner={**config['asset_cap_owner'],'caps':eff['caps']}
    effective_risky_caps={a:eff['caps'][a] for a in ASSETS[:5]}
    objective_owner=config.get('capital_objective_owner')
    if (not owner_ok(objective_owner) or
            objective_owner.get('objective') not in
            ('MAX_FEASIBLE_RISKY_SHARE','NET_UTILITY')):
        return outcome('BLOCKED','EXPLICIT_OWNER_CAPITAL_OBJECTIVE_REQUIRED')
    objective=objective_owner['objective']
    # Critical: optimizer and executed restructure must charge ONE identical
    # owner-authenticated cost schedule; otherwise a NAV-optimal allocation is
    # not the allocation the executor is scoring.
    verified_cost=authorized_execution_cost_rates(config)
    if verified_cost['status']!='PASS':
        return verified_cost
    scen = np.asarray(config['scenario_by_week'][t],dtype=float)
    mass_owner=config.get('scenario_probability_owner')
    mass_modes=config.get('scenario_weight_mode_by_week')
    mass_tape=config.get('scenario_mass_by_week')
    if not owner_ok(mass_owner) or mass_owner.get('mode') not in ('UNIFORM_ISSUED','WEIGHTED_ISSUED'):
        return outcome('BLOCKED','EXPLICIT_OWNER_SCENARIO_PROBABILITY_CONVENTION_REQUIRED')
    if not isinstance(mass_modes,list) or not isinstance(mass_tape,list) or (
            len(mass_modes)!=len(config['scenario_by_week']) or
            len(mass_tape)!=len(config['scenario_by_week'])):
        return outcome('BLOCKED','SCENARIO_MASS_ASOF_CHAIN_REQUIRED')
    if mass_modes[t] != mass_owner['mode']:
        return outcome('BLOCKED','ISSUED_SCENARIO_WEIGHT_MODE_OWNER_MISMATCH')
    try:
        mass=_scenario_mass(mass_tape[t],len(scen))
    except (ValueError,TypeError) as ex:
        return outcome('BLOCKED','INVALID_ASOF_SCENARIO_PROBABILITIES',detail=str(ex))
    capital_step=objective_owner.get('capital_grid_step',0.05)
    if (not isinstance(capital_step,(float,int)) or isinstance(capital_step,bool) or
            not np.isfinite(capital_step) or not 0<capital_step<=0.5):
        return outcome('BLOCKED','CAPITAL_GRID_STEP_OWNER_DOMAIN_INVALID')
    owner_basis=config['risk_owner'].get('risk_basis')
    if owner_basis not in ('GROSS_WEEKLY_LOSS','NET_AFTER_SELF_FINANCING_COST'):
        return outcome('BLOCKED','OWNER_RISK_BASIS_EXPLICITLY_REQUIRED')
    if objective == 'NET_UTILITY':
        cap = net_capital_utility_choice(
            scen,shape_by_week[model][t],start_w,model,rec['regime'],nav=nav,
            risk_owner=config.get('risk_owner'),asset_cap_owner=tightened_owner,
            cost_owner=config.get('cost_owner'),utility_owner=model_config.get('utility_owner'),
            step=float(capital_step),mass=mass,projected_caps5=effective_risky_caps)
    elif objective == 'MAX_FEASIBLE_RISKY_SHARE':
        if owner_basis != 'GROSS_WEEKLY_LOSS':
            return outcome('BLOCKED','GROSS_FRONTIER_CANNOT_SOLVE_NET_OWNER_RISK_BUDGET')
        frontier = risk_frontier(scen,{model:{'status':'PASS','shape':shape_by_week[model][t]}},
                                 rec['regime'],step=float(capital_step),mass=mass,
                                 projected_caps5=effective_risky_caps)
        cap = choose_capital_by_risk(frontier,model,rec['regime'],config.get('risk_owner'),
                                     asset_cap_owner=tightened_owner)
    else:
        return outcome('BLOCKED','UNRECOGNIZED_OWNER_CAPITAL_OBJECTIVE')
    if cap['status']!='PASS':
        return cap
    executable_fraction=(cap['selected']['executed_risky_fraction'] if 'selected' in cap
                         else cap['feasible_executed_risky_fraction'])
    if (rec['regime']=='N' and executable_fraction<0.5-1e-10):
        return outcome('BLOCKED','NORMAL_MINIMUM_CAPITAL_NOT_EXECUTABLE',
                       requested_share=cap['risky_share'], executable_share=executable_fraction)
    decision = benchmark_restructure(model,start_w,shape_by_week[model][t],cov_by_week[t],
                                      regime=rec['regime'],zone=zone,capital_share=cap['risky_share'],
                                      squeeze=sq,threshold_owner=model_config['tolerance_owner'],
                                      utility_owner=model_config['utility_owner'],
                                      scenarios6=scen,cost_rates6=config['cost_rates6'],nav=nav,
                                      risk_limits=config['risk_owner']['caps'][model][rec['regime']],
                                      risk_basis=owner_basis,
                                      decision_at=rec['issue_at'],
                                      cash_cap=float(global_caps['C1']),scenario_mass=mass,
                                      min_effective_tail_support=config['risk_owner'].get('min_effective_tail_support'))
    if decision['status']!='PASS':
        return decision
    post = _arr(decision['post_weights'],6)
    period_return = float(post @ panel['X'][t]); c = decision['cost_fraction']
    new_nav = nav*(1-c)*(1+period_return)
    next_pre = TR.drift(post,panel['X'][t])
    gross_risk=T.portfolio_tail(scen,post,alphas=ALPHAS,mass=mass)
    net_losses=1-(1-c)*(1+scen@post)
    net_risk={alpha:T.var_es(net_losses,alpha,mass) for alpha in ALPHAS}
    table={str(alpha):{'gross_VaR':float(gross_risk[alpha]['VaR']),
                       'gross_ES':float(gross_risk[alpha]['ES']),
                       'net_VaR':float(net_risk[alpha]['VaR']),
                       'net_ES':float(net_risk[alpha]['ES']),
                       'effective_tail_support':float(net_risk[alpha]['effective_tail_support'])}
           for alpha in ALPHAS}
    return outcome('PASS',pre=start_w.tolist(),post=post.tolist(),
                   target=decision['target_weights'],
                   nav_pre=float(nav),nav=float(new_nav),next_pre=next_pre,
                   action=decision['action'],regime=rec['regime'],zone=zone,
                   risky_share_selected=float(cap['risky_share']),
                   risky_share_post=float(post[:5].sum()),
                   structural_deviation=float(decision['gross_structural_deviation']),
                   capital_drift=float(decision['capital_drift']),
                   cost_fraction=float(c),cost=float(nav*c),
                   turnover=decision['turnover'],risk=table,
                   scenario_effective_count=float(1./np.sum(mass**2)),
                   scenario_mass_owner_hash=mass_owner['owner_hash'],
                   gross_return=period_return,
                   ret=float((1-c)*(1+period_return)-1),
                   issued_action_at=rec['issue_at'],source_tape_sha=panel['tape_sha'],
                   owner_risk_hash=config['risk_owner']['owner_hash'],
                   owner_cap_hash=config['asset_cap_owner']['owner_hash'],
                   owner_cost_hash=config['cost_owner']['owner_hash'],
                   cap_violation_pre=decision['cap_violation_pre'],
                   risk_violation_pre=decision['risk_violation_pre'],
                   post_cash_cap=decision['cash_cap'],
                   bottom_squeeze_by_asset=sq['by_asset'],
                   new_buy_permission=sq['new_buy'],
                   selected_risky_cap_headroom=float(cap['risky_share']-post[:5].sum()),
                   projected_risky_capacity=eff['max_risky_capacity'],
                   selected_frontier_risky_fraction=float(
                       cap['selected']['executed_risky_fraction'] if 'selected' in cap
                       else cap['feasible_executed_risky_fraction']),
                   post_asset_cap_headroom={a:float((sq['asset_caps'][a] if a!='C1' else global_caps['C1'])-post[j])
                                            for j,a in enumerate(ASSETS)})



def matched_risk_tolerance_summary(history: list[dict], risk_caps: dict, *,
                                   risk_basis: str = 'NET_AFTER_SELF_FINANCING_COST') -> dict:
    """Matched weekly as-of forecast diagnostics, not independent validation.

    Count every issued regime separately, including zero-support D or G;
    distinguish selected risky target allocation from executed risky holdings.
    Report VaR/ES at all required alphas and model/regime budget violations.
    Historical VaR exceedances are single weekly observations, never events.
    """
    if not isinstance(history,list) or not isinstance(risk_caps,dict):
        return outcome('BLOCKED','RISK_HISTORY_AND_OWNER_LIMITS_REQUIRED')
    if risk_basis not in ('NET_AFTER_SELF_FINANCING_COST', 'GROSS_WEEKLY_LOSS'):
        return outcome('BLOCKED','RISK_SUMMARY_BASIS_OWNER_REQUIRED')
    risk_prefix='net' if risk_basis=='NET_AFTER_SELF_FINANCING_COST' else 'gross'
    return_key='net_return' if risk_prefix=='net' else 'gross_return'
    by_regime={}
    for reg in REGIMES:
        items=[h for h in history if h.get('regime')==reg]
        if not items:
            by_regime[reg]=outcome('NOT_EVALUATED','ZERO_ISSUED_REGIME_WEEKS',weeks=0)
            continue
        limits=risk_caps.get(reg)
        if not isinstance(limits,dict):
            return outcome('BLOCKED','OWNER_MODEL_REGIME_RISK_LIMIT_MISSING',regime=reg)
        details={}
        violations=0
        for alpha in ALPHAS:
            key=str(alpha)
            limit=limits.get(key)
            if (not isinstance(limit,dict) or not all(
                isinstance(limit.get(field),(int,float)) and np.isfinite(limit[field])
                for field in ('VaR','ES'))):
                return outcome('BLOCKED','REGIME_VAR_ES_BUDGET_NOT_VERIFIED',regime=reg,alpha=key)
            vals=np.asarray([[h['risk'][key][risk_prefix+'_VaR'],h['risk'][key][risk_prefix+'_ES']]
                             for h in items],dtype=float)
            if not np.isfinite(vals).all():
                return outcome('FAIL','INVALID_WEEKLY_FORECAST_RISK',regime=reg,alpha=key)
            breaches=((vals[:,0]>limit['VaR']+1e-12)|(vals[:,1]>limit['ES']+1e-12))
            # R23: a breach during a NO_TRADE_LOCKED week is escalated to the owner, it cannot be repaired
            # by a trade, so it is reported separately instead of failing the whole replay.
            locked_mask=np.asarray([bool(h.get('locked_week')) for h in items],dtype=bool)
            violations+=int((breaches & ~locked_mask).sum())
            realized_exceedances=sum(-h[return_key]>h['risk'][key][risk_prefix+'_VaR']
                                     for h in items)
            details[key]={risk_prefix+'_VaR_median':float(np.median(vals[:,0])),
                          risk_prefix+'_VaR_max':float(vals[:,0].max()),
                          risk_prefix+'_ES_median':float(np.median(vals[:,1])),
                          risk_prefix+'_ES_max':float(vals[:,1].max()),
                          'owner_VaR_cap':float(limit['VaR']),
                          'owner_ES_cap':float(limit['ES']),
                          'budget_breach_weeks':int((breaches & ~locked_mask).sum()),
                          'realized_loss_exceedance_weeks':int(realized_exceedances),
                          'exceedance_scope':'WEEK_OBSERVATIONS_NOT_INDEPENDENT_EPISODES'}
            if locked_mask.any():
                details[key]['locked_week_escalations']=int((breaches & locked_mask).sum())
        if violations:
            return outcome('FAIL','POST_TRADE_ISSUED_RISK_BUDGET_BREACH',
                           regime=reg,breach_alpha_week_count=violations,weeks=len(items))
        selected=np.asarray([h['risky_share_selected'] for h in items],float)
        actual=np.asarray([h['risky_share_post'] for h in items],float)
        by_regime[reg]=outcome('PASS',weeks=len(items),
            selected_risky_share_min=float(selected.min()),
            selected_risky_share_median=float(np.median(selected)),
            selected_risky_share_max=float(selected.max()),
            executed_risky_share_min=float(actual.min()),
            executed_risky_share_median=float(np.median(actual)),
            executed_risky_share_max=float(actual.max()),
            var_es=details,weekly_alpha_budget_breaches=violations)
    return outcome('PASS',by_regime=by_regime,matched_week_count=len(history),
                   risk_basis=risk_basis,
                   scope='SAME_ISSUED_WEEKS_FORECAST_RISK_NOT_OOS_CERTIFICATION')


def validate_short_cash_freeze(config: dict) -> dict:
    """Approve *one* explicit C1 100%-cash counterfactual, not a free asset.

    The user requests that non-investment between the latest Short and Bottom
    be compared against each benchmark's own investing policy.  C1 is a proxy,
    not automatically investable or unlimited: the same owner cash, asset-cap,
    funding and single-cost conditions bind the freeze arm.  A missing signed
    historical permission remains BLOCKED rather than inventing a safe haven.
    This gate never changes the actual issued policy or the return tape.
    """
    if not isinstance(config, dict):
        return outcome('BLOCKED', 'CASH_COUNTERFACTUAL_OWNER_CONFIG_REQUIRED')
    cash_owner=config.get('cash_owner')
    cap_owner=config.get('asset_cap_owner')
    cost_owner=config.get('cost_owner')
    if not all(owner_ok(o) for o in (cash_owner,cap_owner,cost_owner)):
        return outcome('BLOCKED','CASH_COUNTERFACTUAL_FUNDING_CAP_COST_OWNER_REQUIRED')
    if (cash_owner.get('asset_id') != 'C1' or
            cash_owner.get('counterfactual_freeze_approved') is not True or
            cash_owner.get('cash_proxy_funding_approved') is not True):
        return outcome('BLOCKED','C1_FREEZE_AND_PROXY_FUNDING_AUTHORIZATION_REQUIRED')
    if cap_owner.get('unit') != 'FRACTION_OF_NAV':
        return outcome('BLOCKED','C1_CASH_CAP_UNIT_REQUIRED')
    caps=cap_owner.get('caps')
    if not isinstance(caps,dict) or set(caps)!=set(ASSETS):
        return outcome('BLOCKED','SIX_ASSET_OWNER_CAPS_REQUIRED_FOR_FREEZE')
    try:
        numbers=np.array([caps[a] for a in ASSETS],dtype=float)
    except (TypeError,ValueError,KeyError):
        return outcome('BLOCKED','INVALID_C1_FREEZE_CAPS')
    if (not np.isfinite(numbers).all() or np.any(numbers<0) or
            np.any(numbers>1)):
        return outcome('BLOCKED','INVALID_C1_FREEZE_CAPS')
    if numbers[5] < 1.-1e-10:
        return outcome('BLOCKED','C1_FULL_FREEZE_EXCEEDS_OWNER_CASH_CAP',
                       cash_cap=float(numbers[5]),required_cash_weight=1.)
    if (cost_owner.get('unit')!='PROPORTIONAL_TRADE_VALUE' or
            set(cost_owner.get('rates') or {})!=set(ASSETS)):
        return outcome('BLOCKED','C1_FREEZE_COST_OWNER_UNIT_OR_SCHEDULE_REQUIRED')
    try:
        rates=np.array([cost_owner['rates'][a] for a in ASSETS],dtype=float)
    except (TypeError,ValueError,KeyError):
        return outcome('BLOCKED','INVALID_C1_FREEZE_COST_SCHEDULE')
    if (not np.isfinite(rates).all() or (rates<0).any() or (rates>=1).any()):
        return outcome('BLOCKED','INVALID_C1_FREEZE_COST_SCHEDULE')
    return outcome('PASS',cash_asset='C1',cash_cap=1.,
                   owner_cash_hash=cash_owner['owner_hash'],
                   owner_cap_hash=cap_owner['owner_hash'],
                   owner_cost_hash=cost_owner['owner_hash'],
                   scope='COUNTERFACTUAL_PERMISSION_NOT_EXECUTION_CERTIFICATION')


def short_to_bottom_counterfactual(panel: dict, parsed: dict, shape_by_week: dict,
                                   cov_by_week: np.ndarray, config: dict) -> dict:
    """Paired event study for every (model, latest Short, first permitted Bottom).

    Must supply externally issued snapshots/costs/caps/utility + risk budgets.
    Baseline 2005.. prior-Short replays identical; INVESTED counterfactual
    ignores only the Short freeze while observing benchmark-specific restructure,
    risk budgets and all investment caps. FREEZE sells to cash at Short and
    accrues cash until first authorized Bottom. NAV compared BEFORE bottom buy.
    Both arms use identical NAV at Short, weeks, asset returns and cost schedule.
    """
    if parsed.get('status')!='PASS' or panel.get('status')!='PASS':
        return outcome('BLOCKED','CANONICAL_ISSUED_PATH_REQUIRED')
    if not all(owner_ok(config.get(k)) for k in ('risk_owner','asset_cap_owner','cost_owner','cash_owner')):
        return outcome('BLOCKED','OFFICIAL_CAP_COST_FUNDING_OWNER_EVIDENCE_REQUIRED')
    if (config['risk_owner'].get('risk_unit') != 'DECIMAL_LOSS_FRACTION_OF_NAV'
            or config['asset_cap_owner'].get('unit') != 'FRACTION_OF_NAV'
            or config['cost_owner'].get('unit') != 'PROPORTIONAL_TRADE_VALUE'):
        return outcome('BLOCKED','OFFICIAL_RISK_CAP_COST_UNITS_OWNER_UNVERIFIED')
    if not config['cost_owner'].get('rates') or set(config['cost_owner']['rates'])!=set(ASSETS):
        return outcome('BLOCKED','SIX_ASSET_COST_OWNER_RATES_REQUIRED')
    if 'issuer_owner' not in config or not owner_ok(config['issuer_owner']):
        return outcome('BLOCKED','EXTERNAL_ISSUER_AUTHENTICATION_REQUIRED')
    cost = [config['cost_owner']['rates'][a] for a in ASSETS]
    if any(not 0 <= k < 1 for k in cost):
        return outcome('FAIL','COST_DOMAIN_INVALID')
    config={**config,'cost_rates6':cost,'issued_records':parsed['records']}
    n = panel['n']
    if len(cov_by_week)!=n or any(len(shape_by_week.get(m,[]))!=n for m in ALL_METHODS):
        return outcome('BLOCKED','MATCHED_SIX_STRATEGY_ASOF_SHAPE_COV_HISTORY_REQUIRED')
    if not isinstance(config.get('scenario_by_week'),list) or len(config['scenario_by_week'])!=n:
        return outcome('BLOCKED','ISSUED_SCENARIOS_FOR_EVERY_WEEK_REQUIRED')
    # No future-looking full-DEV covariance is accepted as issued input.
    if config.get('covariance_mode')!='ISSUED_ASOF_VERIFIED':
        return outcome('BLOCKED','FULL_DEV_FITTED_COVARIANCE_NOT_ISSUED_ASOF')
    per_model={}
    for m in ALL_METHODS:
        if m not in config['models']:
            per_model[m]=outcome('BLOCKED','MODEL_OWNER_CONFIG_REQUIRED')
            continue
        nav = 1.
        pre = np.r_[np.zeros(5),1.]
        baseline_before={}
        history=[]
        blocked=None
        for t in range(n):
            baseline_before[t]=(float(nav),pre.copy())
            step=_continuation_weights(m,t,pre,nav,panel,shape_by_week,cov_by_week,config)
            if step['status']!='PASS':
                blocked={'week_index':t,'week_start':panel['starts'][t].isoformat(),
                         'status':step['status'],'reason':step.get('reason')}
                break
            history.append({'week_start':panel['starts'][t].isoformat(),
                            'week_end':panel['ends'][t].isoformat(),
                            'nav_pre':step['nav_pre'],'nav_post':step['nav'],
                            'pre_weights':step['pre'],'target_weights':step['target'],
                            'post_weights':step['post'],'zone':step['zone'],
                            'regime':step['regime'],'action':step['action'],
                            'risky_share_selected':step['risky_share_selected'],
                            'risky_share_post':step['risky_share_post'],
                            'gross_return':step['gross_return'],'net_return':step['ret'],
                            'cost_fraction':step['cost_fraction'],'cost_money':step['cost'],
                            'turnover':step['turnover'], 'risk':step['risk'],
                            'source_tape_sha':step['source_tape_sha'],
                            'issue_at':step['issued_action_at'],
                            'owner_risk_hash':step['owner_risk_hash'],
                            'owner_cap_hash':step['owner_cap_hash'],
                            'owner_cost_hash':step['owner_cost_hash'],
                            'cap_violation_pre':step['cap_violation_pre'],
                            'risk_violation_pre':step['risk_violation_pre'],
                            'post_cash_cap':step['post_cash_cap'],
                            'bottom_squeeze_by_asset':step['bottom_squeeze_by_asset'],
                            'new_buy_permission':step['new_buy_permission'],
                            'selected_risky_cap_headroom':step['selected_risky_cap_headroom'],
                            'post_asset_cap_headroom':step['post_asset_cap_headroom']})
            nav,pre=step['nav'],step['next_pre']
        if blocked:
            per_model[m]=outcome('BLOCKED','FULL_HISTORY_ISSUED_REPLAY_FAILED',first_blocker=blocked,
                                 valid_weeks=len(history))
            continue
        events=[]
        # Only an actually paired Short->Bottom event needs the special
        # full-C1 funding permission. Never convert a missing approval into
        # an optimistic avoided-loss estimate or silently cap cash at <100%.
        freeze_gate=validate_short_cash_freeze(config) if parsed['episodes'] else None
        for ep in parsed['episodes']:
            if freeze_gate['status']!='PASS':
                events.append(outcome('BLOCKED','SHORT_CASH_COUNTERFACTUAL_INFEASIBLE',
                                      short_id=ep['short_id'],detail=freeze_gate))
                continue
            begin,end=ep['short_index'],ep['bottom_index']
            if begin>=end:
                continue
            nav0,pre0=baseline_before[begin]
            # Existence of a model-specific risk policy for counterfactual required.
            wealth_freeze, wealth_invest=nav0,nav0
            pre_freeze, pre_invest=pre0.copy(),pre0.copy()
            fee_freeze,fee_invest=0.,0.
            trade_freeze,trade_invest=0.,0.
            stopped=None
            weekly_attribution=[]
            cumulative_market_difference=0.
            cumulative_cost_drag_difference=0.
            for t in range(begin,end):
                # Freeze: cash at Short, then keep 100% C1. Cost-once.
                freeze_nav_before=wealth_freeze
                invest_nav_before=wealth_invest
                cash=np.r_[np.zeros(5),1.]
                cf=TR.self_financing_cost(pre_freeze,cash,np.asarray(cost))
                c=float(cf['cost_fraction'])
                freeze_gross_return=float(cash @ panel['X'][t])
                fee_freeze+=freeze_nav_before*c
                trade_freeze+=cf['turnover']
                wealth_freeze=freeze_nav_before*(1-c)*(1+freeze_gross_return)
                pre_freeze=TR.drift(cash,panel['X'][t])
                # Stay-invested arm follows its OWN benchmark restructure, NOT a buy-and-hold index.
                act=_continuation_weights(m,t,pre_invest,wealth_invest,panel,
                                           shape_by_week,cov_by_week,config,ignore_short_freeze=True)
                if act['status']!='PASS':
                    stopped={'week':t,'reason':act.get('reason')}
                    break
                wealth_invest=act['nav'];pre_invest=act['next_pre']
                fee_invest+=act['cost'];trade_invest+=act['turnover']
                # Exact wealth equation: W' = W + W*r - W*c*(1+r).
                # The market and fee increments are expressed in CURRENCY
                # units on each arm's ACTUAL entering wealth. Their sum is
                # equal to the NAV gap, even after the arms have diverged.
                invested_gross_return=float(act['gross_return'])
                invested_fee=float(act['cost_fraction'])
                market_difference=(freeze_nav_before*freeze_gross_return
                                   -invest_nav_before*invested_gross_return)
                cost_drag_difference=(freeze_nav_before*c*(1+freeze_gross_return)
                                      -invest_nav_before*invested_fee*(1+invested_gross_return))
                incremental_gap=((wealth_freeze-wealth_invest)
                                 -(freeze_nav_before-invest_nav_before))
                if not np.isclose(incremental_gap,market_difference-cost_drag_difference,
                                  rtol=1e-10,atol=1e-10*max(1.,abs(freeze_nav_before),abs(invest_nav_before))):
                    stopped={'week':t,'reason':'EXACT_NET_AVOIDED_LOSS_ATTRIBUTION_IDENTITY_FAILED'}
                    break
                cumulative_market_difference+=market_difference
                cumulative_cost_drag_difference+=cost_drag_difference
                weekly_attribution.append({'week_start':panel['starts'][t].isoformat(),
                                           'week_end':panel['ends'][t].isoformat(),
                                           'freeze_nav':float(wealth_freeze),
                                           'invested_nav':float(wealth_invest),
                                           'invested_action':act['action'],
                                           'invested_post_weights':act['post'],
                                           'invested_risky_share':act['risky_share_post'],
                                           'freeze_gross_return':freeze_gross_return,
                                           'invested_gross_return':invested_gross_return,
                                           'market_difference_money':float(market_difference),
                                           'cost_drag_difference_money':float(cost_drag_difference),
                                           'cumulative_market_difference_money':float(cumulative_market_difference),
                                           'cumulative_cost_drag_difference_money':float(cumulative_cost_drag_difference),
                                           'incremental_avoided_money':float(incremental_gap),
                                           'cumulative_avoided_fraction':float((wealth_freeze-wealth_invest)/nav0)})
            if stopped:
                events.append(outcome('BLOCKED','COUNTERFACTUAL_RESTRUCTURE_FAILED',
                                      short_id=ep['short_id'],detail=stopped))
                continue
            avoided=(wealth_freeze-wealth_invest)/nav0
            explained=(cumulative_market_difference-cumulative_cost_drag_difference)/nav0
            if not np.isclose(avoided,explained,rtol=1e-10,atol=1e-10):
                events.append(outcome('FAIL','CUMULATIVE_AVOIDED_LOSS_IDENTITY_FAILED',short_id=ep['short_id']))
                continue
            events.append(outcome('PASS',short_id=ep['short_id'],
                                  last_short_week=panel['starts'][begin].isoformat(),
                                  first_bottom_week=panel['starts'][end].isoformat(),
                                  weeks=end-begin, assets=ep['assets'],
                                  initial_nav=nav0,freeze_nav_at_bottom=wealth_freeze,
                                  invested_nav_at_bottom=wealth_invest,
                                  avoided_loss_fraction=avoided,
                                  missed_gain_fraction=max(0.,-avoided),
                                  market_difference_fraction=float(cumulative_market_difference/nav0),
                                  cost_drag_difference_fraction=float(cumulative_cost_drag_difference/nav0),
                                  attribution_residual_fraction=float(avoided-explained),
                                  identity='AVOIDED_LOSS=MARKET_DIFFERENCE-COST_DRAG_DIFFERENCE',
                                  freeze_cost_money=fee_freeze,invested_cost_money=fee_invest,
                                  freeze_turnover=trade_freeze,invested_turnover=trade_invest,
                                  weekly_attribution=weekly_attribution,
                                  cash_carry_included=True,
                                  cash_freeze_owner_hash=freeze_gate['owner_cash_hash'],
                                  cash_freeze_uses_C1_proxy=True,
                                  bottom_reentry_cost_excluded_from_pre_bottom_comparison=True))
        completed=[e for e in events if e.get('status')=='PASS']
        positive=sum(e['avoided_loss_fraction']>1e-12 for e in completed)
        negative=sum(e['avoided_loss_fraction']< -1e-12 for e in completed)
        near_zero=len(completed)-positive-negative
        per_model[m]=outcome('PASS',n_weeks=n,terminal_nav=nav,n_events=len(events),
                             event_ledger=events,history=history,
                             avoided_loss_event_summary={
                                 'status':('PASS' if completed else 'NOT_EVALUATED'),
                                 'paired_event_denominator':len(events),
                                 'evaluated_paired_events':len(completed),
                                 'positive_avoided_loss':positive,
                                 'negative_missed_gain':negative,
                                 'near_zero_avoided_loss':near_zero,
                                 'blocked_paired_events':len(events)-len(completed),
                                 'censored_unpaired_events':len(parsed.get('censored_episodes',[])),
                                 'independent_episode_count':'NOT_IDENTIFIABLE',
                                 'selection_note':'ISSUED_SHORT_BOTTOM_PAIRS_ONLY_NO_OOS_VALIDATION'},
                             censored_short_ids=parsed.get('superseded_short_ids',[]),
                             censored_event_ledger=parsed.get('censored_episodes',[]),
                             event_support=parsed.get('event_support'),
                             unmatched_latest_short_id=parsed.get('unmatched_latest_short_id'),
                             censored_short_values='NOT_EVALUATED_NO_BOTTOM_EVENT',
                             claim='CONDITIONAL_OFFICIAL_PENDING_EXTERNAL_ISSUER_AUTHENTICATION')
    all_complete=all(d['status']=='PASS' and all(e['status']=='PASS' for e in d.get('event_ledger',[]))
                     for d in per_model.values())
    matched_net={}
    if all_complete:
        for m,entry in per_model.items():
            h=entry['history']
            path=np.array([1.]+[float(v['nav_post']) for v in h],float)
            peak=np.maximum.accumulate(path)
            actions={a:sum(q['action']==a for q in h) for a in
                     ('HOLD','PARTIAL','FULL','CASH')+(('NO_TRADE_LOCKED',) if any(
                         q['action']=='NO_TRADE_LOCKED' for q in h) else ())}
            regimes={q:sum(v['regime']==q for v in h) for q in REGIMES}
            risk_summary=matched_risk_tolerance_summary(h,config['risk_owner']['caps'][m],
                        risk_basis=config['risk_owner']['risk_basis'])
            if risk_summary['status']!='PASS':
                return outcome('NOT_EVALUATED','MATCHED_ISSUED_RISK_SUMMARY_BLOCKED',
                               model=m,detail=risk_summary,models=per_model)
            matched_net[m]={'terminal_wealth_net':float(path[-1]),
                            'risk_tolerance_by_regime':risk_summary['by_regime'],
                            'risk_cap_repair_week_count':sum(q['cap_violation_pre'] for q in h),
                            'tail_risk_repair_week_count':sum(q['risk_violation_pre'] for q in h),
                            'max_drawdown':float(np.max(1.-path/peak)),
                            'total_turnover':float(sum(v['turnover'] for v in h)),
                            'total_explicit_cost_money':float(sum(v['cost_money'] for v in h)),
                            'avoided_loss_event_summary':entry['avoided_loss_event_summary'],
                            'action_counts':actions,'regime_counts':regimes,
                            'weeks':len(h),'starting_nav':1.,
                            'scope':'STRUCTURAL_ASOF_REPLAY_NOT_EXTERNAL_ISSUER_CERTIFICATION'}
    return outcome('PASS' if all_complete else 'NOT_EVALUATED',
                   None if all_complete else 'ONE_OR_MORE_STRATEGIES_OR_EVENTS_BLOCKED',
                   models=per_model,tape_sha=panel['tape_sha'],matched_net=matched_net,
                   event_support=parsed.get('event_support',{}),
                   avoided_loss_population='PAIRED_EVENTS_ONLY_CENSORED_NOT_IMPUTED',
                   avoided_loss_claim=('NOT_EVALUATED_ZERO_PAIRED_SHORT_BOTTOM_EVENTS'
                                       if not parsed['episodes'] else
                                       'CONDITIONAL_TO_VERIFIED_ISSUED_AND_OWNER_EVIDENCE'),
                   comparison='SAME_SHORT_NAV_CASH_FREEZE_VS_OWN_BENCHMARK_RESTRUCTURING')



def validate_asof_frames(frames: list[dict], panel: dict,
                         *, issued_records: list[dict] | None = None) -> dict:
    """Validate externally supplied weekly 6-strategy risk feeds without fitting.

    Frame schema for each week t:
      week_start, issue_at, covariance5, cov_issued_at, cov_source_id,
      scenarios6, scenario_issued_at, scenario_source_id,
      shapes5: {EW,IVOL,MVP,ERC,MaxDiv,POLICY: {shape,issued_at,source_id}}.
    All required sources must exist by issue_at, before the coming return window.
    Reused (source_id, issued_at) cannot silently change payload or shape.
    Verification is structural/content integrity, NOT issuer authentication.
    """
    if panel.get('status')!='PASS' or len(frames)!=panel['n']:
        return outcome('BLOCKED','ONE_ASOF_FRAME_PER_CANONICAL_DEV_WEEK_REQUIRED')
    if issued_records is not None and len(issued_records)!=panel['n']:
        return outcome('BLOCKED','ISSUED_ASOF_WEEK_MATCH_REQUIRED')
    covs,scenarios,masses,mass_modes,shape_by_week=[],[],[],[],{m:[] for m in ALL_METHODS}
    provenance={}
    for t,(f,start) in enumerate(zip(frames,panel['starts'])):
        if not isinstance(f,dict):
            return outcome('FAIL','ASOF_FRAME_MUST_BE_MAPPING',index=t)
        if f.get('week_start')!=start.isoformat():
            return outcome('FAIL','ASOF_FRAME_WEEK_KEY_MISMATCH',index=t)
        issued=f.get('issue_at')
        if not issued:
            return outcome('BLOCKED','ASOF_FRAME_ISSUE_MISSING',index=t)
        try:
            dt=stamp(issued)
            if issued_records is not None:
                issued_packet=issued_records[t]
                if issued_packet.get('week_start')!=start.isoformat():
                    return outcome('FAIL','ISSUED_PACKET_ASOF_KEY_MISMATCH',index=t)
                packet_time=stamp(issued_packet.get('issue_at'))
                if dt>packet_time:
                    return outcome('FAIL','ASOF_FRAME_FUTURE_RELATIVE_TO_ACTION_DECISION',index=t)
            if dt>=datetime.combine(start,datetime.min.time(),tzinfo=timezone.utc):
                return outcome('FAIL','ASOF_FRAME_ISSUED_AFTER_WEEK_START',index=t)
        except (ValueError,TypeError):
            return outcome('FAIL','ASOF_FRAME_ISSUE_CLOCK_INVALID',index=t)
        for kind,payload_key,issue_key,source_key in (
            ('cov','covariance5','cov_issued_at','cov_source_id'),
            ('scenario','scenarios6','scenario_issued_at','scenario_source_id')):
            if not f.get(issue_key) or not f.get(source_key) or payload_key not in f:
                return outcome('BLOCKED','ASOF_RISK_SOURCE_CLOCK_OR_PAYLOAD_MISSING',index=t,kind=kind)
            try:
                if stamp(f[issue_key])>dt:
                    return outcome('FAIL','FUTURE_RISK_SOURCE',index=t,kind=kind)
            except (TypeError,ValueError):
                return outcome('FAIL','RISK_SOURCE_CLOCK_INVALID',index=t,kind=kind)
            key=(kind,f[source_key],f[issue_key])
            digest=canonical_sha256({'matrix':f[payload_key],
                                      'mass':f.get('scenario_mass') if kind=='scenario' else None})
            if key in provenance and provenance[key]!=digest:
                return outcome('FAIL','RISK_SOURCE_ID_VINTAGE_PAYLOAD_CONFLICT',index=t,kind=kind)
            provenance[key]=digest
        try:
            cov=np.asarray(f['covariance5'],dtype=float)
            cov=_covariance(cov,5,positive=True)
            scen=np.asarray(f['scenarios6'],dtype=float)
            if scen.ndim!=2 or scen.shape[1]!=6 or len(scen)<2 or not np.isfinite(scen).all() or np.any(scen<=-1):
                raise ValueError('SCENARIOS_INVALID')
            mass_is_issued=('scenario_mass' in f)
            mass=_scenario_mass(f.get('scenario_mass'),len(scen))
            # The mass belongs to the SAME scenario source and publication clock.
            # It cannot be changed under an old source-vintage identity.
            masses.append(mass)
            mass_modes.append('WEIGHTED_ISSUED' if mass_is_issued else 'UNIFORM_ISSUED')
            if not isinstance(f.get('shapes5'),dict):
                return outcome('BLOCKED','SIX_STRATEGY_ASOF_SHAPE_MAP_REQUIRED',index=t)
            for m in ALL_METHODS:
                record=f['shapes5'].get(m)
                if not _asof_record(record,issued,('shape',)):
                    return outcome('BLOCKED','ISSUED_MODEL_SHAPE_MISSING',index=t,model=m)
                w=_simplex(record['shape'],5)
                key=('shape',record['source_id'],record['issued_at'])
                digest=canonical_sha256(w.tolist())
                if key in provenance and provenance[key]!=digest:
                    return outcome('FAIL','MODEL_SHAPE_SOURCE_VINTAGE_CONFLICT',index=t,model=m)
                provenance[key]=digest
                shape_by_week[m].append(w)
        except (ValueError,KeyError,TypeError) as e:
            return outcome('FAIL','INVALID_ASOF_COV_SCENARIO_OR_SHAPE',index=t,detail=str(e))
        covs.append(cov)
        scenarios.append(scen)
    return outcome('PASS',covariances=np.asarray(covs),scenarios=scenarios,
                   scenario_masses=masses,scenario_weight_modes=mass_modes,
                   shapes=shape_by_week,source_vintages=len(provenance),
                   scope='STRUCTURALLY_VERIFIED_ASOF_NOT_EXTERNALLY_AUTHENTICATED')

def risk_only_dev(dataset_root) -> dict:
    """Perform 5-shape research frontier on genuine 781-week tape; POLICY blocked.
    Full-DEV covariance and scenario empirical distribution are retrospective
    (DEV is not an independent validation set). This output selects NO threshold.
    """
    pan=load_locked_dev(dataset_root)
    if pan['status']!='PASS':
        return pan
    s=np.cov(pan['X'][:,:5],rowvar=False)
    shapes=benchmark_vectors(s)
    n={}
    for regime in REGIMES:
        # Identical unconditional empirical pool: allocation stress test, NOT regime VaR forecast.
        fr=risk_frontier(pan['X'],shapes,regime)
        n[regime]={m: ({'status':d['status'],'reason':d.get('reason')} if d['status']!='PASS' else
                          [{'a':r['a'],'VaR99':r['risk']['0.99']['VaR'],'ES99':r['risk']['0.99']['ES']}
                           for r in d['frontier']]) for m,d in fr['models'].items()}
    return outcome('PASS',weekly_rows=pan['n'], first_week_end=pan['ends'][0].isoformat(),
                   last_week_end=pan['ends'][-1].isoformat(),
                   tape_sha=pan['tape_sha'], risky_share_frontiers=n,
                   no_approved_var_es_budget=True,no_official_policy_values=True,
                   no_issued_short_bottom_events=True,
                   scenario_pool='UNCONDITIONAL_FULL_DEV_NOT_REGIME_CONDITIONAL',
                   claim=RESEARCH)



def _synthetic_six_strategy_fixture(*, with_events: bool = True, r23_main_only_lock: bool = False):
    """Seven-week six-model fixture (one Short at week 2, Bottom E1 at week 5, EXIT at week 6).

    with_events=False keeps the same regimes and returns but removes every Short/Bottom/EXIT record."""
    starts=[date(2005,1,10)+timedelta(weeks=i) for i in range(7)]
    ends=[s+timedelta(days=4) for s in starts]
    panel={'status':'PASS','n':len(starts),'starts':starts,'ends':ends,
           'X':np.tile([.012,.011,.013,.003,.002,.0002],(7,1)),
           'tape_sha':'SYNTHETIC_FIXTURE_NOT_CANONICAL'}
    records=[]
    for i,(start,end) in enumerate(zip(starts,ends)):
        issued=(start-timedelta(days=1)).isoformat()+'T12:00:00+00:00'
        reg='N' if i<2 or i>=6 else 'BOTH'
        r={'week_start':start.isoformat(),'week_end':end.isoformat(),
           'issue_at':issued,'regime':reg,
           'regime_source_id':f'fixture-rq01-{i}','regime_issued_at':issued,
           'short_confirmed':with_events and i==2,'short_id':'S2' if (with_events and i==2) else None,
           'short_issued_at':issued if (with_events and i==2) else None,
           'bottom_permissions':{},'first_n_confirmed':with_events and i==6,
           'exit_id':'X6' if (with_events and i==6) else None,
           'exit_issued_at':issued if (with_events and i==6) else None,
           'exit_for_short_id':'S2' if (with_events and i==6) else None}
        if with_events and i==5:
            r['bottom_start']=start.isoformat()
            r['bottom_permissions']={'E1':{
               'authorized':True,'asset':'E1','issued_at':issued,'source_id':'fixture-bottom-E1',
               'committee_id':'fixture-committee','risk_caps_owner_hash':'fixture-caps',
               'zone_start':start.isoformat(), 'zone_end':(start+timedelta(days=6)).isoformat(),
               'funding_owner':{'approved':True,'owner_hash':'fixture-funding'},
               'eligibility_owner':{'approved':True,'owner_hash':'fixture-eligibility','assets':['E1']},
               'reentry_policy_owner':{'approved':True,'owner_hash':'fixture-reentry'},
               'risk_check':{'status':'PASS','within_caps':True,
                             'issued_at':issued,'source_id':'fixture-risk-check',
                             'risk_caps_owner_hash':'fixture-caps'},
               'committee_rule_owner':{'approved':True,'owner_hash':'rule','k_independent':2},
               'committee_rule_owner_hash':'rule',
               'factor_evidence':{f:{'vote':True,'input_ids':[f],
                                      'issued_at':issued,'source_id':'fixture-'+f}
                    for f in ('EWI','COUNTRY','RISK_COV')}}}
        records.append(r)
    parsed=parse_issued_path(records,panel,active_cap_owner_hash='fixture-caps')
    assert parsed['status']=='PASS' and len(parsed['episodes'])==(1 if with_events else 0)
    shapes={key:[np.full(5,.2) for _ in starts] for key in ALL_METHODS}
    covariance=np.asarray([np.eye(5)*.0004 for _ in starts])
    caps={a:1. for a in ASSETS}
    limits={str(a):{'VaR':1.,'ES':1.} for a in ALPHAS}
    owner={'risk_owner':{'approved':True,'owner_hash':'fixture-risk',
                         'risk_unit':'DECIMAL_LOSS_FRACTION_OF_NAV',
                         'risk_basis':'GROSS_WEEKLY_LOSS',
                         'caps':{m:{r:limits for r in REGIMES} for m in ALL_METHODS}},
           'asset_cap_owner':{'approved':True,'owner_hash':'fixture-caps',
                              'unit':'FRACTION_OF_NAV','caps':caps},
           'cash_owner':{'approved':True,'owner_hash':'fixture-cash',
                         'asset_id':'C1','counterfactual_freeze_approved':True,
                         'cash_proxy_funding_approved':True},
           'cost_owner':{'approved':True,'owner_hash':'fixture-cost',
                         'unit':'PROPORTIONAL_TRADE_VALUE',
                         'rates':{a:.0001 for a in ASSETS}},
           'issuer_owner':{'approved':True,'owner_hash':'fixture-issuer'},
           'scenario_probability_owner':{'approved':True,'owner_hash':'fixture-prob',
                                         'mode':'UNIFORM_ISSUED'},
           'capital_objective_owner':{'approved':True,'owner_hash':'fixture-objective',
                                      'objective':'MAX_FEASIBLE_RISKY_SHARE'},
           'ew_definition_owner':{'approved':True,'owner_hash':'fixture-ew','definition':'EW5_RISKY_SLEEVE'},
           'covariance_mode':'ISSUED_ASOF_VERIFIED',
           **({'r23_main_only_lock':True} if r23_main_only_lock else {}),
           'scenario_by_week':[np.tile([.012,.011,.013,.004,.003,.0001],(100,1))
                               for _ in starts],
           'scenario_mass_by_week':[np.full(100,.01) for _ in starts],
           'scenario_weight_mode_by_week':['UNIFORM_ISSUED' for _ in starts],
           'models':{}}
    for method in ALL_METHODS:
        owner['models'][method]={
           'base_caps':{a:1. for a in ASSETS[:5]},
           'defense':{a:0. for a in ASSETS[:5]},
           'allowance':{a:0. for a in ASSETS[:5]},
           'asset_caps':{a:1. for a in ASSETS[:5]},
           'decay_tau':10.,
           'tolerance_owner':{'approved':True,'owner_hash':'fixture-tolerance',
                              'model':method,'value':0.,'capital_tolerance':0.,
                              'reference_geometry_model':'EW',
                              'geometry':B.GEOMETRY[method if method!='POLICY' else 'EW']},
           'utility_owner':{'approved':True,'owner_hash':'fixture-utility',
                            'mode':'LOG','hurdle':0.,'emergency_repair_approved':True,
                            'short_cash_liquidation_approved':True,
                            'normal_floor_activation_approved':True}}
    return panel,parsed,shapes,covariance,owner


def _synthetic_six_strategy_lifecycle_test() -> dict:
    """Executable law-of-motion test only. NOT evidence for historical NET."""
    panel,parsed,shapes,covariance,owner=_synthetic_six_strategy_fixture()
    output=short_to_bottom_counterfactual(panel,parsed,shapes,covariance,owner)
    assert output['status']=='PASS'
    # The same issued fixture and return tape must NOT deliver an event-level
    # avoided-loss result if the cash funding owner has not authorized C1.
    # Baseline policy NAVs may still be computed, but this paired comparison U.
    from copy import deepcopy
    revoked=deepcopy(owner)
    revoked['cash_owner']['counterfactual_freeze_approved']=False
    infeasible=short_to_bottom_counterfactual(panel,parsed,shapes,covariance,revoked)
    assert infeasible['status']=='NOT_EVALUATED',infeasible
    assert all(e['reason']=='SHORT_CASH_COUNTERFACTUAL_INFEASIBLE'
               for d in infeasible['models'].values()
               for e in d.get('event_ledger',[])),infeasible
    for method,record in output['models'].items():
        assert record['status']=='PASS' and len(record['event_ledger'])==1
        assert record['avoided_loss_event_summary']['paired_event_denominator']==1
        assert record['avoided_loss_event_summary']['negative_missed_gain']==1
        assert record['avoided_loss_event_summary']['positive_avoided_loss']==0
        assert output['matched_net'][method]['risk_tolerance_by_regime']['N']['status']=='PASS'
        assert output['matched_net'][method]['risk_tolerance_by_regime']['D']['status']=='NOT_EVALUATED'
        assert output['matched_net'][method]['risk_tolerance_by_regime']['BOTH']['weeks']==4
        e=record['event_ledger'][0]
        assert e['status']=='PASS' and e['weeks']==3
        assert np.isclose(e['avoided_loss_fraction'],
                          (e['freeze_nav_at_bottom']-e['invested_nav_at_bottom'])/e['initial_nav'])
        assert e['avoided_loss_fraction']<0  # missed gains must not be censored.
        assert np.isclose(e['avoided_loss_fraction'],
                          e['market_difference_fraction'] - e['cost_drag_difference_fraction'],
                          rtol=0,atol=1e-12)
        assert len(e['weekly_attribution'])==e['weeks']
        assert np.isclose(sum(w['market_difference_money'] for w in e['weekly_attribution']),
                          e['market_difference_fraction']*e['initial_nav'],atol=1e-12)
        assert np.isclose(sum(w['cost_drag_difference_money'] for w in e['weekly_attribution']),
                          e['cost_drag_difference_fraction']*e['initial_nav'],atol=1e-12)
        assert e['attribution_residual_fraction'] < 1e-12
    return {'status':'PASS','models':6,'episodes':1,'weeks':7,
            'claim':'SYNTHETIC_LOGIC_NOT_REAL_NET_PERFORMANCE'}



def _rev2_guard_self_test() -> dict:
    """Portable independent fail-closed examples; no private dataset required."""
    sq={'status':'PASS','asset_caps':{a:1. for a in ASSETS[:5]},
        'new_buy':{a:False for a in ASSETS[:5]}}
    toler={'approved':True,'owner_hash':'fixture-t','model':'EW',
           'geometry':B.GEOMETRY['EW'],'value':0.,'capital_tolerance':0.}
    util={'approved':True,'owner_hash':'fixture-u','mode':'LOG','hurdle':0.,
          'emergency_repair_approved':True}
    x=benchmark_restructure('EW',np.array([.8,0,0,0,0,.2]),[1.,0,0,0,0],np.eye(5),
            regime='BOTH',zone='STRESS_PRE_CONFIRMATION',capital_share=.5,
            squeeze=sq,threshold_owner=toler,utility_owner=util,
            scenarios6=np.tile([.05,0,0,0,0,0],(20,1)),cost_rates6=np.zeros(6),nav=1.)
    assert x['status']=='PASS' and sum(x['post_weights'][:5])<=.5000000001
    util['emergency_repair_approved']=False
    y=benchmark_restructure('EW',np.array([.8,0,0,0,0,.2]),[1.,0,0,0,0],np.eye(5),
            regime='BOTH',zone='STRESS_PRE_CONFIRMATION',capital_share=.5,
            squeeze=sq,threshold_owner=toler,utility_owner=util,
            scenarios6=np.tile([.05,0,0,0,0,0],(20,1)),cost_rates6=np.zeros(6),nav=1.)
    assert y['status']=='BLOCKED'
    fake={'E1':{'authorized':True,'asset':'E1','source_id':'fake','committee_id':'fake',
                 'issued_at':'2009-01-01T00:00:00+00:00','risk_caps_owner_hash':'h',
                 'three_factor_structurally_validated':True,'zone_start':'2009-01-01',
                 'zone_end':'2009-01-08'}}
    k={a:.3 for a in ASSETS[:5]};zero={a:0. for a in k}
    z=bottom_risk_squeeze('BOTH','BOTTOM_REENTRY_ZONE',base_caps=k,defense=zero,
           allowance=k,owner_caps=k,permission=fake,
           decision_at='2009-01-02T00:00:00+00:00',active_risk_cap_hash='h')
    assert z['status']=='PASS' and z['new_buy']['E1'] is False
    caps={'approved':True,'owner_hash':'cap','unit':'FRACTION_OF_NAV',
          'caps':{a:1. for a in ASSETS}}
    fees={'approved':True,'owner_hash':'fees','unit':'PROPORTIONAL_TRADE_VALUE',
          'rates':{a:0. for a in ASSETS}}
    risk={'approved':True,'owner_hash':'risk','risk_basis':'NET_AFTER_SELF_FINANCING_COST',
          'risk_unit':'DECIMAL_LOSS_FRACTION_OF_NAV',
          'caps':{'EW':{q:{str(a):{'VaR':1.,'ES':1.} for a in ALPHAS} for q in REGIMES}}}
    u={'approved':True,'owner_hash':'util','mode':'LOG','hurdle':0.}
    shape=np.array([1.,0,0,0,0]);pre=np.r_[np.zeros(5),1.]
    good=net_capital_utility_choice(np.tile([.1,0,0,0,0,0],(30,1)),shape,pre,
              'EW','N',nav=1.,risk_owner=risk,asset_cap_owner=caps,cost_owner=fees,utility_owner=u)
    bad=net_capital_utility_choice(np.tile([-.1,0,0,0,0,0],(30,1)),shape,pre,
              'EW','BOTH',nav=1.,risk_owner=risk,asset_cap_owner=caps,cost_owner=fees,utility_owner=u)
    assert good['status']=='PASS' and good['risky_share']==.75
    assert bad['status']=='PASS' and bad['risky_share']==0.
    start=date(2005,1,10)
    late=validate_asof_frames([{'week_start':start.isoformat(),
                    'issue_at':'2005-01-09T22:00:00+00:00'}],
               {'status':'PASS','n':1,'starts':[start]},
               issued_records=[{'week_start':start.isoformat(),
                                'issue_at':'2005-01-09T12:00:00+00:00'}])
    assert late['reason']=='ASOF_FRAME_FUTURE_RELATIVE_TO_ACTION_DECISION'
    # Friday 12:00 is not post-Friday-close; Sunday 12:00 is admissible.
    assert not _issue_before_window({'issue_at':'2005-01-14T12:00:00+00:00'},
                                      date(2005,1,17),date(2005,1,14))
    assert _issue_before_window({'issue_at':'2005-01-16T12:00:00+00:00'},
                                  date(2005,1,17),date(2005,1,14))
    return {'status':'PASS','new_falsification_cases':7,
            'claim':'SYNTHETIC_GUARD_REGRESSION_NOT_PORTFOLIO_NET_EVIDENCE'}

def _rev3_guard_self_test() -> dict:
    """Tests provenance causality, window clocks and exact cost decomposition.

    All values are synthetic, with no issuer authentication or production claim.
    """
    issue='2009-01-04T12:00:00+00:00'
    earlier='2009-01-04T08:00:00+00:00'
    later='2009-01-04T14:00:00+00:00'
    base={
        'authorized':True,'asset':'E1','issued_at':issue,'source_id':'p1',
        'risk_caps_owner_hash':'c','committee_id':'committee',
        'zone_start':'2009-01-05','zone_end':'2009-01-09',
        'funding_owner':{'approved':True,'owner_hash':'f'},
        'eligibility_owner':{'approved':True,'owner_hash':'e','assets':['E1']},
        'reentry_policy_owner':{'approved':True,'owner_hash':'r'},
        'risk_check':{'status':'PASS','within_caps':True,'risk_caps_owner_hash':'c',
                      'issued_at':earlier,'source_id':'risk'},
        'committee_rule_owner':{'approved':True,'owner_hash':'rule','k_independent':2},
        'committee_rule_owner_hash':'rule',
        'factor_evidence':{f:{'vote':True,'input_ids':[f],
                              'source_id':f,'issued_at':earlier}
                           for f in ('EWI','COUNTRY','RISK_COV')}
    }
    from copy import deepcopy
    action='2009-01-04T16:00:00+00:00'
    assert _bottom_permission_proof('E1',base,action,'c','2009-01-05')
    future_vote=deepcopy(base);future_vote['factor_evidence']['COUNTRY']['issued_at']=later
    assert not _bottom_permission_proof('E1',future_vote,action,'c','2009-01-05')
    future_risk=deepcopy(base);future_risk['risk_check']['issued_at']=later
    assert not _bottom_permission_proof('E1',future_risk,action,'c','2009-01-05')
    future_perm=deepcopy(base);future_perm['issued_at']='2009-01-06T12:00:00+00:00'
    assert not _bottom_permission_proof('E1',future_perm,'2009-01-06T16:00:00+00:00',
                                         'c','2009-01-05')
    assert _bottom_permission_proof('E1',base,action,'c','2009-01-09')
    assert not _bottom_permission_proof('E1',base,action,'c','2009-01-10')
    # Two distinct NAV paths: exact attribution is not just return spread
    # because gross market impact is measured on each arm's actual capital.
    wf=wi=1.;market=drag=0.
    observations=[(.002,-.06,.002,.004),(.002,.04,0.,.003)]
    for rf,ri,cf,ci in observations:
        before_f,before_i=wf,wi
        wf=before_f*(1-cf)*(1+rf)
        wi=before_i*(1-ci)*(1+ri)
        market+=before_f*rf-before_i*ri
        drag+=before_f*cf*(1+rf)-before_i*ci*(1+ri)
    assert abs((wf-wi)-(market-drag)) < 1e-12
    return {'status':'PASS','falsification_cases':7,
            'scope':'SYNTHETIC_PROVENANCE_AND_ACCOUNTING_ONLY'}



def _rev4_guard_self_test() -> dict:
    """Independent six-asset-cap + risk-repair counterexamples, NOT a performance test."""
    sq={'status':'PASS','asset_caps':{a:1. for a in ASSETS[:5]},
        'new_buy':{a:True for a in ASSETS[:5]}}
    tol={'approved':True,'owner_hash':'test-thr','model':'EW',
         'geometry':B.GEOMETRY['EW'],'value':10.,'capital_tolerance':10.}
    own={'approved':True,'owner_hash':'test-util','mode':'LOG','hurdle':0.,
         'emergency_repair_approved':True,'short_cash_liquidation_approved':True}
    pre=np.r_[np.full(5,.1),.5]
    cov=np.eye(5)*.002
    benign=np.tile([.002,.002,.002,.002,.002,0.],(200,1))
    stress=np.tile([-.5,-.5,-.5,-.5,-.5,0.],(200,1))
    args=dict(model='EW',pre=pre,shape5=np.full(5,.2),covariance5=cov,
              regime='BOTH',zone='STRESS_PRE_CONFIRMATION',capital_share=.5,
              squeeze=sq,threshold_owner=tol,utility_owner=own,
              cost_rates6=np.zeros(6),nav=1.)
    # Case 1: HOLD of C1=50% must not pass owner C1 cap 40%.
    capped=benchmark_restructure(**args,scenarios6=benign,cash_cap=.4)
    assert capped['status']=='BLOCKED',capped
    assert capped['reason']=='NO_FEASIBLE_HOLD_OR_RESTRUCTURE_ACTION',capped
    # Case 2: after owner authorizes 50% C1, HOLD is structurally feasible.
    allowed=benchmark_restructure(**args,scenarios6=benign,cash_cap=.5)
    assert allowed['status']=='PASS' and allowed['action']=='HOLD',allowed
    # Case 3: a higher C1 cap alone is not justification to HOLD tail-risk breach.
    allrisk={str(a):{'VaR':.10,'ES':.10} for a in ALPHAS}
    repaired=benchmark_restructure(**args,scenarios6=stress,cash_cap=1.,
                                    risk_limits=allrisk)
    assert repaired['status']=='PASS',repaired
    assert repaired['risk_violation_pre'] is True, repaired
    assert repaired['action']=='CASH' and repaired['post_weights'][5]==1.,repaired
    # Case 4: risk repair requires a distinct emergency owner signature.
    denied=benchmark_restructure(**{**args,'utility_owner':{**own,'emergency_repair_approved':False}},
                                  scenarios6=stress,risk_limits=allrisk)
    assert denied['status']=='BLOCKED' and denied['reason']=='CAP_BREACH_EMERGENCY_REPAIR_OWNER_REQUIRED',denied
    # Case 5: all three alpha budgets are required, no sparse risk policy allowed.
    missing=benchmark_restructure(**args,scenarios6=stress,
             risk_limits={'0.99':{'VaR':.10,'ES':.10}})
    assert missing['status']=='BLOCKED' and missing['reason']=='ALL_THREE_OWNER_VAR_ES_ALPHA_LIMITS_REQUIRED',missing
    # Case 6: six caps cannot sum to less than the NAV simplex.
    infeasible_sq={**sq,'asset_caps':{a:.1 for a in ASSETS[:5]}}
    infeasible=benchmark_restructure(**{**args,'squeeze':infeasible_sq},
                                      scenarios6=benign,cash_cap=.2)
    assert infeasible['status']=='BLOCKED' and infeasible['reason']=='ASSET_AND_CASH_CAP_SIMPLEX_INFEASIBLE',infeasible
    # Case 7: cap of C1=1 permits Short CASH, C1<1 cannot permit Short CASH.
    shargs={**args,'zone':'NO_RISKY_ZONE'}
    cash_ok=benchmark_restructure(**shargs,scenarios6=benign,cash_cap=1.)
    cash_no=benchmark_restructure(**shargs,scenarios6=benign,cash_cap=.8)
    assert cash_ok['status']=='PASS' and cash_ok['action']=='CASH',cash_ok
    assert cash_no['status']=='BLOCKED',cash_no
    # Case 8: even a cash cap >1 must never be admitted.
    bad=benchmark_restructure(**args,scenarios6=benign,cash_cap=1.01)
    assert bad['status']=='BLOCKED' and bad['reason']=='CASH_CAP_OWNER_FRACTION_DOMAIN_INVALID'
    # Case 9: a deliberately breached tail-risk ledger must FAIL rather than
    # still report a matched PASS.
    fake_risk={str(a):{'net_VaR':.2,'net_ES':.3} for a in ALPHAS}
    fake_week={'regime':'N','risky_share_selected':.5,'risky_share_post':.5,
               'net_return':-.01,'risk':fake_risk}
    budgets={'N':{str(a):{'VaR':.1,'ES':.1} for a in ALPHAS}}
    check=matched_risk_tolerance_summary([fake_week],budgets)
    assert check['status']=='FAIL' and check['reason']=='POST_TRADE_ISSUED_RISK_BUDGET_BREACH',check
    # Case 10: no D/G issued weeks means U/NOT_EVALUATED, not fake zero VaR.
    loose={'N':{str(a):{'VaR':1.,'ES':1.} for a in ALPHAS}}
    ok=matched_risk_tolerance_summary([fake_week],loose)
    assert ok['status']=='PASS' and ok['by_regime']['G']['status']=='NOT_EVALUATED',ok
    return {'status':'PASS','new_falsification_cases':10,
            'scope':'SYNTHETIC_SIX_ASSET_CAP_AND_VAR_ES_ACTION_LOGIC_ONLY'}


def _rev5_guard_self_test() -> dict:
    """H0: a full-C1 freeze can be claimed without a feasible funding owner.

    H1: it is blocked unless the exact owner and unit caps permit it.
    Falsifier: lower only C1 to .9 or revoke the cash proxy authority;
    the old pass must become an explicit BLOCKED, without modifying returns.
    """
    from copy import deepcopy
    caps={a:1. for a in ASSETS}
    fees={a:.001 for a in ASSETS}
    fixture={'cash_owner':{'approved':True,'owner_hash':'cash',
                           'asset_id':'C1','counterfactual_freeze_approved':True,
                           'cash_proxy_funding_approved':True},
             'asset_cap_owner':{'approved':True,'owner_hash':'caps',
                                'unit':'FRACTION_OF_NAV','caps':caps},
             'cost_owner':{'approved':True,'owner_hash':'fees',
                            'unit':'PROPORTIONAL_TRADE_VALUE','rates':fees}}
    assert validate_short_cash_freeze(fixture)['status']=='PASS'
    blocked=deepcopy(fixture); blocked['asset_cap_owner']['caps']['C1']=.9
    q=validate_short_cash_freeze(blocked)
    assert q['status']=='BLOCKED' and q['reason']=='C1_FULL_FREEZE_EXCEEDS_OWNER_CASH_CAP',q
    blocked=deepcopy(fixture); blocked['cash_owner']['counterfactual_freeze_approved']=False
    assert validate_short_cash_freeze(blocked)['reason']=='C1_FREEZE_AND_PROXY_FUNDING_AUTHORIZATION_REQUIRED'
    blocked=deepcopy(fixture); blocked['cash_owner']['cash_proxy_funding_approved']=False
    assert validate_short_cash_freeze(blocked)['status']=='BLOCKED'
    blocked=deepcopy(fixture); blocked['cost_owner']['rates']['C1']=float('nan')
    assert validate_short_cash_freeze(blocked)['reason']=='INVALID_C1_FREEZE_COST_SCHEDULE'
    # Impossible nominal masks can never be silently cast to false.
    panel={'status':'PASS','n':1,'starts':[date(2005,1,10)],
           'ends':[date(2005,1,14)]}
    rec={'week_start':'2005-01-10','week_end':'2005-01-14',
         'issue_at':'2005-01-09T12:00:00+00:00','regime':'N',
         'regime_source_id':'r','regime_issued_at':'2005-01-09T12:00:00+00:00',
         'short_confirmed':False,'first_n_confirmed':False,
         'bottom_permissions':{}}
    assert parse_issued_path([None],panel)['reason']=='ISSUED_WEEKLY_PACKET_NOT_MAPPING'
    wrong={**rec,'short_confirmed':'False'}
    assert parse_issued_path([wrong],panel)['reason']=='ISSUED_STATE_FLAGS_MUST_BE_BOOLEAN'
    wrong={**rec,'first_n_confirmed':True,'exit_id':'forged',
           'exit_issued_at':rec['issue_at']}
    assert parse_issued_path([wrong],panel)['reason']=='EXIT_WITHOUT_ACTIVE_ISSUED_SHORT'
    assert parse_issued_path([rec],panel)['status']=='PASS'
    return {'status':'PASS','cases':8,'scope':'CONDITIONAL_OWNER_AND_ISSUED_LOGIC_ONLY'}



def _rev7_guard_self_test() -> dict:
    """Independent adversarial checks for scenario masses and feasible projections."""
    from copy import deepcopy
    owner=lambda suffix,**k:{'approved':True,'owner_hash':'fixture-'+suffix,**k}
    # Weighted mixture ES is neither uniform ES nor a weighted mean of
    # conditional ES; a rare loss changes risk budgets and utility decisions.
    sc=np.zeros((2,6),float)
    sc[:,0]=[.10,-.50]
    shape=np.array([1.,0.,0.,0.,0.]); cash=np.r_[np.zeros(5),1.]
    pr=[.99,.01]
    var_uniform=T.var_es(-sc[:,0],.95)
    var_weighted=T.var_es(-sc[:,0],.95,pr)
    assert var_uniform['ES']>var_weighted['ES']+.1,(var_uniform,var_weighted)
    assert np.isclose(var_weighted['ES'],.02,atol=1e-10),var_weighted
    assert _utility_scores(np.r_[shape,0.],sc,1.,owner('util',mode='LOG'),mass=pr)>           _utility_scores(np.r_[shape,0.],sc,1.,owner('util',mode='LOG'))
    for bad in ([.5,.6],[1.,0.],[-.1,1.1],[np.nan,1.]):
        try:
            _scenario_mass(bad,2)
        except ValueError:
            continue
        raise AssertionError(('BAD_MASS_ACCEPTED',bad))
    assert np.isclose(float(_scenario_mass(None,2).sum()),1.)
    caps=owner('cap',unit='FRACTION_OF_NAV',caps={a:1. for a in ASSETS})
    squeeze=outcome('PASS',asset_caps={a:1. for a in ASSETS[:5]},
                    new_buy={a:a=='E1' for a in ASSETS[:5]})
    e=effective_trade_caps(cash,squeeze,caps)
    assert e['status']=='PASS' and e['caps']['E1']==1. and all(
           e['caps'][a]==0 for a in ASSETS[1:5]),e
    projected={a:e['caps'][a] for a in ASSETS[:5]}
    frontier=risk_frontier(np.tile(sc,(50,1)),{'EW':outcome('PASS',shape=np.full(5,.2))},
                           'D',mass=np.tile(np.asarray(pr)/50,50),
                           projected_caps5=projected)
    sample=frontier['models']['EW']['frontier'][-1]
    assert sample['a']==.5 and abs(sample['executed_risky_fraction']-.1)<1e-10,sample
    assert all(abs(z)<1e-12 for z in sample['weights'][1:5])
    squeeze['new_buy']={a:False for a in ASSETS[:5]}
    e=effective_trade_caps(cash,squeeze,caps)
    assert e['status']=='PASS' and e['max_risky_capacity']==0
    caps['caps']['C1']=.4
    q=effective_trade_caps(cash,squeeze,caps)
    assert q['reason']=='SQUEEZE_CASH_COMBINED_CAP_INFEASIBLE',q
    caps['caps']['C1']=1.
    risk=owner('risk',risk_unit='DECIMAL_LOSS_FRACTION_OF_NAV',
              risk_basis='NET_AFTER_SELF_FINANCING_COST',
              caps={'EW':{'N':{str(a):{'VaR':1.,'ES':1.} for a in ALPHAS}}},
              min_effective_tail_support=.01)
    cost=owner('cost',unit='PROPORTIONAL_TRADE_VALUE',
               rates={a:0. for a in ASSETS})
    u=owner('util',mode='LOG',hurdle=0.)
    sc_safe=np.zeros((200,6),float)
    sc_safe[:,0]=.05
    pre=np.r_[np.zeros(5),1.]
    r=capital_grid_refinement_audit(sc_safe,shape,pre,'EW','N',nav=1.,
                                    risk_owner=risk,asset_cap_owner=caps,
                                    cost_owner=cost,utility_owner=u,
                                    coarse_step=.05,fine_step=.025,
                                    pre_registered_utility_gap=1e-5)
    assert r['status']=='PASS' and r['grids_nested'],r
    assert r['fine_minus_coarse_utility']>=-1e-12
    poor=deepcopy(risk);poor['min_effective_tail_support']=999.
    a=net_capital_utility_choice(sc_safe,shape,pre,'EW','N',nav=1.,risk_owner=poor,
                                  asset_cap_owner=caps,cost_owner=cost,utility_owner=u)
    assert a['reason']=='NO_FEASIBLE_NET_RISK_UTILITY_CAPITAL_ALLOCATION',a
    # Strict action gate: insufficient tail support blocks even HOLD.
    squeeze=outcome('PASS',asset_caps={a:1. for a in ASSETS[:5]},
                    new_buy={a:True for a in ASSETS[:5]})
    p=benchmark_restructure('EW',pre,shape,np.eye(5)*.0004,regime='N',zone='NORMAL',
        capital_share=.5,squeeze=squeeze,
        threshold_owner=owner('thr',model='EW',geometry=B.GEOMETRY['EW'],
                             value=0.,capital_tolerance=0.),
        utility_owner=u,scenarios6=sc_safe,cost_rates6=np.zeros(6),nav=1.,
        risk_limits={str(a):{'VaR':1.,'ES':1.} for a in ALPHAS},
        min_effective_tail_support=999.)
    assert p['reason']=='NO_FEASIBLE_HOLD_OR_RESTRUCTURE_ACTION',p
    # A probability vector is part of the immutable scenario source vintage.
    # Same ID+clock with changed p MUST fail, even if the return matrix matches.
    starts=[date(2005,1,10)+timedelta(weeks=i) for i in range(2)]
    panel={'status':'PASS','n':2,'starts':starts}
    shape_record={'shape':[.2]*5,'issued_at':'2005-01-08T12:00:00+00:00',
                  'source_id':'fixture-static-shape'}
    frames=[]
    for i,begin in enumerate(starts):
        frames.append({'week_start':begin.isoformat(),
                       'issue_at':(begin-timedelta(days=1)).isoformat()+'T12:00:00+00:00',
                       'covariance5':(np.eye(5)*.0004).tolist(),
                       'cov_source_id':'fixture-cov',
                       'cov_issued_at':'2005-01-08T12:00:00+00:00',
                       'scenarios6':sc.tolist(),
                       'scenario_mass':[.99,.01] if i==0 else [.5,.5],
                       'scenario_source_id':'fixture-reused-risk-source',
                       'scenario_issued_at':'2005-01-08T12:00:00+00:00',
                       'shapes5':{m:dict(shape_record) for m in ALL_METHODS}})
    check=validate_asof_frames(frames,panel)
    assert check['reason']=='RISK_SOURCE_ID_VINTAGE_PAYLOAD_CONFLICT',check
    frames[1]['scenario_source_id']='fixture-new-vintage'
    check=validate_asof_frames(frames,panel)
    assert check['status']=='PASS' and check['scenario_weight_modes']==[
        'WEIGHTED_ISSUED','WEIGHTED_ISSUED'],check
    frames[1]['scenario_mass']=[1.2,-.2]
    check=validate_asof_frames(frames,panel)
    assert check['reason']=='INVALID_ASOF_COV_SCENARIO_OR_SHAPE',check
    # ERC cannot reach equal five-way risk contribution if only E1 may be
    # purchased. A permission-projected threshold must not manufacture churn.
    squeeze=outcome('PASS',asset_caps={a:(1. if a=='E1' else 0.)
                                          for a in ASSETS[:5]},
                    new_buy={a:a=='E1' for a in ASSETS[:5]})
    pre=np.array([.1,0.,0.,0.,0.,.9])
    th=owner('erc-th',model='ERC',geometry=B.GEOMETRY['ERC'],value=.01,
             capital_tolerance=.01,threshold_reference='PERMISSION_PROJECTED_TARGET',
             reference_geometry_contract='PROJECTED_TARGET_NOT_CONSTRAINED_OPTIMUM')
    decision=benchmark_restructure('ERC',pre,np.full(5,.2),np.eye(5)*.0004,
          regime='BOTH',zone='BOTTOM_REENTRY_ZONE',capital_share=.5,
          squeeze=squeeze,threshold_owner=th,utility_owner=u,
          scenarios6=np.zeros((200,6)),cost_rates6=np.zeros(6),nav=1.,
          risk_limits={str(a):{'VaR':1.,'ES':1.} for a in ALPHAS})
    assert decision['status']=='PASS',decision
    assert decision['original_benchmark_geometry_deviation']>.7,decision
    assert abs(decision['projected_reference_geometry_deviation'])<1e-12,decision
    assert decision['action']=='HOLD',decision
    th2=dict(th);th2.pop('reference_geometry_contract')
    denied=benchmark_restructure('ERC',pre,np.full(5,.2),np.eye(5)*.0004,
          regime='BOTH',zone='BOTTOM_REENTRY_ZONE',capital_share=.5,
          squeeze=squeeze,threshold_owner=th2,utility_owner=u,
          scenarios6=np.zeros((200,6)),cost_rates6=np.zeros(6),nav=1.)
    assert denied['reason']=='PROJECTED_GEOMETRY_OWNER_INTERPRETATION_REQUIRED',denied
    return {'status':'PASS','cases':18,
            'tested':['PROBABILITY_UTILITY_MATCH','WEIGHTED_ES_ATOM_SPLIT',
                      'ZERO_MASS_OR_NONNORMALIZED_REJECTED',
                      'FROZEN_ASSET_PROJECTION','CAP_INFEASIBLE',
                      'NET_GRID_STABILITY','ACTION_LEVEL_TAIL_SUPPORT']}


def _rev6_guard_self_test() -> dict:
    """Audit-registered adversarial tests: immutable Short/EXIT lineage and censorship.

    H0: an old EXIT can close a later Short and unfinished Short events vanish.
    H1: wrong Short binding is FAIL; censored events never become zero avoided loss.
    Falsifier: make ID/issue clock stale, reuse ended ID, or omit Bottom.
    Comparison: same four native weekly calendar keys and issued clocks.
    Stopping: all seven adversarial cases plus six-strategy baseline must pass.
    Fixtures are synthetic and must NEVER be treated as DEV economic outcomes.
    """
    from copy import deepcopy
    starts=[date(2005,1,10)+timedelta(weeks=i) for i in range(5)]
    panel={'status':'PASS','n':5,'starts':starts,
           'ends':[s+timedelta(days=4) for s in starts]}
    def sample(regimes=('N','BOTH','BOTH','N','BOTH')):
        rows=[]
        for i,begin in enumerate(starts):
            issued=(begin-timedelta(days=1)).isoformat()+'T12:00:00+00:00'
            rows.append({'week_start':begin.isoformat(),
                         'week_end':(begin+timedelta(days=4)).isoformat(),
                         'issue_at':issued,'regime':regimes[i],
                         'regime_source_id':f'RQ01-{i}','regime_issued_at':issued,
                         'short_confirmed':False,'first_n_confirmed':False,
                         'bottom_permissions':{}})
        return rows
    base=sample()
    base[1].update(short_confirmed=True,short_id='S1',short_issued_at=base[1]['issue_at'])
    base[2].update(short_confirmed=True,short_id='S2',short_issued_at=base[2]['issue_at'])
    base[3].update(first_n_confirmed=True,exit_id='X2',
                   exit_issued_at=base[3]['issue_at'],exit_for_short_id='S2')
    base[4].update(short_confirmed=True,short_id='S3',short_issued_at=base[4]['issue_at'])
    res=parse_issued_path(base,panel)
    assert res['status']=='PASS',res
    cs=res['censored_episodes']
    assert [e['reason'] for e in cs]==[
        'SUPERSEDED_BY_NEWER_SHORT','EXIT_BEFORE_VALID_BOTTOM',
        'RIGHT_CENSORED_AT_DEV_END'],cs
    assert [e['short_id'] for e in cs]==['S1','S2','S3']
    assert all(e['status']=='NOT_EVALUATED' and e['avoided_loss_fraction'] is None for e in cs)
    assert res['event_support']['paired_short_bottom']==0
    assert res['event_support']['censored_without_bottom']==3
    wrong=deepcopy(base)
    wrong[3]['exit_for_short_id']='S1'
    q=parse_issued_path(wrong,panel)
    assert q['reason']=='EXIT_DOES_NOT_BIND_ACTIVE_SHORT',q
    wrong=deepcopy(base)
    wrong[3].pop('exit_for_short_id')
    assert parse_issued_path(wrong,panel)['reason']=='EXIT_DOES_NOT_BIND_ACTIVE_SHORT'
    wrong=deepcopy(base)
    wrong[3]['exit_issued_at']=base[0]['issue_at']
    assert parse_issued_path(wrong,panel)['reason']=='EXIT_ISSUED_BEFORE_REFERENCED_SHORT'
    wrong=deepcopy(base)
    wrong[4]['short_id']='S1'
    assert parse_issued_path(wrong,panel)['reason']=='OLD_SHORT_ID_REUSED_AS_NEW_SHORT'
    wrong=deepcopy(base)
    wrong[2]['short_id']='S1'
    wrong[2]['short_issued_at']=base[1]['short_issued_at']
    wrong[3]['exit_for_short_id']='S1'
    q=parse_issued_path(wrong,panel)
    assert q['status']=='PASS' and q['event_support']['superseded_without_bottom']==0,q
    wrong=deepcopy(base)
    wrong[2]['short_id']='S1'
    wrong[2]['short_issued_at']=base[2]['issue_at']
    assert parse_issued_path(wrong,panel)['reason']=='SAME_SHORT_ID_CHANGED_ISSUE_VINTAGE'
    no_events=sample(('N','N','N','N','N'))
    q=parse_issued_path(no_events,panel)
    assert q['status']=='PASS' and q['event_support']['censored_without_bottom']==0
    bad=deepcopy(base)
    bad[1]['short_id']=['UNHASHABLE_SHORT_ID']
    assert parse_issued_path(bad,panel)['reason']=='SHORT_ISSUANCE_INVALID'
    bad=deepcopy(base)
    bad[0]['bottom_permissions']=['NOT_A_MAP']
    assert parse_issued_path(bad,panel)['reason']=='BOTTOM_PERMISSION_MAP_REQUIRED'
    # Prove that missing capital-objective ownership cannot default to
    # MAX_FEASIBLE_RISKY_SHARE, even if all other synthetic mechanics exist.
    rec={**base[0], 'zone':'NORMAL'}
    cfg={'issued_records':[rec],
         'ew_definition_owner':{'approved':True,'owner_hash':'ew',
                                 'definition':'EW5_RISKY_SLEEVE'},
         'asset_cap_owner':{'approved':True,'owner_hash':'caps',
                            'unit':'FRACTION_OF_NAV',
                            'caps':{a:1. for a in ASSETS}},
         'models':{'EW':{'asset_caps':{a:1. for a in ASSETS[:5]},
                          'base_caps':{a:1. for a in ASSETS[:5]},
                          'defense':{a:0. for a in ASSETS[:5]},
                          'allowance':{a:0. for a in ASSETS[:5]}}}}
    single_panel={'X':np.zeros((1,6))}
    q=_continuation_weights('EW',0,np.r_[np.zeros(5),1.],1.,
                             single_panel,{'EW':[np.full(5,.2)]},
                             np.asarray([np.eye(5)*.0004]),cfg)
    assert q['reason']=='EXPLICIT_OWNER_CAPITAL_OBJECTIVE_REQUIRED',q
    cfg['capital_objective_owner']={'approved':False,'owner_hash':'fake',
                                    'objective':'MAX_FEASIBLE_RISKY_SHARE'}
    q=_continuation_weights('EW',0,np.r_[np.zeros(5),1.],1.,
                             single_panel,{'EW':[np.full(5,.2)]},
                             np.asarray([np.eye(5)*.0004]),cfg)
    assert q['reason']=='EXPLICIT_OWNER_CAPITAL_OBJECTIVE_REQUIRED',q
    return {'status':'PASS','cases':12,
            'fixture_only':True,'events_censored':len(cs),
            'tested_failure_types':['MISBOUND_EXIT','PRE_SHORT_EXIT',
                                    'REUSED_SHORT_ID','CHANGED_SHORT_VINTAGE',
                                    'SUPERSESSION','RIGHT_CENSOR','EXIT_BEFORE_BOTTOM']}

def self_test():
    """Inline executable falsification examples; no invented real research output."""
    assert np.allclose(capital_domain('N'),[.5,.55,.6,.65,.7,.75])
    assert capital_domain('D')[-1]==.5 and capital_domain('G')[0]==0
    r=np.array([[.04,0,0,0,0,.001],[-.08,0,0,0,0,.001],[-.02,0,0,0,0,.001],[.01,0,0,0,0,.001]])
    fr=risk_frontier(r,{'EW':outcome('PASS',shape=np.array([1,0,0,0,0]))},'N')
    rev5=_rev5_guard_self_test()
    assert fr['models']['EW']['status']=='PASS' and fr['models']['POLICY']['status']=='BLOCKED'
    assert fr['models']['EW']['frontier'][-1]['a']==.75
    assert choose_capital_by_risk(fr,'EW','N',None)['status']=='BLOCKED'
    caps={a:1. for a in ASSETS}
    limits={str(a):{'VaR':1.,'ES':1.} for a in ALPHAS}
    sel=choose_capital_by_risk(fr,'EW','N',{'approved':True,'owner_hash':'r',
          'risk_unit':'DECIMAL_LOSS_FRACTION_OF_NAV',
          'caps':{'EW':{'N':limits}}},asset_cap_owner={'approved':True,'owner_hash':'c',
                                                     'unit':'FRACTION_OF_NAV','caps':caps})
    assert sel['status']=='PASS' and sel['risky_share']==.75
    s5={a:.5 for a in ASSETS[:5]}; zeros={a:0. for a in s5}
    perm={a:None for a in s5}
    sq=bottom_risk_squeeze('BOTH','BOTTOM_REENTRY_ZONE',base_caps=s5,defense=zeros,
         allowance=s5,permission=perm,owner_caps=s5)
    assert sq['status']=='PASS' and not any(sq['new_buy'].values())
    decision='2009-01-02T00:00:00+00:00'
    perm['E1']={'authorized':True,'asset':'E1','source_id':'fixture',
                'issued_at':'2009-01-01T00:00:00+00:00',
                'committee_id':'fixture-committee','risk_caps_owner_hash':'h',
                'zone_start':'2009-01-02','zone_end':'2009-01-10',
                'funding_owner':{'approved':True,'owner_hash':'fund'},
                'eligibility_owner':{'approved':True,'owner_hash':'elig','assets':['E1']},
                'reentry_policy_owner':{'approved':True,'owner_hash':'policy'},
                'risk_check':{'status':'PASS','within_caps':True,'risk_caps_owner_hash':'h',
                              'source_id':'rcheck','issued_at':'2009-01-01T00:00:00+00:00'},
                'committee_rule_owner':{'approved':True,'owner_hash':'rule','k_independent':2},
                'committee_rule_owner_hash':'rule',
                'factor_evidence':{f:{'vote':True,'input_ids':[f],
                                      'issued_at':'2009-01-01T00:00:00+00:00','source_id':'fixture-'+f}
                    for f in ('EWI','COUNTRY','RISK_COV')}}
    sq=bottom_risk_squeeze('BOTH','BOTTOM_REENTRY_ZONE',base_caps=s5,defense=zeros,
         allowance=s5,permission=perm,owner_caps=s5,
         decision_at=decision,active_risk_cap_hash='h')
    assert sq['new_buy']['E1'] and not sq['new_buy']['E2']
    one_asset = benchmark_restructure('EW', np.r_[np.zeros(5), 1.],
        np.full(5,.2), np.eye(5)*.001, regime='BOTH',
        zone='BOTTOM_REENTRY_ZONE', capital_share=.5, squeeze=sq,
        threshold_owner={'approved':True,'owner_hash':'test-thr',
                         'model':'EW','geometry':B.GEOMETRY['EW'],
                         'value':0.,'capital_tolerance':0.},
        utility_owner={'approved':True,'owner_hash':'test-util',
                       'mode':'LOG','hurdle':0.},
        scenarios6=np.tile([.05,.01,.01,.01,.01,0.],(100,1)),
        cost_rates6=np.zeros(6),nav=1.)
    assert one_asset['status']=='PASS',one_asset
    assert one_asset['post_weights'][0]>0 and np.allclose(one_asset['post_weights'][1:5],0)
    # Identical holdings -> exact zero charge and equal wealth accounting.
    p=np.array([.1,.1,.1,.1,.1,.5]);c=TR.self_financing_cost(p,p,np.ones(6)*.001)
    assert c['cost_fraction'] < 1e-12
    # A future-issued regime cannot pass a historic week decision.
    rows={'issue_at':'2005-01-18T10:00:00+00:00'}
    assert not _issue_before_window(rows,date(2005,1,17),date(2005,1,14))
    integration = _synthetic_six_strategy_lifecycle_test()
    assert integration['status']=='PASS'
    guards=_rev2_guard_self_test()
    assert guards['status']=='PASS'
    rev3=_rev3_guard_self_test()
    assert rev3['status']=='PASS'
    rev4=_rev4_guard_self_test()
    assert rev4['status']=='PASS'
    rev6=_rev6_guard_self_test()
    assert rev6['status']=='PASS'
    rev7=_rev7_guard_self_test()
    assert rev7['status']=='PASS'
    return {'status':'PASS','baseline_cases':10,
            'rev2_new_falsification_cases':guards['new_falsification_cases'],
            'rev3_new_falsification_cases':rev3['falsification_cases'],
            'rev4_new_falsification_cases':rev4['new_falsification_cases'],
            'rev5_new_falsification_cases':rev5['cases'],
            'rev6_new_falsification_cases':rev6['cases'],
            'rev7_new_falsification_cases':rev7['cases'],
            'rev6_censored_event_fixture':rev6['events_censored'],
            'six_strategy_lifecycle':integration,
            'scope':'SYNTHETIC_LOGIC_ONLY'}



# REV8: PRE-REGISTERED FINITE RESEARCH-ONLY REAL-DATA TESTS.
# This is not the issuer-based historical six-policy replay.  Full-DEV
# covariance, and any benchmark weights computed from it, are known in
# hindsight.  Treat estimates as arithmetic/function-validation only.
REV8_RESEARCH_THRESHOLDS = {
    'EW': (0., .02, .05, .10),
    'IVOL': (0., .02, .05, .10),
    'MVP': (0., 1.e-7, 1.e-6, 1.e-5),  # WEEKLY VARIANCE, not percent
    'ERC': (0., .02, .05, .10),
    'MaxDiv': (0., .025, .05, .10),
}
REV8_RESEARCH_COSTS = (0., .0005, .001, .0025)
REV8_RESEARCH_ALLOCATIONS = (('NORMAL_HYPOTHETICAL',.50),
                             ('NORMAL_HYPOTHETICAL',.625),
                             ('NORMAL_HYPOTHETICAL',.75),
                             ('STRESS_HYPOTHETICAL',.375),
                             ('STRESS_HYPOTHETICAL',.50))


def validate_research_threshold_grid_config(thresholds: dict, costs, allocations) -> dict:
    """Pre-flight domains: never compare model geometries on a common unit."""
    if set(thresholds)!=set(METHODS):
        return outcome('BLOCKED','FIVE_BENCHMARK_GEOMETRIES_REQUIRED')
    for model,vec in thresholds.items():
        a=np.asarray(vec,float)
        if (a.ndim!=1 or len(a)<2 or not np.isfinite(a).all() or
                (a<0).any() or np.any(np.diff(a)<=0)):
            return outcome('BLOCKED','FINITE_ASCENDING_OWN_GEOMETRY_THRESHOLDS_REQUIRED',model=model)
    c=np.asarray(costs,float)
    if c.ndim!=1 or len(c)<1 or not np.isfinite(c).all() or (c<0).any() or (c>=1).any():
        return outcome('BLOCKED','FINITE_SCALAR_HYPOTHETICAL_COST_GRID_REQUIRED')
    if not isinstance(allocations,(tuple,list)) or not allocations:
        return outcome('BLOCKED','NONEMPTY_REGISTERED_ALLOCATIONS_REQUIRED')
    # Validate each allocation before hashing/comparison: malformed user inputs
    # must return a declared BLOCKED result instead of escaping ValueError.
    try:
        if any(not isinstance(z,(tuple,list)) or len(z)!=2 or
               not isinstance(z[1],(float,int)) or isinstance(z[1],bool) or
               not np.isfinite(z[1]) for z in allocations):
            return outcome('BLOCKED','INVALID_HYPOTHETICAL_CAPITAL_DOMAIN')
        if len({(str(label),float(a)) for label,a in allocations})!=len(allocations):
            return outcome('BLOCKED','DUPLICATE_ALLOCATION_TRIALS_DISALLOWED')
    except (TypeError,ValueError):
        return outcome('BLOCKED','INVALID_HYPOTHETICAL_CAPITAL_DOMAIN')
    for label,a in allocations:
        if (isinstance(a,(int,float)) and not isinstance(a,bool) and np.isfinite(a) and
            ((label=='NORMAL_HYPOTHETICAL' and 0.5<=a<=0.75) or
             (label=='STRESS_HYPOTHETICAL' and 0<=a<=0.5))):
            continue
        return outcome('BLOCKED','INVALID_HYPOTHETICAL_CAPITAL_DOMAIN',case=(label,a))
    return outcome('PASS',claim=RESEARCH)


def replay_hindsight_benchmark_threshold(panel: dict, method: str,
                                         frozen_shape5, covariance5,
                                         *, risky_fraction: float,
                                         allocation_case: str,
                                         geometry_threshold: float,
                                         capital_threshold: float,
                                         cost_bps: float) -> dict:
    """DEV research oracle: weekly execute own-geometry trigger, one cost per trade.

    Frozen full-DEV shape is deliberately NOT an as-issued forecast.  This
    validates six-asset ledger identities and the trade/threshold mechanics.
    Only five FROZEN benchmark shapes, never synthetic POLICY or regime labels.
    Cash owner, bond investability and official execution rights are NOT implied.
    """
    if method not in METHODS or panel.get('status')!='PASS':
        return outcome('BLOCKED','CANONICAL_DEV_AND_REAL_BENCHMARK_REQUIRED')
    legal_case=(allocation_case=='NORMAL_HYPOTHETICAL' and 0.5<=risky_fraction<=0.75
                or allocation_case=='STRESS_HYPOTHETICAL' and 0<=risky_fraction<=0.5)
    if (not legal_case or not np.isfinite(risky_fraction) or
            not np.isfinite(geometry_threshold) or geometry_threshold<0 or
            not np.isfinite(capital_threshold) or capital_threshold<0 or
            not np.isfinite(cost_bps) or not 0<=cost_bps<10000):
        return outcome('BLOCKED','RESEARCH_PARAMETER_DOMAIN_INVALID')
    shape=_simplex(frozen_shape5,5)
    cov=_covariance(np.asarray(covariance5,float),5,positive=True)
    tar=_portfolio(shape,float(risky_fraction))
    X=np.asarray(panel['X'],float)
    if X.ndim!=2 or X.shape[1]!=6 or len(X)<2 or (X<=-1).any():
        return outcome('BLOCKED','VALID_DEV_RETURN_WINDOWS_REQUIRED')
    rate=np.full(6,cost_bps/10000.)
    nav=1.
    pre=np.array([0.,0.,0.,0.,0.,1.])
    realised=[]
    navpath=[1.]
    trigger_count=0
    action_mask=bytearray()
    cumulative_cost=0.
    total_turnover=0.
    max_identity_residual=0.
    maximum_risky_fraction=0.
    # Covariance is full-DEV frozen: cache the exact deterministic geometry
    # algebra.  B.structural_deviation is checked independently against this
    # routine at multiple legal portfolio vectors in _rev8_geometry_oracle_test.
    target_var=float(shape@cov@shape)
    sig=np.sqrt(np.diag(cov))
    target_div=float(sig@shape/np.sqrt(target_var))
    regime_floor=0.5 if allocation_case=='NORMAL_HYPOTHETICAL' else 0.
    regime_cap=0.75 if allocation_case=='NORMAL_HYPOTHETICAL' else 0.5
    for i,r in enumerate(X):
        # With zero risky, structure is mathematically undefined, so only
        # capital drift is used.  Initial allocation is an explicit trade.
        a=float(pre[:5].sum())
        if a>1e-12:
            v=pre[:5]/a
            if method in ('EW','IVOL'):
                d=float(np.sum(np.abs(v-shape)))
            else:
                var=float(v@cov@v)
                if var<=0:
                    return outcome('NOT_EVALUATED','ZERO_VARIANCE_OWN_GEOMETRY',week=i)
                if method=='MVP':
                    d=float(max(0.,var-target_var))
                elif method=='ERC':
                    rc=v*(cov@v)/var
                    if np.min(rc)<-1e-12:
                        return outcome('NOT_EVALUATED','NEGATIVE_RC_GEOMETRY',week=i)
                    d=float(np.max(np.abs(rc-0.2)))
                else:
                    d=float(max(0.,target_div-(sig@v)/np.sqrt(var)))
        else:
            d=0.
        due=(d>geometry_threshold+1e-14 or
             abs(a-risky_fraction)>capital_threshold+1e-14 or
             a>regime_cap+1e-10 or a<regime_floor-1e-10)
        post=tar if due else pre
        if due:
            charge=TR.self_financing_cost(pre,post,rate)
            c=float(charge['cost_fraction'])
        else:
            c=0.
            charge={'turnover':0.}
        if due: trigger_count+=1
        action_mask.append(int(due))
        gross=float(post@r)
        gross_wealth=nav*(1.+gross)
        net_wealth=nav*(1.-c)*(1.+gross)
        residual=abs(net_wealth-(gross_wealth-nav*c*(1.+gross)))
        max_identity_residual=max(max_identity_residual,residual)
        cumulative_cost+=nav*c
        total_turnover+=float(charge['turnover'])
        if net_wealth<=0 or not np.isfinite(net_wealth):
            return outcome('FAIL','NONPOSITIVE_OR_INVALID_RESEARCH_NAV',week=i)
        realised.append(float(net_wealth/nav-1.))
        nav=float(net_wealth)
        navpath.append(nav)
        maximum_risky_fraction=max(maximum_risky_fraction,float(post[:5].sum()))
        pre=TR.drift(post,r)
    path=np.array(navpath)
    loss=-np.array(realised)
    tail={str(alpha):{k:float(v[k]) for k in ('VaR','ES','effective_tail_support')}
          for alpha in ALPHAS for v in [T.var_es(loss,float(alpha))]}
    if max_identity_residual>1.e-10:
        return outcome('FAIL','SELF_FINANCING_RECONCILIATION_FAILED',residual=max_identity_residual)
    return outcome('PASS',model=method,geometry=B.GEOMETRY[method],
                   risky_fraction=float(risky_fraction),
                   allocation_case=allocation_case,
                   geometry_threshold=float(geometry_threshold),
                   capital_threshold=float(capital_threshold),
                   cost_bps=float(cost_bps),n_native_weeks=int(len(X)),
                   n_rebalance=int(trigger_count),
                   action_mask_sha256=hashlib.sha256(action_mask).hexdigest(),
                   total_turnover=float(total_turnover),
                   terminal_wealth_net_hypothetical=float(nav),
                   max_drawdown=float(np.max(1.-path/np.maximum.accumulate(path))),
                   cumulative_cost_money=float(cumulative_cost),
                   max_accounting_residual=float(max_identity_residual),
                   maximum_executed_risky_fraction=float(maximum_risky_fraction),
                   realised_tail_net_weekly=tail,claim=RESEARCH,
                   model_fit='FULL_DEV_COVARIANCE_LOOKAHEAD_DIAGNOSTIC_NOT_ISSUED',
                   disclaimer='COST_BPS_AND_CAPITAL_CASES_HYPOTHETICAL_NOT_APPROVED')


def validate_real_dev_main_functions(dataset_root: str | Path,
                                     *, capital_threshold: float=.025,
                                     thresholds=None, costs=REV8_RESEARCH_COSTS,
                                     allocations=REV8_RESEARCH_ALLOCATIONS) -> dict:
    """Run finite same-DEV falsifiable test. No DEV folds/OOS/issuer invention.

    Comparison: own-geometry thresholds vs zero tolerance, same five benchmark
    shapes and same locked return tape; cost sensitivity controlled by research
    bps levels.  Output explicitly does NOT select a production threshold.
    """
    th=REV8_RESEARCH_THRESHOLDS if thresholds is None else thresholds
    chk=validate_research_threshold_grid_config(th,costs,allocations)
    if chk['status']!='PASS':return chk
    p=load_locked_dev(dataset_root)
    if p['status']!='PASS':return p
    cov=np.cov(p['X'][:,:5],rowvar=False)
    shapes=benchmark_vectors(cov)
    rows=[]
    for method in METHODS:
        item=shapes[method]
        if item['status']!='PASS':
            return outcome('NOT_EVALUATED','BENCHMARK_SHAPE_NOT_SOLVED',model=method)
        for label,a in allocations:
            for tolerance in th[method]:
                for rate in costs:
                    x=replay_hindsight_benchmark_threshold(p,method,item['shape'],cov,
                                    risky_fraction=float(a),allocation_case=label,geometry_threshold=float(tolerance),
                                    capital_threshold=float(capital_threshold),cost_bps=float(rate*10000))
                    if x['status']!='PASS':
                        return outcome('NOT_EVALUATED','REAL_DEV_FUNCTION_TRIAL_FAILED',
                                       failed_model=method,case=(label,a,tolerance,rate),detail=x)
                    rows.append({'allocation_case':label,**x})
    # The first run must validate full matrix; there may be unequal threshold
    # grid lengths, so derive the count from the actual preregistration.
    actual_expected=sum(len(th[m])*len(costs)*len(allocations) for m in METHODS)
    if len(rows)!=actual_expected:
        return outcome('FAIL','RESEARCH_TRIAL_COUNT_IDENTITY_FAILED')
    threshold_masks={}
    for m in METHODS:
        threshold_masks[m]={}
        for label,a in allocations:
            key=f'{label}:{a:.6f}'
            threshold_masks[m][key]={}
            for rate in costs:
                cc=[row for row in rows if row['model']==m and
                    row['allocation_case']==label and row['risky_fraction']==a and
                    abs(row['cost_bps']-rate*10000)<1.e-9]
                threshold_masks[m][key][str(float(rate*10000))]={
                    'distinct_executed_masks':len({row['action_mask_sha256'] for row in cc}),
                    'registered_threshold_count':len(th[m]),
                    'identifiable_by_action_mask':len({row['action_mask_sha256'] for row in cc})>1,
                }
    return outcome('PASS',claim=RESEARCH,n_dev_weeks=p['n'],
                   tape_sha=p['tape_sha'],n_configurations=len(rows),
                   configuration_rows=rows,
                   threshold_action_identifiability=threshold_masks,
                   policy_status='BLOCKED_ISSUED_POLICY_SHAPE_AND_OWNER_EVIDENCE_REQUIRED',
                   threshold_selection='NOT_SELECTED_FULL_DEV_IN_SAMPLE',
                   diagnosis='BOUNDARY_CAPITAL_POINTS_MAY_FORCE_REBALANCE_AND_MASK_GEOMETRY_THRESHOLDS',
                   event_short_bottom='NOT_EVALUATED_NO_AUTHENTIC_ISSUED_EPISODES',
                   capital_cases='HYPOTHETICAL_NOT_ES_BUDGET_OPTIMISED',
                   model_fit='ONE_FULL_DEV_ESTIMATE_NO_TIME_FOLDS',
                   acceptance='ALL_NET_WEALTH_IDENTITIES_AND_GEOMETRY_SOLVERS_VALID',
                   formal_oos='SEALED_NOT_ACCESSED')




def research_threshold_pareto_diagnostics(research_result: dict) -> dict:
    """A same-DEV *non-selection* check of 5 geometry-specific threshold grids.

    Dominance within identical model, requested allocation and cost scenario:
      terminal wealth NET higher or equal, tail ES99 lower or equal,
      max drawdown lower or equal, turnover lower or equal, at least one strict.
    This is NOT official model selection, nor an uncertainty interval; linked
    observations and adaptive testing prevent a selection claim.
    """
    if research_result.get('status')!='PASS' or research_result.get('claim')!=RESEARCH:
        return outcome('BLOCKED','REAL_DEV_TAPE_TRIAL_REQUIRED')
    rows=research_result.get('configuration_rows',[])
    frontiers={}
    for m in METHODS:
        for label,a in REV8_RESEARCH_ALLOCATIONS:
            for bps in (0.,5.,10.,25.):
                key=f'{m}:{label}:{a:.6f}:{bps:.1f}bps'
                cells=[q for q in rows if q['model']==m and q['allocation_case']==label
                       and abs(q['risky_fraction']-a)<1e-12 and abs(q['cost_bps']-bps)<1e-9]
                if not cells:
                    return outcome('BLOCKED','MISSING_MATCHED_THRESHOLD_COMPARISONS',key=key)
                nondominated=[]
                for i,q in enumerate(cells):
                    dominated=False
                    for j,r in enumerate(cells):
                        if i==j:continue
                        weak=(r['terminal_wealth_net_hypothetical']>=q['terminal_wealth_net_hypothetical']-1e-12
                              and r['realised_tail_net_weekly']['0.99']['ES']<=q['realised_tail_net_weekly']['0.99']['ES']+1e-12
                              and r['max_drawdown']<=q['max_drawdown']+1e-12
                              and r['total_turnover']<=q['total_turnover']+1e-12)
                        strict=(r['terminal_wealth_net_hypothetical']>q['terminal_wealth_net_hypothetical']+1e-12
                                or r['realised_tail_net_weekly']['0.99']['ES']<q['realised_tail_net_weekly']['0.99']['ES']-1e-12
                                or r['max_drawdown']<q['max_drawdown']-1e-12
                                or r['total_turnover']<q['total_turnover']-1e-12)
                        if weak and strict:
                            dominated=True;break
                    if not dominated:nondominated.append(q['geometry_threshold'])
                frontiers[key]={'thresholds_not_dominated_on_same_DEV':nondominated,
                               'distinct_action_masks':len({r['action_mask_sha256'] for r in cells}),
                               'threshold_count':len(cells)}
    return outcome('PASS',pareto=frontiers,
                   selection='NOT_SELECTED_SAME_DEV_IS_NOT_INDEPENDENT_VALIDATION',
                   uncertainty='NOT_IDENTIFIABLE_NO_INDEPENDENT_EPISODES',
                   claim=RESEARCH)


def _rev8_geometry_oracle_test():
    """Native formulas vs existing library at multiple admissible vectors."""
    s=np.array([[.002,.0001,0,0,0],[.0001,.003,0,0,0],
                [0,0,.005,0,0],[0,0,0,.00003,0],[0,0,0,0,.0002]])
    shapes=benchmark_vectors(s)
    for method in METHODS:
        tar=np.asarray(shapes[method]['shape'],float)
        for v in (np.ones(5)/5, np.array([.10,.15,.15,.4,.2]),tar):
            ref=B.structural_deviation(method,v,s,target=tar)
            assert ref['status']=='PASS'
            var=float(v@s@v)
            targetvar=float(tar@s@tar)
            if method in ('EW','IVOL'): actual=float(np.abs(v-tar).sum())
            elif method=='MVP':actual=float(max(0.,var-targetvar))
            elif method=='ERC': actual=float(np.abs(v*(s@v)/var-.2).max())
            else:
                sig=np.sqrt(np.diag(s))
                actual=float(max(0.,sig@tar/np.sqrt(targetvar)-sig@v/np.sqrt(var)))
            assert abs(actual-ref['deviation'])<1.e-12,(method,actual,ref['deviation'])
    return outcome('PASS',native_benchmark_geometries=len(METHODS),oracle_vectors=3)


def _rev8_falsification_test():
    oracle=_rev8_geometry_oracle_test()
    assert oracle['status']=='PASS'
    sc=np.array([[.03,0,0,0,0,.001],[-.04,0,0,0,0,.001],[.01,0,0,0,0,.001],
                 [.00,0,0,0,0,.001]])
    sh={'EW':{'status':'PASS','shape':[.2]*5}}
    block=risk_frontier(sc,sh,'N',projected_caps5={a:0.05 for a in ASSETS[:5]})
    assert block['models']['EW']['status']=='BLOCKED'
    free=risk_frontier(sc,sh,'N',projected_caps5={a:.2 for a in ASSETS[:5]})
    assert free['models']['EW']['status']=='PASS'
    assert all(q['executed_risky_fraction']>=.5-1e-10 for q in free['models']['EW']['frontier'])
    assert validate_research_threshold_grid_config(REV8_RESEARCH_THRESHOLDS,
                                                   REV8_RESEARCH_COSTS,
                                                   REV8_RESEARCH_ALLOCATIONS)['status']=='PASS'
    assert validate_research_threshold_grid_config({**REV8_RESEARCH_THRESHOLDS,'ERC':(0,float('nan'))},
                                                   REV8_RESEARCH_COSTS,
                                                   REV8_RESEARCH_ALLOCATIONS)['status']=='BLOCKED'
    own={'approved':True,'owner_hash':'synthetic-cost','unit':'PROPORTIONAL_TRADE_VALUE',
         'rates':{a:0.001 for a in ASSETS}}
    good={'cost_owner':own,'cost_rates6':[.001]*6}
    assert authorized_execution_cost_rates(good)['status']=='PASS'
    bad={**good,'cost_rates6':[.001]*5+[0.]}
    assert authorized_execution_cost_rates(bad)['reason']=='OWNER_OPTIMIZER_EXECUTOR_COST_SCHEDULE_MISMATCH'
    sq={'status':'PASS','asset_caps':{a:1. for a in ASSETS[:5]},
        'new_buy':{a:True for a in ASSETS[:5]}}
    tol={'approved':True,'owner_hash':'synthetic-th','model':'EW',
         'geometry':B.GEOMETRY['EW'],'value':0.,'capital_tolerance':0.}
    ownu={'approved':True,'owner_hash':'synthetic-util','mode':'LOG','hurdle':0.,
          'emergency_repair_approved':True}
    args=dict(model='EW',pre=[.1]*5+[.5],shape5=[.2]*5,covariance5=np.eye(5)*.002,
              regime='BOTH',zone='STRESS_PRE_CONFIRMATION',capital_share=.5,
              squeeze=sq,utility_owner=ownu,scenarios6=np.tile([.001]*6,(200,1)),
              cost_rates6=np.zeros(6),nav=1.)
    malformed=benchmark_restructure(**args,threshold_owner={**tol,'value':float('nan')})
    assert malformed['status']=='BLOCKED'
    malformed2=benchmark_restructure(**args,threshold_owner={**tol,'capital_tolerance':float('nan')})
    assert malformed2['status']=='BLOCKED'
    micro={'status':'PASS','X':np.array([[-.02]*5+[0.],[-.02]*5+[0.]])}
    mini_cov=np.eye(5)*.002
    opts=dict(risky_fraction=.5,geometry_threshold=10.,capital_threshold=.025,cost_bps=0.)
    normal=replay_hindsight_benchmark_threshold(micro,'EW',[.2]*5,mini_cov,
                        allocation_case='NORMAL_HYPOTHETICAL',**opts)
    stress=replay_hindsight_benchmark_threshold(micro,'EW',[.2]*5,mini_cov,
                        allocation_case='STRESS_HYPOTHETICAL',**opts)
    assert normal['n_rebalance']==2 and stress['n_rebalance']==1,(normal,stress)
    return outcome('PASS',cases=9,geometry_oracle=oracle,scope='SYNTHETIC_FALSIFICATION_NOT_DATA_VALIDATION')



# REV9 research-only risk-budget frontier on the source-locked real DEV tape.
# Goal: show the exact relationship among finite capital thresholds, VaR/ES,
# independently provided hypothetical loss budgets and self-financing costs.
# These are retrospective 781-week scenarios, not six as-issued NET policies.
REV9_HYPOTHETICAL_NET_ES_BUDGETS = (0.005, 0.01, 0.02, 0.04)
REV9_CAPITAL_STEP = 0.025


def _contiguous_feasible_segments(candidates: list[dict]) -> list[dict]:
    """Return every feasible grid component; never presume ES monotonicity in a."""
    segments=[]
    start=None
    prev=None
    for x in candidates:
        if x['feasible']:
            if start is None:start=float(x['a'])
            prev=float(x['a'])
        elif start is not None:
            segments.append({'min_a':start,'max_a':prev})
            start=None
    if start is not None:segments.append({'min_a':start,'max_a':prev})
    return segments


def validate_real_dev_capital_risk_thresholds(dataset_root: str | Path,
                                               *, budgets=REV9_HYPOTHETICAL_NET_ES_BUDGETS,
                                               step: float=REV9_CAPITAL_STEP,
                                               hypothetical_cost_bps: float=10.0) -> dict:
    """Direct DEV validation: 5 risky shapes x 4 state DOMAINS x 4 budgets.

    H0: some finite capital grid points satisfy a stated VaR/ES budget.
    H1: at least one stated budget/state/model combination is infeasible.
    Falsifier: enumerate all points rather than assuming tail monotonicity;
    output every feasible segment and a precise not-feasible reason. All results
    are retrospective with full-DEV benchmark estimates and hypothetical costs.

    A budget here constrains BOTH one-week NET VaR and NET ES at all three alphas
    in decimal fraction of NAV. The initial trade is from fully funded C1. The
    scenario tape itself has no verified issuer clock: *never* promote this.
    """
    pan=load_locked_dev(dataset_root)
    if pan['status']!='PASS':return pan
    if (not isinstance(step,(int,float)) or isinstance(step,bool) or
        not np.isfinite(step) or not 0<step<=0.5 or not np.isfinite(hypothetical_cost_bps)
        or not 0<=hypothetical_cost_bps<10000):
        return outcome('BLOCKED','RESEARCH_STEP_OR_COST_DOMAIN_INVALID')
    try:
        bs=np.asarray(budgets,dtype=float)
    except (ValueError,TypeError):
        return outcome('BLOCKED','INVALID_HYPOTHETICAL_LOSS_BUDGETS')
    if (bs.ndim!=1 or len(bs)<2 or not np.isfinite(bs).all() or
        (bs<0).any() or np.any(np.diff(bs)<=0)):
        return outcome('BLOCKED','ASCENDING_NONNEGATIVE_RESEARCH_BUDGETS_REQUIRED')
    X=np.asarray(pan['X'],float)
    cov=np.cov(X[:,:5],rowvar=False)
    shapes=benchmark_vectors(cov)
    rate=np.full(6,float(hypothetical_cost_bps)/10000.)
    pre=np.array([0.,0.,0.,0.,0.,1.])
    all_rows=[]
    status=[]
    for model in METHODS:
        obj=shapes[model]
        if obj['status']!='PASS':return outcome('NOT_EVALUATED','FIVE_SHAPES_REQUIRED',model=model)
        for regime in REGIMES:
            candidates=[]
            for a in capital_domain(regime,step):
                w=_portfolio(obj['shape'],float(a))
                cost_info=TR.self_financing_cost(pre,w,rate)
                cost=float(cost_info['cost_fraction'])
                realised_net=(1.-cost)*(1.+X@w)-1.
                risk={str(alpha):T.var_es(-realised_net,float(alpha)) for alpha in ALPHAS}
                candidates.append({'a':float(a),'post_weights':w.tolist(),
                                   'cost_fraction':cost,
                                   'turnover':float(cost_info['turnover']),
                                   'risk':{str(alpha):{'VaR':float(risk[str(alpha)]['VaR']),
                                                       'ES':float(risk[str(alpha)]['ES']),
                                                       'effective_tail_support':float(risk[str(alpha)]['effective_tail_support'])}
                                           for alpha in ALPHAS}})
            for budget in bs:
                tested=[{'a':c['a'],
                         'feasible':bool(all(c['risk'][str(q)][metric]<=budget+1e-12
                                             for q in ALPHAS for metric in ('VaR','ES')))}
                        for c in candidates]
                segments=_contiguous_feasible_segments(tested)
                count=sum(int(c['feasible']) for c in tested)
                all_rows.append({'model':model,'regime':regime,
                                 'budget_decimal_nav_loss':float(budget),
                                 'feasible_grid_points':int(count),
                                 'total_grid_points':len(candidates),
                                 'feasible_segments':segments,
                                 'maximum_feasible_allocation':max((c['a'] for c in tested if c['feasible']),default=None),
                                 'claim':'RESEARCH_FEASIBILITY_NOT_POLICY_RECOMMENDATION'})
            status.append({'model':model,'regime':regime,
                           'capital_frontier':candidates,
                           'origin':'FULL_DEV_RETROSPECTIVE_NOT_ASOF'})
    expected=len(METHODS)*len(REGIMES)*len(bs)
    if len(all_rows)!=expected:return outcome('FAIL','CAPITAL_RISK_TRIAL_COUNT_IDENTITY_BROKEN')
    return outcome('PASS',run_id=canonical_sha256({'kind':'REV9_REAL_DEV_CAPITAL_VAREs',
                         'sha':pan['tape_sha'],'budgets':bs.tolist(),'step':float(step),
                         'cost_bps':hypothetical_cost_bps}),
                   n_dev_weeks=pan['n'],source_tape_sha256=pan['tape_sha'],
                   n_model_regime=len(status),n_budget_tests=expected,
                   n_total_budget_grid_comparisons=sum(q['total_grid_points'] for q in all_rows),
                   budgets=bs.tolist(),capital_step=float(step),cost_bps=hypothetical_cost_bps,
                   budget_feasibility=all_rows,model_regime_risk=status,
                   normal_bounds=[0.5,0.75],stress_bounds=[0.,0.5],
                   risk_unit='NET_SINGLE_WEEK_LOSS_FRACTION_OF_NAV',
                   cost_convention='ONE_INITIAL_SELF_FINANCING_TRADE_FROM_CASH_PER_SCENARIO',
                   policy_status='BLOCKED_NO_ISSUED_POLICY_SHAPE_OR_COST_RISK_OWNER',
                   threshold_selection='NOT_SELECTED_FULL_DEV_IN_SAMPLE',
                   certified_net_wealth='NOT_EVALUATED',
                   claim=RESEARCH)


def _rev9_main_function_falsification_test() -> dict:
    """Adversarial NORMAL-floor / non-monotonic segments / empty-grid tests."""
    assert _contiguous_feasible_segments([{'a':0.,'feasible':True},
                 {'a':.1,'feasible':False},{'a':.2,'feasible':True}])==[
                 {'min_a':0.,'max_a':0.},{'min_a':.2,'max_a':.2}]
    for bad in (float('nan'),float('inf'),False,0.,-.1):
        try:capital_domain('N',bad)
        except ValueError:pass
        else:raise AssertionError('INVALID_CAPITAL_STEP_ACCEPTED')
    assert validate_research_threshold_grid_config(REV8_RESEARCH_THRESHOLDS,
                       REV8_RESEARCH_COSTS,[])['status']=='BLOCKED'
    assert research_matched_joint_threshold_gate({'status':'PASS','claim':RESEARCH})['status']=='BLOCKED'
    cov=np.eye(5)*.0004
    sq={'status':'PASS','asset_caps':{a:1. for a in ASSETS[:5]},
        'new_buy':{a:True for a in ASSETS[:5]}}
    th={'approved':True,'owner_hash':'TH-FIXTURE','model':'EW',
        'geometry':B.GEOMETRY['EW'],'value':.5,'capital_tolerance':.5}
    utl={'approved':True,'owner_hash':'UTIL-FIXTURE','mode':'LOG',
         'hurdle':0.,'emergency_repair_approved':True,
         'normal_floor_activation_approved':True}
    args=dict(model='EW',pre=[0.,0.,0.,0.,0.,1.],
              shape5=[.2]*5,covariance5=cov,regime='N',zone='NORMAL',
              capital_share=.5,squeeze=sq,threshold_owner=th,utility_owner=utl,
              scenarios6=np.zeros((200,6)),cost_rates6=np.zeros(6),nav=1.)
    q=benchmark_restructure(**args)
    assert q['status']=='PASS' and q['executed_risky_share']>=.5-1e-10,q
    frozen={**sq,'new_buy':{a:False for a in ASSETS[:5]}}
    no_buy=benchmark_restructure(**{**args,'squeeze':frozen})
    assert no_buy['status']=='BLOCKED',no_buy
    # NAV-aware shape movement may leave Normal upper/lower satisfied after costs.
    st=benchmark_restructure(**{**args,'regime':'BOTH',
                                'zone':'STRESS_PRE_CONFIRMATION','capital_share':.5,
                                'squeeze':frozen})
    assert st['status']=='PASS' and st['executed_risky_share']==0.,st
    return outcome('PASS',cases=9,
                   validated=['NORMAL_HARD_FLOOR','NO_BUY_INFEASIBILITY',
                              'STRESS_NO_FORCED_RISK_PURCHASE',
                              'NAN_GRID_INPUT_REJECTED','DISJOINT_FEASIBLE_SEGMENTS'])




def research_matched_joint_threshold_gate(threshold_result: dict,
                                         *, budgets=REV9_HYPOTHETICAL_NET_ES_BUDGETS,
                                         hypothetical_max_drawdown: float | None = None,
                                         hypothetical_max_turnover: float | None = None) -> dict:
    """Joint finite-grid acceptance across own-geometry, capital, and loss caps.

    For each executed five-benchmark threshold configuration, test all alpha
    single-week realized NET VaR/ES against each scalar hypothetical budget.
    Optional drawdown/turnover maximum constraints use declared units. No
    continuous L/U interval is inferred from sparse grid successes; L/U is an
    envelope of tested points only. Identical trade masks are equivalence
    classes, NOT repeated independent evidence. All scores stay in-sample.
    """
    if (threshold_result.get('status')!='PASS' or
        threshold_result.get('claim')!=RESEARCH or
        not threshold_result.get('tape_sha')):
        return outcome('BLOCKED','MATCHED_CANONICAL_DEV_THRESHOLD_RUN_REQUIRED')
    try:
        arr=np.asarray(budgets,dtype=float)
    except (ValueError,TypeError):
        return outcome('BLOCKED','INVALID_RESEARCH_BUDGET_GRID')
    if (arr.ndim!=1 or len(arr)<2 or not np.isfinite(arr).all() or
            (arr<0).any() or np.any(np.diff(arr)<=0)):
        return outcome('BLOCKED','ORDERED_HYPOTHETICAL_BUDGETS_REQUIRED')
    for v in (hypothetical_max_drawdown,hypothetical_max_turnover):
        if v is not None and (not isinstance(v,(int,float)) or isinstance(v,bool)
                              or not np.isfinite(v) or v<0):
            return outcome('BLOCKED','FINITE_NONNEGATIVE_OPTIONAL_CONSTRAINT_REQUIRED')
    rows=threshold_result['configuration_rows']
    if not rows or len(rows)!=threshold_result.get('n_configurations'):
        return outcome('BLOCKED','COMPLETE_MATCHED_THRESHOLD_ROWS_REQUIRED')
    out=[]
    equivalence={}
    for m in METHODS:
        own=[r for r in rows if r['model']==m]
        for r in own:
            for alpha in ALPHAS:
                v=r['realised_tail_net_weekly'][str(alpha)]
                if v['effective_tail_support']<=0 or not all(np.isfinite(v[k]) for k in ('VaR','ES')):
                    return outcome('FAIL','INVALID_EXECUTED_TAIL_OBSERVATION',model=m)
            if r['max_accounting_residual']>1e-10:
                return outcome('FAIL','SELF_FINANCING_EXECUTION_DISCREPANCY',model=m)
            if r['maximum_executed_risky_fraction']>(.75 if r['allocation_case']=='NORMAL_HYPOTHETICAL' else .5)+1e-10:
                return outcome('FAIL','RESEARCH_REGIME_CAP_BREACH',model=m)
            k=(m,r['allocation_case'],r['risky_fraction'],r['cost_bps'])
            equivalence.setdefault(k,{}).setdefault(r['action_mask_sha256'],[]).append(r['geometry_threshold'])
            for cap in arr:
                tail_ok=all(r['realised_tail_net_weekly'][str(q)][metric] <= cap+1e-12
                            for q in ALPHAS for metric in ('VaR','ES'))
                draw_ok=hypothetical_max_drawdown is None or r['max_drawdown']<=hypothetical_max_drawdown+1e-12
                turnover_ok=hypothetical_max_turnover is None or r['total_turnover']<=hypothetical_max_turnover+1e-12
                out.append({'model':m,'allocation_case':r['allocation_case'],
                            'risky_fraction':r['risky_fraction'],'cost_bps':r['cost_bps'],
                            'geometry':r['geometry'],'geometry_threshold':r['geometry_threshold'],
                            'VaR_ES_budget_decimal_nav':float(cap),
                            'all_alpha_tail_pass':bool(tail_ok),'max_dd_pass':bool(draw_ok),
                            'turnover_pass':bool(turnover_ok),
                            'feasible':bool(tail_ok and draw_ok and turnover_ok),
                            'action_mask_sha256':r['action_mask_sha256']})
    equivalence_rows=[]
    for (m,case,alloc,cost), groups in equivalence.items():
        equivalence_rows.append({'model':m,'case':case,'risky_share':alloc,'cost_bps':cost,
                                 'n_registered_thresholds':sum(len(q) for q in groups.values()),
                                 'n_identifiable_action_masks':len(groups),
                                 'indistinguishable_threshold_sets':[
                                     sorted([float(x) for x in vals]) for vals in groups.values()],
                                 'interpretation':'SAME_ACTIONS_WITHIN_DEV_NOT_INDEPENDENT_TESTS'})
    bounds=[]
    for m in METHODS:
        for cap in arr:
            sample=[r['geometry_threshold'] for r in out if r['model']==m and
                    abs(r['VaR_ES_budget_decimal_nav']-cap)<1e-12 and r['feasible']]
            bounds.append({'model':m,'risk_cap':float(cap),
                           'n_feasible_trials':len(sample),
                           'tested_threshold_L':min(sample) if sample else None,
                           'tested_threshold_U':max(sample) if sample else None,
                           'interpretable_only_at_registered_grid_points':True,
                           'threshold_set_NOT_owner_approved':True})
    expected=len(rows)*len(arr)
    if len(out)!=expected:
        return outcome('FAIL','JOINT_GRID_CARDINALITY_MISMATCH')
    return outcome('PASS',run_id=canonical_sha256({'kind':'REV9_JOINT_RISK_GEOMETRY',
                       'sha':threshold_result['tape_sha'],'budgets':arr.tolist(),
                       'n_configurations':len(rows),
                       'dd_cap':hypothetical_max_drawdown,'turnover_cap':hypothetical_max_turnover}),
                   n_dev_weeks=threshold_result['n_dev_weeks'],
                   source_tape_sha256=threshold_result['tape_sha'],
                   n_configurations=len(rows),n_joint_veto_checks=len(out),
                   n_feasible_joint_checks=sum(int(r['feasible']) for r in out),
                   budget_grid=arr.tolist(),hypothetical_max_drawdown=hypothetical_max_drawdown,
                   hypothetical_max_turnover=hypothetical_max_turnover,
                   geometry_grid_native_units=True,threshold_bounds=bounds,
                   action_equivalence_classes=equivalence_rows,
                   selection='NOT_SELECTED_NO_AUTHORISED_MULTI_OBJECTIVE_LOSS_OR_INDEPENDENT_VALIDATION',
                   owner_gates='NOT_PASSED',
                   evidence='FULL_DEV_SAME_SAMPLE_ONLY_NO_ISSUED_STATE',claim=RESEARCH)


def validate_joint_dev_main_thresholds(dataset_root: str | Path) -> dict:
    """Run all 400 actual matched research executions and joint risk vetoes."""
    base=validate_real_dev_main_functions(dataset_root)
    if base['status']!='PASS':return base
    gate=research_matched_joint_threshold_gate(base)
    if gate['status']!='PASS':return gate
    gate['all_executed_paths_single_cost_once']=True
    gate['risk_cap_source']='EXPLICIT_HYPOTHETICAL_GRID_NOT_AUTHORISED_OWNER_LIMITS'
    return gate



# REV10: all interval starts are candidate counterfactual STARTS, not inferred
# issuer SHORT confirmations. Each interval length is pre-registered in NATIVE
# weeks. This exercise evaluates economic algebra and threshold sensitivity
# on the source-locked DEV tape; it never constructs fictitious RQ02 labels.
REV10_RESEARCH_OWN_GEOMETRY_THRESHOLDS = {
    'EW': (0.0, 0.05),
    'IVOL': (0.0, 0.05),
    'MVP': (0.0, 1e-7),
    'ERC': (0.0, 0.05),
    'MaxDiv': (0.0, 0.05),
}
REV10_HORIZON_NATIVE_WEEKS = (4, 12)
REV10_RESEARCH_COST_BPS = (0.0, 10.0)


def _rev10_research_path(panel: dict, model: str, shape5, cov5,
                         *, risky_fraction: float, geometry_threshold: float,
                         capital_threshold: float, cost_bps: float) -> dict:
    """Complete NAV/pre/post path from 2005, matched to REV8 geometry oracle.

    This is a full-DEV fitted RESEARCH benchmark; no regime is inferred from
    outcomes, no issuer or costs owner is asserted. Investment arm trades using
    its own geometric rule throughout the DEV history, including the interval.
    """
    if model not in METHODS or panel.get('status') != 'PASS':
        return outcome('BLOCKED', 'RESEARCH_SOURCE_AND_OWN_GEOMETRY_REQUIRED')
    if (not np.isfinite(risky_fraction) or not 0.5 <= risky_fraction <= 0.75 or
        not np.isfinite(geometry_threshold) or geometry_threshold < 0 or
        not np.isfinite(capital_threshold) or capital_threshold < 0 or
        not np.isfinite(cost_bps) or not 0 <= cost_bps < 10000):
        return outcome('BLOCKED', 'INVALID_PRE_REGISTERED_RESEARCH_DOMAIN')
    X=np.asarray(panel['X'],dtype=float)
    cov=_covariance(np.asarray(cov5,dtype=float),5,positive=True)
    shape=_simplex(shape5,5)
    target=_portfolio(shape,float(risky_fraction))
    rates=np.full(6,float(cost_bps)/10000.)
    navpath=np.ones(len(X)+1,dtype=float)
    prepath=np.empty((len(X),6),dtype=float)
    postpath=np.empty((len(X),6),dtype=float)
    costpath=np.zeros(len(X),dtype=float)
    actionmask=bytearray()
    pre=np.r_[np.zeros(5),1.]
    for t,ret in enumerate(X):
        prepath[t]=pre
        a=float(pre[:5].sum())
        if a < 1e-12:
            deviation=0.
        else:
            dev=B.structural_deviation(model,pre[:5]/a,cov,target=shape)
            if dev['status']!='PASS':
                return outcome('NOT_EVALUATED','RESEARCH_GEOMETRY_NOT_DEFINED',model=model,week=t,
                               detail=dev.get('reason'))
            deviation=float(dev['deviation'])
        due=(deviation>geometry_threshold+1e-14 or
             abs(a-risky_fraction)>capital_threshold+1e-14 or
             a<0.5-1e-10 or a>0.75+1e-10)
        post=target if due else pre
        charge=TR.self_financing_cost(pre,post,rates) if due else {'cost_fraction':0.}
        cost=float(charge['cost_fraction'])
        wealth=navpath[t]*(1-cost)*(1+float(post@ret))
        if not np.isfinite(wealth) or wealth<=0:
            return outcome('FAIL','INVALID_DEV_WEALTH',week=t)
        navpath[t+1]=wealth
        postpath[t]=post
        costpath[t]=cost
        actionmask.append(int(due))
        pre=TR.drift(post,ret)
    return outcome('PASS',navpath=navpath,prepath=prepath,postpath=postpath,
                   costpath=costpath,mask_sha256=hashlib.sha256(actionmask).hexdigest(),
                   n_trade=sum(actionmask),model=model,terminal_wealth=float(navpath[-1]),
                   scope=RESEARCH)


def _rev10_freeze_interval_oracle(panel: dict, path: dict, start: int, end: int,
                                  cost_bps: float) -> dict:
    """Independent direct recursion for ONE possible cash-freeze window.

    Tests the vectorised NAV-ratio identity against direct weekly cost-once
    recursion. The invested branch follows model-specific 2005-started actions;
    the other branch pays the liquidation fee once, then earns C1 cash carry.
    No reinvestment at end is counted before the hypothetical Bottom starts.
    """
    if not 0 <= start < end <= panel['n']:
        return outcome('BLOCKED','INVALID_NATIVE_WEEK_WINDOW')
    rates=np.full(6,float(cost_bps)/10000.)
    cash=np.r_[np.zeros(5),1.]
    cash_cost=float(TR.self_financing_cost(path['prepath'][start],cash,rates)['cost_fraction'])
    freeze=1.
    invested=1.
    for t in range(start,end):
        cf=cash_cost if t==start else 0.
        freeze*=((1-cf)*(1+float(panel['X'][t,5])))
        invested*=((1-float(path['costpath'][t]))*
                   (1+float(path['postpath'][t]@panel['X'][t])))
    return outcome('PASS',freeze_growth=float(freeze),invested_growth=float(invested),
                   avoided_fraction=float(freeze-invested),
                   cash_liquidation_cost_fraction=cash_cost,
                   scope='INDEPENDENT_CANDIDATE_INTERVAL_ALGEBRA_ONLY')


def research_real_dev_short_bottom_interval_frontier(dataset_root: str | Path,
        *, horizons=REV10_HORIZON_NATIVE_WEEKS,
        own_thresholds=REV10_RESEARCH_OWN_GEOMETRY_THRESHOLDS,
        cost_bps=REV10_RESEARCH_COST_BPS, risky_fraction: float=.625,
        capital_threshold: float=.025) -> dict:
    """Real DEV, event-AGNOSTIC candidate interval freeze-vs-invest sensitivity.

    H0: holding C1 rather than investing cannot underperform for ANY candidate
    interval. H1: there are negative avoided-loss intervals (missed upside).
    Falsifier: every eligible native interval start is tested for fixed 4/12
    week horizons; no date threshold or start-date selection. All overlapping
    windows are dependent; results are *not* independent RQ02 Short episodes.

    Cost bps, all windows, capital and thresholds are hypothetical registered
    sensitivities. No claim that C1 is fundable, B1/B2 investable, or that an
    issuer issued an actual Short or Bottom at ANY of these dates.
    """
    panel=load_locked_dev(dataset_root)
    if panel.get('status')!='PASS':return panel
    if (set(own_thresholds)!=set(METHODS) or
        not isinstance(horizons,(tuple,list)) or not isinstance(cost_bps,(tuple,list)) or
        not 0.5<=risky_fraction<=0.75 or not np.isfinite(capital_threshold) or
        capital_threshold<0):
        return outcome('BLOCKED','REV10_PREREGISTRATION_DOMAIN_INVALID')
    horizon_vals=[]
    for h in horizons:
        if isinstance(h,bool) or not isinstance(h,(int,np.integer)) or not 1<=h<panel['n']:
            return outcome('BLOCKED','NATIVE_INTERVAL_HORIZON_INVALID')
        horizon_vals.append(int(h))
    if not horizon_vals or len(set(horizon_vals))!=len(horizon_vals):
        return outcome('BLOCKED','UNIQUE_HORIZON_GRID_REQUIRED')
    if any(not isinstance(c,(int,float)) or isinstance(c,bool) or not np.isfinite(c)
           or not 0<=c<10000 for c in cost_bps):
        return outcome('BLOCKED','HYPOTHETICAL_COST_GRID_INVALID')
    if len(set(cost_bps))!=len(cost_bps):
        return outcome('BLOCKED','UNIQUE_COST_GRID_REQUIRED')
    th={k:tuple(v) for k,v in own_thresholds.items()}
    if any(not v or any(not np.isfinite(z) or z<0 for z in v)
           or len(set(v))!=len(v) for v in th.values()):
        return outcome('BLOCKED','OWN_GEOMETRY_THRESHOLD_GRID_INVALID')
    cov=np.cov(panel['X'][:,:5],rowvar=False)
    shapes=benchmark_vectors(cov)
    if any(shapes[m]['status']!='PASS' for m in METHODS):
        return outcome('NOT_EVALUATED','FIVE_NATIVE_BENCHMARK_SHAPES_REQUIRED')
    x=panel['X']; n=panel['n']
    cash_factor=np.r_[1.,np.cumprod(1+x[:,5])]
    if not np.isfinite(cash_factor).all() or np.any(cash_factor<=0):
        return outcome('FAIL','CASH_PROXY_COMPOUNDING_INVALID')
    records=[]
    comparisons=0
    max_oracle_residual=0.
    paths_validated=0
    for model in METHODS:
        for tol in th[model]:
            for fee in cost_bps:
                p=_rev10_research_path(panel,model,shapes[model]['shape'],cov,
                      risky_fraction=risky_fraction,geometry_threshold=float(tol),
                      capital_threshold=capital_threshold,cost_bps=float(fee))
                if p['status']!='PASS':return p
                original=replay_hindsight_benchmark_threshold(panel,model,
                     shapes[model]['shape'],cov,risky_fraction=risky_fraction,
                     allocation_case='NORMAL_HYPOTHETICAL',
                     geometry_threshold=float(tol),capital_threshold=capital_threshold,
                     cost_bps=float(fee))
                if (original['status']!='PASS' or
                    not np.isclose(p['terminal_wealth'],original['terminal_wealth_net_hypothetical'],
                                   atol=1e-9,rtol=1e-9) or
                    p['n_trade']!=original['n_rebalance'] or
                    p['mask_sha256']!=original['action_mask_sha256']):
                    return outcome('FAIL','INDEPENDENT_2005_REPLAY_MISMATCH',
                                   model=model,threshold=tol,cost_bps=fee)
                paths_validated+=1
                nav=p['navpath']; cash_pre=p['prepath']
                k=np.full(6,float(fee)/10000.)
                cash=np.r_[np.zeros(5),1.]
                cash_cost=np.array([TR.self_financing_cost(q,cash,k)['cost_fraction']
                                    for q in cash_pre],float)
                for h in horizon_vals:
                    starts=np.arange(n-h+1,dtype=int)
                    ends=starts+h
                    freeze=(1-cash_cost[starts])*(cash_factor[ends]/cash_factor[starts])
                    invested=nav[ends]/nav[starts]
                    gap=freeze-invested
                    if not np.isfinite(gap).all():
                        return outcome('FAIL','NONFINITE_CANDIDATE_INTERVAL_PNL')
                    probes=sorted(set([0,len(starts)//2,len(starts)-1]))
                    for idx in probes:
                        direct=_rev10_freeze_interval_oracle(panel,p,int(starts[idx]),int(ends[idx]),float(fee))
                        if direct['status']!='PASS':return direct
                        residual=float(abs(direct['avoided_fraction']-gap[idx]))
                        max_oracle_residual=max(max_oracle_residual,residual)
                        if residual>1e-10:
                            return outcome('FAIL','CASH_FREEZE_INDEPENDENT_ORACLE_MISMATCH',
                                           model=model,horizon=h,index=int(idx),residual=residual)
                    comparisons+=len(starts)
                    records.append({'model':model,'own_geometry':B.GEOMETRY[model],
                        'geometry_threshold':float(tol),'capital_threshold':float(capital_threshold),
                        'risky_fraction':float(risky_fraction),'cost_bps':float(fee),
                        'horizon_native_weeks':h,'candidate_intervals':len(starts),
                        'positive_avoided_count':int(np.sum(gap>1e-12)),
                        'negative_missed_gain_count':int(np.sum(gap< -1e-12)),
                        'near_zero_count':int(np.sum(abs(gap)<=1e-12)),
                        'mean_avoided_fraction':float(np.mean(gap)),
                        'median_avoided_fraction':float(np.median(gap)),
                        'p05_avoided_fraction':float(np.quantile(gap,.05)),
                        'p95_avoided_fraction':float(np.quantile(gap,.95)),
                        'worst_candidate_freeze_underperformance':float(np.min(gap)),
                        'best_candidate_freeze_outperformance':float(np.max(gap)),
                        'trade_count_full_history':int(p['n_trade']),
                        'action_mask_sha256':p['mask_sha256'],
                        'terminal_benchmark_net_wealth_research':float(nav[-1]),
                        'scope':'DEPENDENT_OVERLAPPING_CANDIDATE_WINDOWS_NOT_ISSUED_EVENTS'})
    if any(z['candidate_intervals']!=z['positive_avoided_count']+
           z['negative_missed_gain_count']+z['near_zero_count'] for z in records):
        return outcome('FAIL','CANDIDATE_WINDOW_DENOMINATOR_IDENTITY_FAILED')
    return outcome('PASS',n_dev_weeks=n,source_tape_sha256=panel['tape_sha'],
         n_native_benchmark_paths=paths_validated,n_configurations=len(records),
         total_dependent_candidate_intervals=comparisons,
         independent_cash_freeze_oracle_max_residual=max_oracle_residual,
         model_threshold_cost_horizon_rows=records,
         null_hypothesis='CASH_FREEZE_NEVER_UNDERPERFORMS_IN_PRE_REGISTERED_CANDIDATE_WINDOWS',
         falsifier='NEGATIVE_AVOIDED_LOSS_FRACTION',
         short_bottom_issued_events='NOT_EVALUATED_UNAUTHENTICATED_RQ02_ISSUED_EVENTS',
         policy='BLOCKED_ISSUED_POLICY_WEIGHT_HISTORY',
         official_net='BLOCKED_OWNER_COST_CASH_FUNDING_BOND_INVESTABILITY',
         disclaimer='FULL_DEV_RETROSPECTIVE_FIT_COSTS_CAPITAL_AND_C1_ARE_HYPOTHETICAL',
         threshold_promotion='NONE',claim=RESEARCH,
         run_id=canonical_sha256({'sha':panel['tape_sha'],'thresholds':th,
                           'horizons':horizon_vals,'costs':cost_bps,'risky_fraction':risky_fraction,
                           'capital_threshold':capital_threshold}))


def _rev10_interval_falsification_test() -> dict:
    """Synthetic proof: freeze can *lose*; liquidation fee charged once."""
    tiny=outcome('PASS', n=3, X=np.array([[.1,0,0,0,0,0],
                                          [-.2,0,0,0,0,0],
                                          [.1,0,0,0,0,0]]))
    path={'prepath':np.array([[.5,0,0,0,0,.5],
                             [.5,0,0,0,0,.5],
                             [.5,0,0,0,0,.5]]),
          'postpath':np.array([[.5,0,0,0,0,.5]]*3),
          'costpath':np.zeros(3)}
    first=_rev10_freeze_interval_oracle(tiny,path,0,1,0.)
    second=_rev10_freeze_interval_oracle(tiny,path,1,2,0.)
    charged=_rev10_freeze_interval_oracle(tiny,path,0,1,10.)
    expected_fee=TR.self_financing_cost(path['prepath'][0],np.r_[np.zeros(5),1.],
                                       np.full(6,0.001))['cost_fraction']
    if (first['status']!='PASS' or second['status']!='PASS' or charged['status']!='PASS'
        or not np.isclose(first['avoided_fraction'],-.05)
        or not np.isclose(second['avoided_fraction'],.1)
        or not np.isclose(charged['cash_liquidation_cost_fraction'],expected_fee)
        or not np.isclose(charged['avoided_fraction'],first['avoided_fraction']-expected_fee)):
        return outcome('FAIL','REV10_CASH_FREEZE_ORACLE_COUNTEREXAMPLE_FAILED')
    return outcome('PASS',negative_avoided_fraction=first['avoided_fraction'],
                   positive_avoided_fraction=second['avoided_fraction'],
                   cash_liquidation_cost_fraction=expected_fee,
                   claim='SYNTHETIC_ACCOUNTING_ORACLE_NOT_EMPIRICAL_SHORT_EVENT')


def main(argv=None) -> int:
    p=argparse.ArgumentParser(description='W05 additive user-priority extension REV10 (code only)')
    p.add_argument('--dataset-root')
    p.add_argument('--mode',choices=('risk','horizon','replay','validate-functions','validate-capital-frontier','validate-joint-thresholds','validate-interval-frontier'),default='risk')
    p.add_argument('--issued-json')
    p.add_argument('--owner-json')
    p.add_argument('--asof-json')
    p.add_argument('--self-test',action='store_true')
    args=p.parse_args(argv)
    if args.self_test:
        out=self_test()
        rev8=_rev8_falsification_test()
        rev9=_rev9_main_function_falsification_test()
        out['rev8_falsification']=rev8
        out['rev9_falsification']=rev9
        rev10=_rev10_interval_falsification_test()
        out['rev10_falsification']=rev10
        if rev10['status']!='PASS':out['status']='FAIL'
        if rev8['status']!='PASS' or rev9['status']!='PASS':out['status']='FAIL'
    elif not args.dataset_root:
        out=outcome('BLOCKED','DATASET_ROOT_REQUIRED')
    elif args.mode=='risk':
        out=risk_only_dev(args.dataset_root)
    elif args.mode=='horizon':
        pan=load_locked_dev(args.dataset_root)
        out=research_multiweek_risk(pan) if pan['status']=='PASS' else pan
    elif args.mode=='validate-capital-frontier':
        out=validate_real_dev_capital_risk_thresholds(args.dataset_root)
    elif args.mode=='validate-joint-thresholds':
        out=validate_joint_dev_main_thresholds(args.dataset_root)
    elif args.mode=='validate-interval-frontier':
        out=research_real_dev_short_bottom_interval_frontier(args.dataset_root)
    elif args.mode=='validate-functions':
        out=validate_real_dev_main_functions(args.dataset_root)
        if out.get('status')=='PASS':
            summary=research_threshold_pareto_diagnostics(out)
            if summary['status']!='PASS':
                out=summary
            else:
                out['threshold_pareto_diagnostic']=summary['pareto']
    else:
        pan=load_locked_dev(args.dataset_root)
        if pan['status']!='PASS':
            out=pan
        elif not args.issued_json or not args.owner_json or not args.asof_json:
            out=outcome('BLOCKED','EXTERNAL_WEEKLY_ISSUED_OWNER_AND_ASOF_SOURCE_FILES_REQUIRED',
                        n_dev_weeks=pan['n'],tape_sha=pan['tape_sha'])
        else:
            issued=json.loads(Path(args.issued_json).read_text(encoding='utf-8'))
            owners=json.loads(Path(args.owner_json).read_text(encoding='utf-8'))
            asof=json.loads(Path(args.asof_json).read_text(encoding='utf-8'))
            parsed=parse_issued_path(issued.get('records',[]),pan,
                                     active_cap_owner_hash=owners.get('asset_cap_owner',{}).get('owner_hash'))
            if parsed['status']!='PASS':
                out=parsed
            else:
                checked=validate_asof_frames(asof.get('frames',[]),pan,
                                             issued_records=parsed['records'])
                if checked['status']!='PASS':
                    out=checked
                else:
                    config={**owners,'covariance_mode':'ISSUED_ASOF_VERIFIED',
                            'scenario_by_week':checked['scenarios'],
                            'scenario_mass_by_week':checked['scenario_masses'],
                            'scenario_weight_mode_by_week':checked['scenario_weight_modes']}
                    executed=short_to_bottom_counterfactual(pan,parsed,checked['shapes'],
                                                             checked['covariances'],config)
                    # Display compact execution summary; full event/history records
                    # available through the Python function (no extra output files).
                    if 'models' in executed:
                        out={**{k:v for k,v in executed.items() if k!='models'},
                             'models':{m:{'status':v['status'],'reason':v.get('reason'),
                                          'n_weeks':v.get('n_weeks'),'terminal_nav':v.get('terminal_nav'),
                                          'n_events':v.get('n_events'),
                                          'event_ledger':v.get('event_ledger'),
                                          'first_blocker':v.get('first_blocker')}
                                       for m,v in executed['models'].items()}}
                    else:
                        out=executed
    print(json.dumps(out,ensure_ascii=False,default=lambda o:o.tolist() if isinstance(o,np.ndarray) else str(o),
                     indent=2))
    return 0 if out['status']=='PASS' else 2


if __name__=='__main__':
    raise SystemExit(main())
