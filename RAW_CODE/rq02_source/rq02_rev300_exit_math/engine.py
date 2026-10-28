"""REV300 RQ02 Exit A/B: independently computed research rules, no learned claim.

A uses fitted owner-issued IL, IS, purified JL, JS packets; supports K-of-4.
B derives signed RAW, distinct-source, progression and timing evidence from the
seven EWI state tape. A and B are independently scoreable at H18/H12/H6/H3.
A/B future first-N decisions do not overwrite current RQ01 regime.
This module contains RESEARCH-only rules; no labels, coefficients or thresholds
are learned in this code. Thresholds must be explicitly supplied with owners.

GLOBAL NO-U contract: A and B votes are binary PASS/FAIL; a decision that cannot be proven from the
evaluable components is a non-vote record (NOT_EVALUATED / BLOCKED + reason) without any vote.  The EXIT-B
native-source gate is FAIL when all source statuses are computable and n_native < K (REV373 returned U).
A censored first-N label is LABEL_CENSORED (never 0, never a third prediction class).
"""
from __future__ import annotations

from calendar import monthrange
from collections import defaultdict
from datetime import date
from math import isfinite
from numbers import Real
from typing import Any

from research_adapters.vote_contract import (LABEL_CENSORED, NonVote, ThirdVoteError, decide_and,
                                             decide_at_least, decide_or, is_nonvote, not_evaluated)
from rq02_source.rq02_rev280_dual_exit.core import (
    EWI, FAMILIES, EXIT_HORIZONS, STATE_RANK, STRESS, _dev,
    _family_proof, add_months, day, parallel_exit_controller, _verified_events,
)
from rq02_source.rq02_rev279_onset_exit_country.core import (
    _check_predicted_rq01_view, _onset_confirmed_asof, regime_fields,
)

VOTES = frozenset(('PASS', 'FAIL'))      # binary vote domain; non-evaluable components are NonVote records
SOURCES = frozenset(EWI)


def _month_end(d):
    d = day(d)
    return date(d.year, d.month, monthrange(d.year, d.month)[1])


def decision_interval(votes: dict, k: int) -> dict:
    """Exact K-of-N bound rule: PASS iff L>=K; FAIL iff L+m<K (m = non-evaluable); otherwise no vote."""
    if isinstance(k,bool) or not isinstance(k,int) or not 1 <= k <= len(votes):
        raise ValueError('K_OUTSIDE_1_TO_N')
    if not votes:
        raise ValueError('VOTE_DOMAIN')
    for v in votes.values():
        if v == 'U':
            raise ThirdVoteError('LEGACY_U_VOTE_REJECTED')
        if not (is_nonvote(v) or v in VOTES):
            raise ValueError('VOTE_DOMAIN')
    lower=sum(v=='PASS' for v in votes.values() if not is_nonvote(v))
    m=sum(is_nonvote(v) for v in votes.values())
    out={'K':k,'confirmed_min':lower,'possible_max':lower+m,'unresolved':m,'n':len(votes)}
    d=decide_at_least(list(votes.values()),k)
    if is_nonvote(d):
        return {'execution_status':d.status,'reason':d.reason,**out}
    return {'vote':d,'execution_status':'EXECUTED',**out}


def _vote_map(votes: dict) -> tuple[dict, dict]:
    """{name: PASS/FAIL | NonVote} -> (binary votes, {name: non-vote reason})."""
    return ({k:v for k,v in votes.items() if not is_nonvote(v)},
            {k:v.reason for k,v in votes.items() if is_nonvote(v)})


def branch_A(*, issue, horizon, components=None, k=1, k_owner_id=None):
    """Actual 4-family K aggregation from each existing signed worker family.

    k=1 is ANY and k=4 is ALL. Intermediate K requires distinct threshold
    owner identity; otherwise the decision is BLOCKED (not a free post-hoc tuning knob, no vote).
    """
    u=_dev(issue)
    if horizon not in EXIT_HORIZONS: raise ValueError('INVALID_EXIT_HORIZON')
    if not isinstance(components or {},dict) or set(components or {})-set(FAMILIES):
        raise ValueError('FAMILY_COMPONENT_REGISTRY_INVALID')
    votes={}; missing={}
    for family in FAMILIES:
        votes[family],reason=_family_proof(component=(components or {}).get(family),
                                             issue=u,horizon=horizon,family=family)
        if reason: missing[family]=reason
    interval=decision_interval(votes,k)
    if k not in (1,4) and not k_owner_id:
        interval={**{q:v for q,v in interval.items() if q not in ('vote','execution_status','reason')},
                  'execution_status':'BLOCKED','reason':'K2_K3_THRESHOLD_OWNER_REQUIRED'}
    fam,fam_ne=_vote_map(votes)
    base={'branch':'HORIZON_A','issue':u.isoformat(),
          'target':add_months(u,horizon).isoformat(),'horizon_months':horizon,
          'families':fam,'families_not_evaluated':fam_ne,'missing_reasons':missing,
          'aggregation':interval,'threshold_owner':k_owner_id if k in (2,3) else 'EXISTING_FAMILY_POLICY'}
    if interval['execution_status']=='EXECUTED':
        return {**base,'vote':interval['vote'],'execution_status':'EXECUTED','status':'RESEARCH_VOTE_ONLY'}
    return {**base,'execution_status':interval['execution_status'],'status':interval['execution_status'],
            'reason':interval['reason']}


def _num(v, name, *, nonnegative=True):
    if isinstance(v,bool) or not isinstance(v,Real) or not isfinite(float(v)):
        raise ValueError('INVALID_NUMERIC_'+name)
    x=float(v)
    if nonnegative and x<0:raise ValueError('NEGATIVE_'+name)
    return x


def validate_B_threshold(*, issue, horizon, config):
    """Full specified threshold schema; config is not itself a fitted model."""
    u=_dev(issue)
    if config is None:return {'status':'BLOCKED','reason':'B_NUMERIC_THRESHOLD_CONFIG_MISSING'}
    if not isinstance(config,dict):raise ValueError('B_CONFIG_MAPPING_REQUIRED')
    if config.get('branch')!='STATE_B' or config.get('track')!='EXIT' or config.get('horizon_months')!=horizon:
        raise ValueError('B_OWNER_BRANCH_TRACK_HORIZON_MISMATCH')
    for k in ('owner_id','threshold_id','version','evidence_id'):
        if not config.get(k):return {'status':'BLOCKED','reason':'B_MISSING_'+k.upper()}
    if not config.get('issued_at'):return {'status':'BLOCKED','reason':'B_CONFIG_ISSUE_MISSING'}
    if day(config['issued_at'])>u:raise ValueError('B_FUTURE_THRESHOLD_CONFIG')
    required=('min_anchor_age_days','max_anchor_age_days','min_distinct_states',
              'max_latest_gap_days','min_state_rank','min_raw_sources',
              'min_sources_with_native_periods','max_confirmation_span_days')
    rules=config.get('rules')
    if not isinstance(rules,dict) or set(required)!=set(rules):
        return {'status':'BLOCKED','reason':'B_RULES_INCOMPLETE_OR_EXTRA'}
    for name in required:
        if isinstance(rules[name],bool) or not isinstance(rules[name],int) or rules[name]<0:
            raise ValueError('B_INVALID_RULE_'+name)
    if rules['min_anchor_age_days']>rules['max_anchor_age_days']:
        raise ValueError('B_AGE_RANGE_INVERTED')
    if not 1<=rules['min_distinct_states']<=4:raise ValueError('B_STATES_RANGE')
    if not 1<=rules['min_state_rank']<=4:raise ValueError('B_RANK_RANGE')
    if not 1<=rules['min_raw_sources']<=7:raise ValueError('B_RAW_SOURCE_RANGE')
    if not 1<=rules['min_sources_with_native_periods']<=7:raise ValueError('B_NATIVE_SOURCE_RANGE')
    sources=config.get('sources')
    if not isinstance(sources,dict) or set(sources)!=SOURCES:
        return {'status':'BLOCKED','reason':'B_ALL_SEVEN_SOURCE_THRESHOLD_CARDS_REQUIRED'}
    for src,card in sources.items():
        if not isinstance(card,dict) or set(card)!=set(('raw_unit','exit_direction','min_abs_signed_delta')):
            return {'status':'BLOCKED','reason':'B_SOURCE_CARD_INCOMPLETE:'+src}
        if not isinstance(card['raw_unit'],str) or not card['raw_unit']:
            raise ValueError('B_RAW_UNIT_REQUIRED:'+src)
        if card['exit_direction'] not in (-1,1) or isinstance(card['exit_direction'],bool):
            raise ValueError('B_EXIT_DIRECTION_REQUIRED:'+src)
        _num(card['min_abs_signed_delta'],'SOURCE_'+src)
    return {'status':'OK','rules':rules,'sources':sources,
            'fit_status':'NOT_FITTED_BY_REV300',
            'caller_fit_claim':config.get('fit_status','UNVERIFIED'),
            'owner_id':config['owner_id'],'threshold_id':config['threshold_id']}


def _current_epoch_events(controller:dict) -> list[dict]:
    """Only issued post-reset confirmations from current active state cycle."""
    reset=controller.get('last_reescalation_at')
    reset_occ=controller.get('last_reescalation_occurred')
    out=[]
    for event in controller.get('events',()):
        if event['state'] not in STATE_RANK:continue
        if reset is not None and day(event['available_at'])<=day(reset):continue
        if reset_occ is not None and day(event['occurred_at'])<day(reset_occ):continue
        out.append(event)
    return out


def _select_anchor(controller:dict) -> dict | None:
    epoch=controller.get('state_epoch')
    eligible=[a for a in controller.get('anchors',()) if a.get('epoch')==epoch]
    return min(eligible,key=lambda a:(a['first_issued_at'],a['source'],a['anchor_id'])) if eligible else None


def _distinct_native_periods(events:list[dict]) -> dict[str,set]:
    d=defaultdict(set)
    for e in events:
        if e.get('native_period'):
            d[e['source']].add(str(e['native_period']))
    return d


def _raw_evidence(events:list[dict],cards:dict) -> tuple[dict,dict]:
    """One most-recent native signed measurement per EWI, never 7x carried values."""
    newest={}
    for e in events:
        src=e['source']
        key=(str(e['available_at']),str(e['occurred_at']),str(e.get('issue_id','')))
        if src not in newest or key>newest[src][0]:newest[src]=(key,e)
    votes={}; details={}
    for src in EWI:
        e=newest.get(src,(None,None))[1]
        if e is None or e.get('raw_delta') is None or not e.get('native_period'):
            votes[src]=not_evaluated('NO_NATIVE_RAW_DELTA');details[src]='NO_NATIVE_RAW_DELTA';continue
        card=cards[src]
        if e.get('raw_unit')!=card['raw_unit']:
            raise ValueError('B_SOURCE_UNIT_MISMATCH:'+src)
        delta=_num(e['raw_delta'],'RAW_'+src,nonnegative=False)
        signed=card['exit_direction']*delta
        threshold=float(card['min_abs_signed_delta'])
        votes[src]='PASS' if signed>=threshold else 'FAIL'
        details[src]={'signed_exit_delta':signed,'threshold':threshold,
                      'native_period':str(e['native_period']),'raw_unit':card['raw_unit']}
    return votes,details


def _b_nonvote(status, reason):
    return {'branch':'STATE_B','execution_status':status,'status':status,'reason':reason}


def native_source_gate(n_native: int, n_unverifiable: int, k: int):
    """EXIT-B distinct-native-source gate (binary with partial-input determination).

    PASS iff n_native >= K; FAIL iff n_native + n_unverifiable < K; otherwise not determined.  A source with
    no current-epoch confirmation is a computed non-confirmation (not missing data), so with all seven source
    statuses computable and n_native = 3 < K = 5 the gate is a genuine FAIL (REV373 returned U)."""
    if n_native >= k:
        return 'PASS'
    if n_native + n_unverifiable < k:
        return 'FAIL'
    return not_evaluated('NATIVE_SOURCE_COUNT_NOT_DETERMINED')


def branch_B(*, issue, horizon, controller, config):
    """Computes Exit B from FIRST issued state, state-interval clocks, 7 RAW cards.

    A is neither received nor read. No learned claim: any PASS is a rule vote.
    Other independent B fitting/evaluation must still be executed on real labels.
    """
    u=_dev(issue)
    if horizon not in EXIT_HORIZONS:raise ValueError('INVALID_EXIT_HORIZON')
    validated=validate_B_threshold(issue=u,horizon=horizon,config=config)
    if validated['status']!='OK':return _b_nonvote('BLOCKED',validated['reason'])
    if controller.get('status')!='OK':return _b_nonvote('NOT_EVALUATED','NO_ACTIVE_EXIT_STATE')
    anchor=_select_anchor(controller)
    if not anchor:return _b_nonvote('NOT_EVALUATED','NO_FIRST_STATE')
    events=_current_epoch_events(controller)
    if not events:return _b_nonvote('NOT_EVALUATED','NO_CURRENT_EPOCH_CONFIRMATIONS')
    rules=validated['rules']
    first=day(anchor['first_issued_at'])
    latest=max(day(e['available_at']) for e in events)
    age=(u-first).days
    gap=(u-latest).days
    span=(latest-first).days
    if min(age,gap,span)<0:raise ValueError('B_EVENT_AFTER_ISSUE')
    native=_distinct_native_periods(events)
    raw_votes,detail=_raw_evidence(events,validated['sources'])
    n_native=sum(bool(native.get(src)) for src in EWI)
    # a source whose current-epoch events carry no native-period metadata has an unverifiable native status
    n_unverifiable=sum(1 for src in EWI
                       if not native.get(src) and any(e['source']==src for e in events))
    states={e['state'] for e in events}
    ranks=[STATE_RANK[e['state']] for e in events]
    gates={
        'anchor_age':('PASS' if rules['min_anchor_age_days']<=age<=rules['max_anchor_age_days'] else 'FAIL'),
        'latest_gap':('PASS' if gap<=rules['max_latest_gap_days'] else 'FAIL'),
        'state_breadth':('PASS' if len(states)>=rules['min_distinct_states'] else 'FAIL'),
        'state_rank':('PASS' if max(ranks)>=rules['min_state_rank'] else 'FAIL'),
        'native_sources':native_source_gate(n_native,n_unverifiable,rules['min_sources_with_native_periods']),
        'confirmation_span':('PASS' if span<=rules['max_confirmation_span_days'] else 'FAIL'),
    }
    raw=decision_interval(raw_votes,rules['min_raw_sources'])
    gates['raw_signed_K']=(raw['vote'] if raw['execution_status']=='EXECUTED'
                           else not_evaluated(raw['reason']))
    # Conjunction of *different* functional constraints, source K handled separately.
    vote=decide_and(*gates.values())
    g_votes,g_ne=_vote_map(gates)
    r_votes,r_ne=_vote_map(raw_votes)
    out={'branch':'STATE_B','horizon_months':horizon,
         'issue':u.isoformat(),'target':add_months(u,horizon).isoformat(),
         'anchor_id':anchor['anchor_id'],'earliest_state':anchor['state'],
         'anchor_at':anchor['first_issued_at'],'current_epoch':controller['state_epoch'],
         'gates':g_votes,'gates_not_evaluated':g_ne,
         'raw_source_votes':r_votes,'raw_sources_not_evaluated':r_ne,'raw_count_bounds':raw,
         'raw_details':detail,'diagnostics':{'anchor_age_days':age,
         'latest_gap_days':gap,'confirmation_span_days':span,'distinct_states':len(states),
         'distinct_native_sources':n_native,'unverifiable_native_sources':n_unverifiable,
         'current_epoch_event_count':len(events)},
         'threshold_id':validated['threshold_id'],'owner_id':validated['owner_id'],
         'fit_status':validated['fit_status'],
         'caller_fit_claim':validated['caller_fit_claim'],
         'calibrated_probability':'NOT_AVAILABLE'}
    if is_nonvote(vote):
        return {**out,'execution_status':vote.status,'status':vote.status,'reason':vote.reason}
    return {**out,'vote':vote,'execution_status':'EXECUTED','status':'RESEARCH_RULE_VOTE_UNFITTED'}


def _fusion_input(x, name):
    if x is None:
        return not_evaluated(name+'_NOT_ISSUED')
    if is_nonvote(x):
        return x
    if x == 'U':
        raise ThirdVoteError('LEGACY_U_FUSION_INPUT_REJECTED')
    if x not in VOTES:
        raise ValueError('FUSION_VOTE_DOMAIN')
    return x


def counterfactual_fusions(A,B):
    """Diagnostic decisions ONLY; not active policy without authorizing fusion.

    Inputs: PASS/FAIL, a NonVote, or None (branch not issued).  Outputs: binary or NonVote (no third token)."""
    a,b=_fusion_input(A,'A'),_fusion_input(B,'B')
    return {'A_ONLY':a,'B_ONLY':b,'AND':decide_and(a,b),'OR':decide_or(a,b)}


def fusion_fields(fusions: dict) -> dict:
    """Serialise fusion diagnostics: determined votes plus the list of non-determined fusions."""
    return {'fusions_DIAGNOSTIC_ONLY':{k:v for k,v in fusions.items() if not is_nonvote(v)},
            'fusions_not_determined':sorted(k for k,v in fusions.items() if is_nonvote(v))}


def current_operational_bridge(*, issue, rq01_regime, state_controller,
                               first_n_confirmed=False, horizon_months=None):
    """One-sided Stress shield is NOT a new RQ01 N/D/G/BOTH label.

    Issued EWI state confirmation holds an operational Stress protection after
    first state until owner-confirmed first-N, even if A/B predict Exit at T.
    On RQ01 N without first-N proof, return OWNER_CONFLICT (fail closed).
    Under 3 months output current RQ01 regime only; no future regime is forecast by this bridge.
    A non-issued RQ01 regime is a NonVote (never N, never a source) and yields no shield.
    """
    _dev(issue)
    if rq01_regime == 'U':raise ThirdVoteError('LEGACY_U_RQ01_REGIME_REJECTED_USE_NONVOTE')
    if not (is_nonvote(rq01_regime) or rq01_regime in (*STRESS,'N')):raise ValueError('RQ01_REGIME_DOMAIN')
    if horizon_months is not None and horizon_months not in (1,2,3,6,12,18):
        raise ValueError('INVALID_EXIT_OR_BRIDGE_HORIZON')
    active=(state_controller.get('status')=='OK' and _select_anchor(state_controller) is not None)
    if first_n_confirmed:
        if rq01_regime!='N':raise ValueError('FIRST_N_RQ01_REGIME_CONFLICT')
        policy='NORMAL_CONFIRMED';shield=False
    elif active and rq01_regime in STRESS:
        policy='STRESS_SHIELD';shield=True
    elif active and rq01_regime=='N':
        policy='OWNER_CONFLICT_ESCALATE_STRESS_SHIELD';shield=True
    elif rq01_regime in STRESS:
        policy='RQ01_STRESS_CURRENT';shield=True
    elif rq01_regime=='N':
        policy='NORMAL_CURRENT_ONLY';shield=False
    else:policy='NO_ISSUED_RQ01_REGIME_NO_SHIELD';shield=False
    return {**regime_fields(rq01_regime,'rq01_current_regime'),'operational_risk_policy':policy,
            'stress_shield':shield,'forecast_future_regime_status':'NOT_ISSUED_BY_BRIDGE',
            'rq02_sub_H3_prediction':'NOT_DEFINED' if horizon_months in (1,2) else 'NOT_APPLICABLE',
            'rq01_label_not_overwritten':True}


def dual_exit_issue(*, issue, country, episode_id, onset, rq01_view,
                    state_events=(), first_n=None, follower_obligations=(),
                    a_by_horizon=None,b_config_by_horizon=None,a_k_by_horizon=None,
                    a_k_owner_by_horizon=None):
    """A,B run on same issue/country/episode/target and never gate one another."""
    u=_dev(issue)
    ctrl=parallel_exit_controller(issue=u,country=country,episode_id=episode_id,
          onset=onset,rq01_view=rq01_view,state_events=state_events,
          first_n=first_n,follower_obligations=follower_obligations)
    regime=_check_predicted_rq01_view(rq01_view,u)
    if ctrl.get('status')=='FIRST_N_CONFIRMED':
        bridge=current_operational_bridge(issue=u,rq01_regime=regime,
                state_controller=ctrl,first_n_confirmed=True)
        return {'status':'EXIT_ALREADY_OCCURRED','controller':ctrl,'branches':(),
                'operational':bridge}
    if ctrl.get('status')!='OK':
        # A/B cannot be issued if the owner reports N without first-N proof.
        # But a verifiable EWI confirmation still imposes a separate defensive
        # owner-CONFLICT notice, never a fabricated change to the RQ01 label.
        reason=ctrl.get('reason',ctrl.get('status','NO_RISK_SET'))
        protective={**regime_fields(regime,'rq01_current_regime'),
                    'forecast_future_regime_status':'NOT_ISSUED_BY_BRIDGE',
                    'operational_risk_policy':'NO_ACTIVE_EXIT_RISK_SET_NO_SHIELD','stress_shield':False,
                    'rq01_label_not_overwritten':True}
        if regime=='N' and reason=='RQ01_N_FIRST_N_COMPLETION_PROOF_REQUIRED' and onset is not None:
            start=_onset_confirmed_asof(onset,u)
            audit_events=_verified_events(state_events=state_events,issue=u,
                               onset_date=start,country=country,episode_id=episode_id)
            if any(e['state'] in STATE_RANK for e in audit_events):
                protective['operational_risk_policy']='OWNER_CONFLICT_ESCALATE_STRESS_SHIELD'
                protective['stress_shield']=True
        status=ctrl.get('status') if ctrl.get('status') in ('BLOCKED','NOT_EVALUATED') else 'NOT_EVALUATED'
        return {'status':status,'reason':reason,'branches':(),
                'controller':ctrl,'operational':protective}
    result=[]
    for h in EXIT_HORIZONS:
        a=branch_A(issue=u,horizon=h,components=(a_by_horizon or {}).get(h),
                 k=(a_k_by_horizon or {}).get(h,1),
                 k_owner_id=(a_k_owner_by_horizon or {}).get(h))
        b=branch_B(issue=u,horizon=h,controller=ctrl,
                   config=(b_config_by_horizon or {}).get(h))
        aa,bb=a.get('vote'),b.get('vote')          # absent = branch decision not executed (no vote)
        row={'issue':u.isoformat(),'target':add_months(u,h).isoformat(),
             'horizon_months':h,'country':country,'episode_id':episode_id,
             'A_execution_status':a['execution_status'],'B_execution_status':b['execution_status'],
             'A_evidence':a,'B_evidence':b,
             **fusion_fields(counterfactual_fusions(aa,bb)),
             'active_combined_policy':'UNAUTHORIZED',
             'disagreement':('NOT_COMPARABLE' if aa is None or bb is None
                             else ('DISAGREE' if aa!=bb else 'AGREE'))}
        if aa is not None:row['A_exit_vote']=aa
        if bb is not None:row['B_exit_vote']=bb
        result.append(row)
    bridge=current_operational_bridge(issue=u,rq01_regime=regime,
                                      state_controller=ctrl)
    return {'status':'RESEARCH_RULES_ONLY','controller':ctrl,
            'operational':bridge,'branches':tuple(result),
            'target_definition':'FIRST_N_EXACTLY_AT_TARGET_MONTH',
            'official_FNR_FPR':'NOT_EVALUATED'}


def exact_first_n_target(*, issue, horizon, first_n_date, stress_at_issue:bool,
                         label_complete_through=None):
    """Gold-only label builder; NEVER a prediction feature.

    first_n_date None means NO_EXIT_WITHIN_DOCUMENTED_EPISODE; with incomplete follow-up the label is
    LABEL_CENSORED (NOT_EVALUATED at the scorer, never 0).
    """
    u=_dev(issue)
    if horizon not in EXIT_HORIZONS:raise ValueError('HORIZON_NOT_REGISTERED')
    if stress_at_issue is not True:raise ValueError('NOT_IN_STRESS_RISK_SET')
    target=add_months(u,horizon)
    if first_n_date is None:
        # EXACT-month label: a negative needs follow-up through the END of the target month (an N on the
        # last days of that month would otherwise be missed); missing follow-up proves nothing.
        if label_complete_through is None or day(label_complete_through)<_month_end(target):
            return LABEL_CENSORED
        return 0
    e=day(first_n_date)
    if e<=u:raise ValueError('EXIT_ALREADY_OCCURED_BY_ISSUE')
    return int((e.year,e.month)==(target.year,target.month))


def evaluate_matched_exit(*, records, labels, fusions=('A_ONLY','B_ONLY','AND','OR')):
    """Same keyed eligible population for A/B/fusions; a non-issued decision is counted, never dropped or TN.

    Labels: { (country,episode_id,issue,horizon) : 0/1 }.
    Scores are **development diagnostics**, not OOS certification.
    """
    seen=set();rows=[]
    for r in records:
        key=(r['country'],r['episode_id'],r['issue'],r['horizon_months'])
        if key in seen:raise ValueError('DUPLICATE_ISSUED_KEY')
        seen.add(key)
        if key not in labels:continue
        y=labels[key]
        if y not in (0,1) or isinstance(y,bool):raise ValueError('EXIT_GOLD_NOT_BINARY')
        av,bv=r.get('A_exit_vote'),r.get('B_exit_vote')
        rows.append((key,y,counterfactual_fusions(av,bv)))
    scores={}
    for name in fusions:
        if name not in ('A_ONLY','B_ONLY','AND','OR'):raise ValueError('UNKNOWN_FUSION')
        tp=fp=tn=fn=up=un=0
        for key,y,f in rows:
            v=f[name]
            if is_nonvote(v):                     # no decision issued: separate counter, never a class
                if y:up+=1
                else:un+=1
            elif y and v=='PASS':tp+=1
            elif y:fn+=1
            elif v=='PASS':fp+=1
            else:tn+=1
        npos=tp+fn+up;nneg=fp+tn+un
        scores[name]={'TP':tp,'FN':fn,'FP':fp,'TN':tn,'N_NOT_EVALUATED_pos':up,'N_NOT_EVALUATED_neg':un,
            'n_total':len(rows),'n_country_episodes':len({(k[0],k[1]) for k,_,__ in rows}),
            'FNR_evaluated':fn/(tp+fn) if tp+fn else None,
            'FPR_evaluated':fp/(fp+tn) if fp+tn else None,
            'FNR_worst_case_not_evaluated':(fn+up)/npos if npos else None,
            'FPR_worst_case_not_evaluated':(fp+un)/nneg if nneg else None,
            'PPV_evaluated':tp/(tp+fp) if tp+fp else None,
            'formal_PASS':'NOT_EVALUATED_IF_ISSUED_PROOF_ABSENT'}
    return {'status':'RESEARCH_MATCHED_DEVELOPMENT_DIAGNOSTIC',
            'n_matched_keys':len(rows),'n_gold_total':len(labels),
            'scores':scores,'target':'FIRST_N_EXACTLY_AT_TARGET_MONTH',
            'warning':'A non-issued decision is not TN; adaptive DEV scores are NOT independent validation.'}


def finite_breakpoint_geometry(*, observed_values, comparator='>='):
    """Exact M01 geometry: one decision mask per unique empirical cutoff.

    Exhaustiveness only for THIS one-dimensional fixed-score threshold family.
    No label, OOS result, episode weight or favorable threshold is inferred.
    """
    if comparator not in ('>=','<='):raise ValueError('COMPARATOR_MUST_BE_GE_OR_LE')
    v=tuple(_num(x,'BREAKPOINT',nonnegative=False) for x in observed_values)
    if not v:return {'status':'NO_NUMERIC_VALUES','breakpoints':(),'masks':()}
    cuts=tuple(sorted(set(v)))
    masks=[];seen=set()
    for threshold in cuts:
        mask=tuple(int(x>=threshold if comparator=='>=' else x<=threshold) for x in v)
        if mask in seen:raise RuntimeError('BREAKPOINT_MASK_NOT_UNIQUE')
        seen.add(mask)
        masks.append({'threshold':threshold,'mask':mask,'passing':sum(mask)})
    # Add feasible all-zero mask at immediate threshold past maximum/minimum.
    from math import nextafter, inf
    end=nextafter(max(cuts),inf) if comparator=='>=' else nextafter(min(cuts),-inf)
    mask=tuple(int(x>=end if comparator=='>=' else x<=end) for x in v)
    if mask not in seen:masks.append({'threshold':end,'mask':mask,'passing':sum(mask)})
    return {'status':'EXACT_FIXED_SCORE_GEOMETRY','n_observations':len(v),
       'breakpoints':cuts,'masks':tuple(masks),'n_unique_masks':len(masks),
       'warning':'No label, no matched loss, no fitted threshold. Only scalar geometry.'}


def first_n_label(*, issue, horizon, first_n_date, stress_at_issue,
                  label_complete_through=None, mode='FIRST_N_EXACT_MONTH'):
    """Explicit target contrast: active benchmark EXACT vs archived BY horizon.

    Missing follow-up yields LABEL_CENSORED. Gold labels are NEVER handed to B predictor.
    """
    if mode not in ('FIRST_N_EXACT_MONTH','FIRST_N_BY_HORIZON'):
        raise ValueError('EXIT_TARGET_MODE_UNAUTHORIZED')
    u=_dev(issue)
    if horizon not in EXIT_HORIZONS:raise ValueError('EXIT_HORIZON_NOT_REGISTERED')
    if stress_at_issue is not True:raise ValueError('NO_STRESS_RISK_SET')
    t=add_months(u,horizon)
    if first_n_date is None:
        need=_month_end(t) if mode=='FIRST_N_EXACT_MONTH' else t   # exact-month needs the whole target month
        return 0 if label_complete_through and day(label_complete_through)>=need else LABEL_CENSORED
    end=day(first_n_date)
    if end<=u:raise ValueError('FIRST_N_ALREADY_OCCURRED')
    if mode=='FIRST_N_EXACT_MONTH':return int((end.year,end.month)==(t.year,t.month))
    return int(end<=t)
