"""RQ02 REV279: ONSET/EXIT-only issue-time controller; never ex-post gold as feature.

Research-only adapter. Active RQ02 targets: ONSET and EXIT. RQ01 Peak,
De-escalation, Trough, Recovery are INSPECTION STATES, NOT RQ02 targets.
No fitted probabilistic claims or historical prediction rewrites.

GLOBAL NO-U: an issued RQ01 regime is one of N/D/G/BOTH.  A view without an issued regime is a non-vote
record (NonVote) that no consumer may read as N or as a source.  The historical owner tokens U/UNAVAILABLE
(File 01 H5 frozen Owner C) are accepted only through the read-only compatibility boundary
`vote_contract.normalize_frozen_owner_c` and are never rewritten.
"""
from __future__ import annotations
from datetime import date, datetime
from calendar import monthrange
from math import isfinite

from research_adapters.vote_contract import (FROZEN_OWNER_C_LEGACY_TOKENS, NON_VOTE_STATUSES, NonVote,
                                             is_nonvote, normalize_frozen_owner_c)

TRACKS = frozenset({'ONSET', 'EXIT'})
SOURCES = frozenset({'CISS','CLIFS','DFC','GFC','VIX','CG','DSR-P'})
STATES = frozenset({'ENTRY','PEAK','DEESCALATION','TROUGH','RECOVERY',
                    'REESCALATION','EXIT'})
REGIMES = frozenset({'N','D','G','BOTH'})          # issued RQ01 regimes only (no third 'U' class)


def regime_fields(regime, prefix='rq01_regime'):
    """Serialise an issued regime (str) or a non-issued RQ01 view (NonVote) without a third class."""
    if is_nonvote(regime):
        return {prefix: None, prefix + '_status': regime.status, prefix + '_reason': regime.reason}
    if regime is None:
        return {prefix: None, prefix + '_status': 'NOT_EVALUATED', prefix + '_reason': 'RQ01_REGIME_NOT_ISSUED'}
    if regime not in REGIMES:
        raise ValueError('RQ01_REGIME_DOMAIN:' + repr(regime))
    return {prefix: regime, prefix + '_status': 'ISSUED', prefix + '_reason': None}


def day(x):
    if isinstance(x, datetime): return x.date()
    if isinstance(x, date): return x
    return date.fromisoformat(x)


def add_months(d, n):
    d = day(d)
    z = 12*d.year + d.month-1+n
    year, month = divmod(z,12)
    year, month = year,month+1
    last = monthrange(year,month)[1]
    # Issues are monthly: a month-END date stays a month-END date (Feb 28 + 1 month = Mar 31);
    # any other day keeps its day number clamped to the target month.
    if d.day == monthrange(d.year,d.month)[1]:
        return date(year,month,last)
    return date(year,month,min(d.day,last))


def month_distance(a,b):
    a,b=day(a),day(b)
    return 12*(b.year-a.year) + b.month-a.month


def active_track(track):
    if track not in TRACKS:
        raise ValueError('RQ02_ONLY_ONSET_EXIT__INVALID_TRACK:'+str(track))
    return track


def valid_issue_confirmation(event, issue):
    """Require *issued* RQ01 provenance and real available_at; not retrospective ESRB gold."""
    issue=day(issue)
    if event.get('source') not in SOURCES: raise ValueError('SOURCE_NOT_7EWI')
    if event.get('state') not in STATES: raise ValueError('UNKNOWN_RQ01_INSPECTION_STATE')
    for f in ('occurred_at','available_at','source_hash','issue_id'):
        if not event.get(f): raise ValueError('MISSING_RQ01_ISSUED_PROOF:'+f)
    if event.get('confirmed_by') != 'RQ01_ISSUED' or event.get('release_verified') is not True:
        raise ValueError('RQ01_STATE_NOT_ISSUED_VERIFIED')
    occurred, available=day(event['occurred_at']), day(event['available_at'])
    if occurred>available: raise ValueError('ISSUED_BEFORE_SOURCE_EVENT')
    if available>issue: raise ValueError('FUTURE_RQ01_STATE_LEAK')
    if not (date(2005,1,1)<=issue<=date(2019,12,31)):
        raise ValueError('DEV_ONLY__OOS_SEALED')
    return occurred,available


def _check_predicted_rq01_view(view, issue):
    """Issued regime (N/D/G/BOTH) or a NonVote for a provenance-verified view that carries no issued regime."""
    if not isinstance(view,dict):
        raise ValueError('RQ01_ISSUED_VIEW_REQUIRED')
    regime=view.get('regime')
    status=view.get('issue_status')
    legacy=isinstance(regime,str) and regime.strip().upper() in FROZEN_OWNER_C_LEGACY_TOKENS
    if regime not in REGIMES and status not in NON_VOTE_STATUSES and not legacy:
        raise ValueError('RQ01_ISSUED_VIEW_REQUIRED')
    if view.get('available_at') is None or not view.get('source_hash'):
        raise ValueError('RQ01_VIEW_PROVENANCE_MISSING')
    if day(view['available_at'])>day(issue): raise ValueError('FUTURE_RQ01_VIEW_LEAK')
    if not view.get('issue_verified'):raise ValueError('RQ01_VIEW_NOT_VERIFIED')
    if regime in REGIMES:
        return regime
    if legacy:
        # immutable historical owner token: read-only boundary, never an N/D/G/BOTH guess
        norm=normalize_frozen_owner_c(regime)
        return NonVote(norm['execution_status'],norm['reason'],(norm['owner_conflict'],))
    return NonVote(status,view.get('issue_reason') or 'RQ01_VIEW_HAS_NO_ISSUED_REGIME')


def _onset_confirmed_asof(onset, issue):
    if onset is None:return None
    if onset.get('confirmed_by') != 'RQ01_ISSUED' or onset.get('release_verified') is not True:
        raise ValueError('ONSET_MUST_BE_ISSUED_NOT_ESRB_EXPOST')
    for f in ('occurred_at','available_at','source_hash','issue_id'):
        if not onset.get(f): raise ValueError('MISSING_ONSET_ISSUED_PROOF:'+f)
    t,a=day(onset['occurred_at']),day(onset['available_at'])
    if t>a:raise ValueError('ONSET_BEFORE_AVAILABILITY_INVALID')
    if a>day(issue):raise ValueError('FUTURE_ONSET_CONFIRMATION')
    if onset.get('regime') not in ('D','G','BOTH'):raise ValueError('ONSET_REGIME_UNRESOLVED')
    return t


def phase_controller(*, issue, rq01_view, onset=None, state_events=(),
                     expected_followers=(), exit_horizon_months=3,
                     eligible_peak_sources=None, exit_completion=None):
    """Issue-time controller with explicit separation of current state vs future forecast.

    ONSET forecast stops when the first verified issued RQ01 Onset confirmation
    becomes known. EXIT gate is armed after that confirmation, but active EXIT
    requires all registered eligible sources to have genuinely confirmed PEAK,
    with no issue-valid immature follower expectation and no new phase at issue.

    Forecasts issued earlier remain immutable even when a later state arrives.
    This function never inspects future state events or future labels.
    """
    issue=day(issue)
    if not date(2005,1,1)<=issue<=date(2019,12,31):
        raise ValueError('DEV_ONLY__OOS_SEALED')
    if not isinstance(exit_horizon_months,int) or not 3<=exit_horizon_months<=18:
        raise ValueError('EXIT_HORIZON_MUST_BE_3_TO_18_CALENDAR_MONTHS')
    current=_check_predicted_rq01_view(rq01_view,issue)
    if onset is None:
        if state_events or expected_followers:raise ValueError('PHASE_WITHOUT_CONFIRMED_ONSET')
        return dict(active_track='ONSET', controller='FORECAST_ONSET',
                    exit_gate_armed=False, forecast_target=None,
                    **regime_fields(current,'rq01_issued_current_regime'), current_regime_owner='RQ01',
                    eligible_peak_sources=[], missing_peak_sources=[],
                    pending_followers=[], latest_confirmed_event_at=None,
                    note='Future onset forecast is separate, issued by Onset model')
    entry_t=_onset_confirmed_asof(onset,issue)
    if exit_completion is not None:
        for f in ('occurred_at','available_at','source_hash','issue_id'):
            if not exit_completion.get(f): raise ValueError('EXIT_COMPLETION_ISSUED_PROOF_MISSING:'+f)
        if (exit_completion.get('regime')!='N' or
            exit_completion.get('confirmed_by')!='RQ01_ISSUED' or
            exit_completion.get('release_verified') is not True):
            raise ValueError('EXIT_COMPLETION_NOT_AUTHENTIC_ISSUED_FIRST_N')
        ended=day(exit_completion['occurred_at'])
        released=day(exit_completion['available_at'])
        if not (entry_t<=ended<=released<=issue):
            raise ValueError('EXIT_COMPLETION_FUTURE_OR_PRE_ONSET')
        # Next risk cycle resumes ONSET; actual prior issued Exit predictions
        # remain immutable and score separately against matured first-N gold.
        return dict(active_track='ONSET',controller='ONSET_REARMED_AFTER_ISSUED_EXIT',
                    exit_gate_armed=False,forecast_target=None,
                    **regime_fields(current,'rq01_issued_current_regime'),current_regime_owner='RQ01',
                    eligible_peak_sources=[],missing_peak_sources=[],pending_followers=[],
                    latest_confirmed_event_at=released.isoformat(),
                    note='RQ01-issued first-N ended prior Stress; do not retro rewrite prior Exit predictions')
    events=[];seen=set()
    for ev in state_events:
        occurred, available=valid_issue_confirmation(ev,issue)
        if occurred<entry_t: raise ValueError('PRE_ONSET_PHASE_AS_NEW_POST_ONSET_EVENT')
        key=(ev['source'],ev['state'],occurred,available)
        if key in seen:raise ValueError('DUPLICATED_STATE_CONFIRMATION')
        seen.add(key)
        events.append(dict(ev, _occurred=occurred,_available=available))
    if eligible_peak_sources is None:
        eligible_peak_sources=SOURCES
    selected=set(eligible_peak_sources)
    if not selected or not selected<=SOURCES:
        raise ValueError('INVALID_ELIGIBLE_PEAK_SOURCES')
    got={e['source'] for e in events if e['state']=='PEAK'}
    missing=sorted(selected-got)
    pending=[]
    for f in expected_followers:
        for name in ('source','state','trigger_available_at','expected_by','evidence_id'):
            if not f.get(name):raise ValueError('INVALID_PENDING_FOLLOWER:'+name)
        if f['source'] not in SOURCES or f['state'] not in STATES:
            raise ValueError('INVALID_FOLLOWER_SOURCE_STATE')
        if day(f['trigger_available_at'])>issue:raise ValueError('FUTURE_PENDING_TRIGGER')
        # Registration is an issue-time signed forecast of prospective follower,
        # NOT an ex-post inferred state arrival. The last day is inclusive.
        if day(f['expected_by'])>=issue and not any(
            e['source']==f['source'] and e['state']==f['state']
            and e['_available']<=issue and e['_available']>=day(f['trigger_available_at'])
            for e in events):
            pending.append(f['source']+':'+f['state'])
    # A new confirmed event blocks an immediate Exit forecast at the same issue;
    # next issue can reassess, without altering older issued forecasts.
    last_available=max((e['_available'] for e in events),default=None)
    # Completing the last source PEAK may activate Exit at this same issued
    # issue, as requested. A fresh successor phase at the issue pauses Exit.
    new_post_peak_this_issue=any(e['_available']==issue and
                                 e['state'] not in ('PEAK','ENTRY') for e in events)
    eligible=not missing and not pending and not new_post_peak_this_issue
    return dict(active_track='EXIT' if eligible else None,
                controller='EXIT_FORECAST' if eligible else 'STRESS_HOLD',
                exit_gate_armed=True,
                forecast_target=add_months(issue,exit_horizon_months).isoformat() if eligible else None,
                **regime_fields(current,'rq01_issued_current_regime'),current_regime_owner='RQ01',
                onset_confirmed_source_regime=onset['regime'],
                # a disagreement is only defined against an ISSUED current regime (a non-issued view is not N)
                source_regime_at_issue_disagreement=(None if is_nonvote(current)
                                                     else current!=onset['regime']),
                eligible_peak_sources=sorted(selected), missing_peak_sources=missing,
                pending_followers=pending,
                latest_confirmed_event_at=last_available.isoformat() if last_available else None,
                note='Current t to t+h uses RQ01 issued view; future Exit is never an overwrite')


def score_exit_at_maturity(*, issued_forecast, observed_first_n, observed_at,
                          matured_at, asof):
    """Retrospective scoring only: no feature use of gold before target maturity."""
    issue=day(issued_forecast['issued_at'])
    t=day(issued_forecast['target_at'])
    if month_distance(issue,t)<3: raise ValueError('EXIT_BELOW_MINIMUM_3_MONTHS')
    if day(asof)<t or day(matured_at)>day(asof):return {'status':'NOT_MATURED'}
    if observed_first_n not in (True,False):return {'status':'NOT_EVALUATED'}
    if observed_first_n and (observed_at is None or day(observed_at)>t):
        return {'status':'INVALID_FIRST_N_LABEL_TIMING'}
    return {'status':'EVALUATED', 'issue':issue.isoformat(), 'target':t.isoformat(),
            'predicted_first_n':bool(issued_forecast['predicted_first_n']),
            'actual_first_n':observed_first_n,
            'correct':bool(issued_forecast['predicted_first_n'])==observed_first_n}


def conditional_next_state(*, cause_specific_hazards, support, model_provenance,
                           minimum_support=10):
    """Mutually exclusive first next RQ01 states; no fabricated fit or probabilities.

    Input hazard[h][state] must come from a real eligible matched fitted artifact;
    distribution of non-exclusive eventual events is NOT this object. Output
    next first-event probabilities and survivor mass by horizon.
    """
    if not model_provenance or model_provenance.get('fit_window')!='DEV_2005_2019' or not model_provenance.get('source_hash') or not model_provenance.get('fitted'):
        return {'status':'NOT_EVALUATED','reason':'NO_VERIFIED_FITTED_TRANSITION_ARTIFACT'}
    if not isinstance(support,int) or support<minimum_support:
        return {'status':'NOT_IDENTIFIABLE','reason':'INSUFFICIENT_DISTINCT_EPISODE_SUPPORT'}
    survival=1.0;out={};rows=[]
    for h, hazard in enumerate(cause_specific_hazards,1):
        if not hazard or any(k not in STATES for k in hazard):raise ValueError('INVALID_NEXT_STATE')
        vals=list(hazard.values())
        if any(not isinstance(x,(float,int)) or not isfinite(x) or x<0 or x>1 for x in vals):
            raise ValueError('INVALID_HAZARD_PROBABILITY')
        if sum(vals)>1+1e-12:raise ValueError('NEXT_STATE_HAZARDS_EXCEED_ONE')
        for k,v in hazard.items():out[k]=out.get(k,0.0)+survival*v
        rows.append({'h':h,'survival_at_start':survival,'cause_hazards':dict(hazard)})
        survival*=1-sum(vals)
    return {'status':'DEVELOPMENT_ONLY_NOT_CERTIFIED','next_event_mass':out,
            'survival_through_h':survival,'horizon_months':len(rows),
            'P_any_first_transition_by_h':1-survival,'rows':rows,
            'support':support}


def conditional_exit_timing_research(*, competing_risk_result, exit_horizons=(3,6,9,12,18),
                                     min_horizon=3):
    """Value of a future Exit target vs probability of a NEW inner-RQ01 state.

    Not a deployment rule: P(first EXIT by h) and P(first phase other than EXIT
    by h) are identifiable only with real issue-time fitted competing risks,
    explicit censoring and episode support. A controller must not select
    an Exit horizon based on these outputs unless an approved cost/threshold
    policy and matched evaluation exist.
    """
    if competing_risk_result.get('status')!='DEVELOPMENT_ONLY_NOT_CERTIFIED':
        return {'status':'NOT_EVALUATED','reason':'COMPETING_RISK_HAZARD_NOT_FITTED'}
    rows=competing_risk_result['rows'];cum=0.0;other=0.0;mass=[]
    for r in rows:
        survival=r['survival_at_start'];haz=r['cause_hazards'];
        cum+=survival*haz.get('EXIT',0.)
        other+=survival*sum(p for s,p in haz.items() if s!='EXIT')
        mass.append((cum,other))
    out=[]
    for h in exit_horizons:
        if not isinstance(h,int) or h<min_horizon:
            raise ValueError('EXIT_FORECAST_AT_LEAST_3_MONTHS')
        if h>len(mass):
            out.append({'h':h,'status':'NOT_EVALUATED','reason':'INSUFFICIENT_HAZARD_HORIZON'})
        else:
            ex,ph=mass[h-1]
            out.append({'h':h,'status':'RESEARCH_ONLY',
                        'P_first_exit_by_h':ex,
                        'P_new_RQ01_state_before_exit_by_h':ph,
                        'P_no_first_transition_by_h':1-ex-ph})
    return {'status':'RESEARCH_ONLY_NO_THRESHOLD_OR_POLICY_APPROVAL',
            'exit_horizon_candidates':out,
            'warning':'New phase and true first-N must be mutually exclusive FIRST events; no retrospective forecast rewrites'}
