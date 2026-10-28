"""REV280 | RQ02 ENTRY/EXIT ONLY, dual parallel Exit strategies.

DESIGN CONTRACT (research-only, NO new fitted parameters):
- A: every issue after verified onset, 18/12/6/3 month first-N horizon,
  IL/IS/JL/JS from the existing worker modules, frozen owner thresholds.
- B: issue-time first hit of *highest presently known* RQ01 progression state
  (Peak / De-escalation / Trough / Recovery), with durable earlier anchors.
  All issued forecasts continue; a new state holds CURRENT Stress, not a
  retrospective rewrite of any forecast. A later higher-ranked stage creates
  a new anchor WITHOUT deleting an earlier horizon projection.
- T-18/T-12/T-6/T-3 from the TRUE Exit is a retrospective sample JOIN only:
  using known Exit dates as live features or fitting on positives only is ILLEGAL.
- Current-state label at issue belongs to RQ01; RQ02 predicts future first-N.
- DEV 2005--2019 ONE BLOCK, OOS 2020--2024 SEALED for execution.
- Breakout/Recovery are NEVER RQ02 forecast tracks; Recovery is an optional
  RQ01 *observed state*, not a target.
- GLOBAL NO-U: every EXIT decision is binary PASS/FAIL or a non-vote record (NOT_EVALUATED / BLOCKED with a
  reason).  A family vote 'U' is rejected; a non-issued RQ01 view is a NonVote, never N; an issued-forecast
  ledger row for an undetermined decision carries prediction=None and issue_status=NOT_ISSUED.
"""
from __future__ import annotations
from calendar import monthrange
from datetime import date
from hashlib import sha256
import json
from research_adapters.vote_contract import (NON_VOTE_STATUSES, NonVote, ThirdVoteError, decide_and, decide_or,
                                             is_nonvote, not_evaluated)
from rq02_source.rq02_rev279_onset_exit_country.core import (
    active_track, _check_predicted_rq01_view, _onset_confirmed_asof,
    valid_issue_confirmation, day, add_months, month_distance, regime_fields,
)

EXIT_HORIZONS=(18,12,6,3)
EWI=('CISS','CLIFS','DFC','GFC','VIX','CG','DSR-P')
FAMILIES=('IL','IS','JL','JS')
# Candidate ordering is a PRE-DECLARED policy, NOT a fitted transition probability.
# Absence of an issued higher state must never be treated as evidence it won't occur.
STATE_RANK={'PEAK':1,'DEESCALATION':2,'TROUGH':3,'RECOVERY':4}
STRESS=frozenset(('D','G','BOTH'))


def _dev(t):
    d=day(t)
    if not date(2005,1,1)<=d<=date(2019,12,31):
        raise ValueError('DEV_ONLY_OOS_SEALED')
    return d


def _hash(value):
    return sha256(json.dumps(value,sort_keys=True,default=str,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _proof_onset(*, onset, issue, country, episode_id, rq01_view):
    u=_dev(issue)
    if not country or not episode_id: raise ValueError('COUNTRY_EPISODE_REQUIRED')
    active_track('EXIT')
    r=_check_predicted_rq01_view(rq01_view,u)
    if onset is None: return u,r,None
    start=_onset_confirmed_asof(onset,u)
    if start>u: raise ValueError('ONSET_FUTURE')
    if onset.get('country') not in (None,country) or onset.get('episode_id') not in (None,episode_id):
        raise ValueError('ONSET_COUNTRY_EPISODE_MISMATCH')
    return u,r,start


def _verified_events(*, state_events, issue, onset_date, country, episode_id):
    """Issue-as-of EWI state tape, de-duplicate signed firsthit identities."""
    u=_dev(issue)
    out=[]; seen=set(); firsthits=set()
    for e in state_events:
        if not isinstance(e,dict): raise ValueError('EVENT_MAPPING_REQUIRED')
        # On future events, prefix replay may supply the entire event tape. Never
        # inspect their state until their available_at is past; quarantine them.
        if e.get('available_at') and day(e['available_at'])>u: continue
        occurred,avail=valid_issue_confirmation(e,u)
        if onset_date is None or occurred<onset_date: raise ValueError('PHASE_OUTSIDE_CONFIRMED_EPISODE')
        if e.get('country') not in (None,country) or e.get('episode_id') not in (None,episode_id):
            raise ValueError('CROSS_EPISODE_STATE_INJECTION')
        identity=(e['source'],e['state'],occurred.isoformat(),avail.isoformat(),e['issue_id'])
        if identity in seen: raise ValueError('DUPLICATE_SOURCE_STATE')
        seen.add(identity)
        if e['state'] in STATE_RANK:
            firsthit=(e['source'],e['state'],occurred.isoformat())
            if firsthit in firsthits: raise ValueError('DUPLICATE_FIRSTHIT_VINTAGE')
            firsthits.add(firsthit)
        out.append({**e,'occurred_at':occurred.isoformat(),'available_at':avail.isoformat()})
    return sorted(out,key=lambda e:(e['available_at'],e['source'],e['state'],e['occurred_at']))


def state_anchor_ledger(*, issue, country, episode_id, onset, rq01_view, state_events=()):
    """MECE state anchors: first issued hit at each newly achieved *rank*.

    If all seven EWI have only Peak, earliest issued Peak is anchor. If a source
    first reaches Recovery later, preserve the Peak anchor, add Recovery anchor.
    Same-ranked followers never move the first rank anchor forward.
    Future-issued events are NOT read; raw occurrence vs availability separate.
    """
    u,regime,onset_date=_proof_onset(onset=onset,issue=issue,country=country,
                                   episode_id=episode_id,rq01_view=rq01_view)
    if onset_date is None:
        if state_events: raise ValueError('RQ01_STATES_WITHOUT_ONSET')
        return {'status':'ONSET_NOT_CONFIRMED','active_track':'ONSET','anchors':(),
                'selected_anchor':None,'events':(),**regime_fields(regime)}
    if is_nonvote(regime):
        # no issued RQ01 current state: the EXIT risk set cannot be armed (no vote, never read as N)
        return {'status':'NOT_EVALUATED','reason':'RQ01_ISSUED_CURRENT_STATE_NOT_ISSUED',
                'active_track':None,'anchors':(),'selected_anchor':None,'events':(),
                **regime_fields(regime)}
    if regime=='N':
        # owner conflict: RQ01 says N without an issued first-N completion proof
        return {'status':'BLOCKED','reason':'RQ01_N_FIRST_N_COMPLETION_PROOF_REQUIRED',
                'active_track':None,'anchors':(),'selected_anchor':None,'events':(),
                **regime_fields(regime)}
    events=_verified_events(state_events=state_events,issue=u,onset_date=onset_date,
                            country=country,episode_id=episode_id)
    anchors=[];highest=0;epoch=0;last_reescalation_at=None;last_reescalation_occurred=None
    for e in events:
        if e['state']=='REESCALATION':
            # A new cycle may start inside the SAME Stress episode; preserve
            # all old forecasts/anchors but reset the eligible newest state.
            epoch+=1;highest=0
            last_reescalation_at=e['available_at']
            last_reescalation_occurred=e['occurred_at']
            continue
        if last_reescalation_occurred and e['occurred_at']<last_reescalation_occurred:
            # Late publication of old-epoch state cannot restart progress after
            # an already issued re-escalation reset. Retain in history only.
            continue
        rank=STATE_RANK.get(e['state'],0)
        if rank>highest:
            highest=rank
            anchors.append({'state':e['state'],'rank':rank,'epoch':epoch,
                'source':e['source'],'occurred_at':e['occurred_at'],
                'first_issued_at':e['available_at'],
                'anchor_id':_hash((country,episode_id,epoch,e['state'],e['source'],
                                   e['occurred_at'],e['available_at']))[:20],
                'source_hash':e['source_hash'],'issue_id':e['issue_id']})
    eligible_now=[p for p in anchors if p['epoch']==epoch]
    return {'status':'OK','active_track':'EXIT','exit_gate_armed':True,
        'anchors':tuple(anchors),'selected_anchor':eligible_now[-1] if eligible_now else None,
        'state_epoch':epoch,'last_reescalation_at':last_reescalation_at,
        'last_reescalation_occurred':last_reescalation_occurred,
        'events':tuple(events),**regime_fields(regime),'onset_at':onset_date.isoformat(),
        'issue':u.isoformat(),'country':country,'episode_id':episode_id,
        'event_tape_hash':_hash(events)}


def _validate_exit_first_n(*, first_n, issue, onset_at, country, episode_id):
    if first_n is None: return False
    for k in ('occurred_at','available_at','source_hash','issue_id'):
        if not first_n.get(k):raise ValueError('FIRST_N_PROVENANCE_MISSING:'+k)
    if first_n.get('confirmed_by')!='RQ01_ISSUED' or first_n.get('release_verified') is not True or first_n.get('regime')!='N':
        raise ValueError('FIRST_N_NOT_ISSUED_CONFIRMED')
    if first_n.get('country') not in (None,country) or first_n.get('episode_id') not in (None,episode_id):
        raise ValueError('FIRST_N_CROSS_EPISODE')
    end,available=day(first_n['occurred_at']),day(first_n['available_at'])
    if not day(onset_at)<end<=available<=day(issue):
        raise ValueError('FIRST_N_PRE_ONSET_OR_UNISSUED')
    return True


def parallel_exit_controller(*, issue, country, episode_id, onset, rq01_view,
                             state_events=(),first_n=None, follower_obligations=()):
    """Always keep A horizon forecasts running; B starts at first confirmed Peak+.

    Follower obligations produce a persistent HOLD for current risk and issuance
    decisions, but do NOT suppress the forecast clock. Distinct control clocks:
    gate_armed, conventional_active, event_trigger_active, pending_follower,
    current_rq01_regime; no artificial third RQ02 forecast target.
    """
    # First-N completion MUST be handled before the ongoing Stress-state
    # validator: the RQ01 current view can legitimately be N on that issue.
    u0,current,onset_date=_proof_onset(onset=onset,issue=issue,country=country,
              episode_id=episode_id,rq01_view=rq01_view)
    if onset_date is not None and first_n is not None and _validate_exit_first_n(
             first_n=first_n,issue=u0,onset_at=onset_date,
             country=country,episode_id=episode_id):
        if current!='N':
            return {'status':'BLOCKED','reason':'FIRST_N_RQ01_CURRENT_CONFLICT',
                    'active_track':None,'exit_forecasts_continue':False,
                    'exit_gate_armed':False,'current_state_owner':'RQ01'}
        return {'status':'FIRST_N_CONFIRMED','active_track':'ONSET',
               'exit_forecasts_continue':False,'conventional_exit_active':False,
               'state_exit_active':False,'exit_gate_armed':False,
               'current_state_owner':'RQ01','rq01_regime':'N'}
    view=state_anchor_ledger(issue=issue,country=country,episode_id=episode_id,
                             onset=onset,rq01_view=rq01_view,state_events=state_events)
    if view['status']=='ONSET_NOT_CONFIRMED':
        return {**view,'conventional_exit_active':False,'state_exit_active':False,
                'exit_forecasts_continue':False,'current_state_owner':'RQ01'}
    if view['status']!='OK':
        return {**view,'conventional_exit_active':False,'state_exit_active':False,
                'exit_forecasts_continue':False,'current_state_owner':'RQ01'}
    u=day(issue)
    pending=[]
    for f in follower_obligations:
        for k in ('source','state','trigger_available_at','expected_by','evidence_id'):
            if not f.get(k):raise ValueError('FOLLOWER_PROOF_MISSING:'+k)
        if f['source'] not in EWI or f['state'] not in STATE_RANK:
            raise ValueError('FOLLOWER_SOURCE_STATE_INVALID')
        if day(f['trigger_available_at'])>u:continue
        if day(f['expected_by'])<day(f['trigger_available_at']):
            raise ValueError('FOLLOWER_DEADLINE_BEFORE_TRIGGER')
        confirmed=any(e['source']==f['source'] and e['state']==f['state']
              and day(e['available_at'])>=day(f['trigger_available_at'])
              for e in view['events'])
        if not confirmed and day(f['expected_by'])>=u:
            pending.append({'source':f['source'],'state':f['state'],
                            'expected_by':day(f['expected_by']).isoformat(),
                            'evidence_id':f['evidence_id']})
    recent=any(day(e['available_at'])==u and (e['state'] in STATE_RANK or e['state']=='REESCALATION') for e in view['events'])
    # A state confirmation or live pending follower protects the CURRENT stress,
    # but MUST NOT stop the two exit projection engines or delete any older issue.
    return {**view,'controller':'PARALLEL_EXIT_ISSUANCE',
            'conventional_exit_active':True,
            'state_exit_active':view['selected_anchor'] is not None,
            'exit_forecasts_continue':True,'current_state_owner':'RQ01',
            'persistent_pending_followers':tuple(pending),
            'new_state_at_issue':recent,'protect_current_stress':bool(recent or pending)}


def _ne(reason):
    return not_evaluated(reason), reason


def _family_proof(*, component, issue, horizon, family):
    """(binary family vote, None) or (NonVote, reason).  A legacy 'U' family vote is rejected."""
    if not isinstance(component,dict):return _ne('MISSING_FAMILY_COMPONENT')
    if component.get('track')!='EXIT' or component.get('family')!=family or component.get('horizon_months')!=horizon:
        raise ValueError('EXIT_FAMILY_TRACK_OR_HORIZON_MISMATCH')
    if not component.get('source_hash') or not component.get('threshold_owner_id') or not component.get('threshold_id'):
        return _ne('THRESHOLD_OR_SOURCE_PROVENANCE_MISSING')
    if component.get('released_verified') is not True or component.get('native_verified') is not True:
        return _ne('NATIVE_OR_ISSUE_NOT_VERIFIED')
    if not component.get('model_id') or component.get('fitted_provenance') is not True:
        return _ne('FIT_ARTIFACT_NOT_VERIFIED')
    if not component.get('available_at'):return _ne('MISSING_AVAILABLE_AT')
    if day(component['available_at'])>day(issue):raise ValueError('FUTURE_EXIT_FEATURE')
    # Joint input MUST be a purified, qualified family; gate1/2 proof no fallback.
    if family in ('JL','JS') and (component.get('gate2_qualified') is not True
            or component.get('pure_joint_verified') is not True):
        return _ne('PURE_JOINT_OR_GATE2_UNVERIFIED')
    v=component.get('vote')
    if v in ('PASS','FAIL'):
        return v,None
    if v is None and component.get('execution_status') in NON_VOTE_STATUSES:
        return _ne(component.get('reason') or 'FAMILY_COMPONENT_NOT_EVALUATED')
    if v == 'U':
        raise ThirdVoteError('LEGACY_U_FAMILY_VOTE_REJECTED')
    raise ValueError('FAMILY_VOTE_DOMAIN')


def exit_four_family_gate(*, issue, horizon_months, family_components,
                          policy='ANY_PASS'):
    """Output a candidate first-N flag from ORIGINAL Exit Long/Short thresholds.

    Inputs are fitted and signed worker-family artifacts, NOT fabricated raw
    thresholds. Missing family is never imputed 0 (it is a non-evaluable component).
    ANY_PASS: any PASS->PASS; all FAIL->FAIL; otherwise not determined (no decision).
    ALL_PASS: any FAIL->FAIL; all PASS->PASS; otherwise not determined (no decision).
    FOUR_K: source-owned evidence supplied; count threshold 1..4 required.
    """
    u=_dev(issue)
    if horizon_months not in EXIT_HORIZONS: raise ValueError('EXIT_HORIZON_REGISTER_ONLY')
    if policy not in ('ANY_PASS','ALL_PASS'):raise ValueError('POLICY_NOT_PREDECLARED')
    if not isinstance(family_components,dict) or set(family_components)-set(FAMILIES):
        raise ValueError('FAMILY_COMPONENT_REGISTRY_INVALID')
    votes={}; reasons={}
    for family in FAMILIES:
        votes[family],reason=_family_proof(component=family_components.get(family),
                   issue=u,horizon=horizon_months,family=family)
        if reason:reasons[family]=reason
    result=decide_or(*votes.values()) if policy=='ANY_PASS' else decide_and(*votes.values())
    out={'branch_votes':{k:v for k,v in votes.items() if not is_nonvote(v)},'missing_reasons':reasons,
            'policy':policy,'issue':u.isoformat(),'target':add_months(u,horizon_months).isoformat(),
            'horizon_months':horizon_months,'threshold_registry_source':'ORIGINAL_EXIT_ONLY',
            'warning':'Thresholds and fitted coefficients are strictly CALLER-PROVEN; no fit is inferred.'}
    if is_nonvote(result):
        return {'status':'NOT_EVALUATED','reason':result.reason,**out}
    return {'status':'RESEARCH_DECISION_ONLY','candidate_N':result,**out}


def state_conditioned_exit_gate(*, issue, horizon_months, anchor,
                                original_gate, conditional_packet=None):
    """THESIS/ANTITHESIS/SYNTHESIS: B is a genuine extra model, not renamed A.

    Thesis A: IL+IS+purified JL+JS Exit source threshold votes at issue.
    Antithesis B: first issued closest-known source-state anchor and conditional
    state/lag/long-short evidence. No artifact => NOT_EVALUATED (NOT fake new signal, no vote).
    Synthesis: B predicts first-N only on A_PASS AND B_PASS; A_FAIL blocks N.
    A/B forecast records remain separately issued and never overwritten.
    """
    u=_dev(issue)
    if horizon_months not in EXIT_HORIZONS:raise ValueError('EXIT_HORIZON_REGISTER_ONLY')
    if anchor is None:return {'status':'NOT_EVALUATED','reason':'NO_FIRST_STATE_ANCHOR'}
    old=original_gate.get('candidate_N')            # None = original gate not determined
    if old=='FAIL':
        return {'status':'RESEARCH_DECISION_ONLY','candidate_N':'FAIL',
                'reason':'ORIGINAL_EXIT_THRESHOLD_FAILED'}
    if conditional_packet is None:
        return {'status':'NOT_EVALUATED',
                'reason':'NO_STATE_CONDITIONAL_FITTED_THRESHOLD_ARTIFACT'}
    if (not isinstance(conditional_packet,dict) or
        conditional_packet.get('track')!='EXIT' or
        conditional_packet.get('branch')!='STATE_B' or
        conditional_packet.get('horizon_months')!=horizon_months or
        conditional_packet.get('anchor_id')!=anchor['anchor_id']):
        raise ValueError('STATE_CONDITIONAL_OWNER_HORIZON_ANCHOR_MISMATCH')
    if not conditional_packet.get('available_at') or day(conditional_packet['available_at'])>u:
        raise ValueError('STATE_CONDITIONAL_FUTURE_OR_MISSING')
    req=('model_id','threshold_owner_id','threshold_id','source_hash','evidence_id')
    if any(not conditional_packet.get(k) for k in req):
        return {'status':'NOT_EVALUATED','reason':'STATE_CONDITIONAL_PROOF_INCOMPLETE'}
    if (conditional_packet.get('fit_window')!='DEV_2005_2019' or
        conditional_packet.get('fitted') is not True or
        conditional_packet.get('native_verified') is not True):
        return {'status':'NOT_EVALUATED','reason':'STATE_CONDITIONAL_FIT_NOT_VERIFIED'}
    new=conditional_packet.get('vote')
    if new=='U':raise ThirdVoteError('LEGACY_U_STATE_CONDITIONAL_VOTE_REJECTED')
    if new is None and conditional_packet.get('execution_status') in NON_VOTE_STATUSES:
        new=not_evaluated(conditional_packet.get('reason') or 'STATE_CONDITIONAL_NOT_EVALUATED')
    elif new not in ('PASS','FAIL'):
        raise ValueError('STATE_CONDITIONAL_VOTE_DOMAIN')
    v=decide_and(old if old is not None else not_evaluated('ORIGINAL_EXIT_GATE_NOT_DETERMINED'), new)
    common={'reason':'OLD_EXIT_AND_NEW_STATE_CONDITIONAL_THRESHOLD','original_gate':old,
            'state_conditional_gate':None if is_nonvote(new) else new,
            'conditional_model_id':conditional_packet['model_id']}
    if is_nonvote(v):
        return {'status':'NOT_EVALUATED',**common,'reason':v.reason}
    return {'status':'RESEARCH_DECISION_ONLY','candidate_N':v,**common}


def state_lag_longshort_features(*, issue, state_controller):
    """Issue-valid candidate covariates for B, not a fitted transition model.

    Include 7x7-source issued state progression and measured age relative to
    first anchor; no inference about future state, episode Exit, or gold.
    Exact Long/Short worker source RAW features remain separate inputs.
    """
    u=_dev(issue)
    if state_controller.get('status')!='OK':
        return {'status':'NOT_EVALUATED','reason':'NO_ISSUED_STATE_EPISODE'}
    if state_controller.get('issue')!=u.isoformat():
        raise ValueError('STATE_CONTEXT_ISSUE_MISMATCH')
    anchor=state_controller.get('selected_anchor')
    if anchor is None:return {'status':'NOT_EVALUATED','reason':'NO_ISSUED_STATE_ANCHOR'}
    per_source={}
    epoch_start=state_controller.get('last_reescalation_at')
    epoch_origin=state_controller.get('last_reescalation_occurred')
    for e in state_controller['events']:
        if epoch_start and e['available_at']<=epoch_start:continue
        if epoch_origin and e['occurred_at']<epoch_origin:continue
        if e['state'] in STATE_RANK:
            old=per_source.get(e['source'])
            if old is None or STATE_RANK[e['state']]>old['rank']:
                per_source[e['source']]={'state':e['state'],
                    'rank':STATE_RANK[e['state']],
                    'first_issued_at':e['available_at']}
    return {'status':'RESEARCH_FEATURES_ONLY_NOT_FITTED',
        'anchor_id':anchor['anchor_id'],'anchor_rank':anchor['rank'],
        'months_since_first_anchor_issue':month_distance(anchor['first_issued_at'],u),
        'issued_source_states':per_source,
        'missing_sources':tuple(x for x in EWI if x not in per_source),
        'n_sources_issued':len(per_source),
        'source_epoch_age_months':{j:month_distance(v['first_issued_at'],u)
                                   for j,v in per_source.items()},
        'riskset_clock':'ISSUE_KNOWN_ONLY',
        'IL_IS_JL_JS_inputs':'REQUIRE_WORKER_PACKET_NOT_FAKED'}


def _issued_record(*, issue, country, episode_id, horizon, branch, anchor_id,
                   threshold_decision, rq01_regime, new_state_at_issue):
    u=_dev(issue)
    if branch not in ('HORIZON_A','STATE_B') or horizon not in EXIT_HORIZONS:
        raise ValueError('EXIT_BRANCH_OR_HORIZON_INVALID')
    if threshold_decision['issue']!=u.isoformat() or threshold_decision['horizon_months']!=horizon:
        raise ValueError('THRESHOLD_DECISION_ISSUE_MISMATCH')
    candidate=threshold_decision.get('candidate_N')   # None = threshold decision not determined
    # Fresh confirmed state suppresses live issuance of new N forecast at THAT
    # issue; no retrospectively mutating prior issued N predictions. The future
    # forecasting engine continues and may issue an N forecast at a later issue.
    issue_status='ISSUED'
    if candidate is None:
        # no binary threshold decision: NO forecast is issued for this key (no FORECAST_U, no third class)
        decision=None
        issue_status='NOT_ISSUED'
        reason='THRESHOLD_DECISION_NOT_DETERMINED:'+str(threshold_decision.get('reason'))
    elif new_state_at_issue and candidate=='PASS':
        decision='STRESS'
        reason='ISSUED_STATE_CONFIRMATION_PROTECTS_STRESS_CURRENT_ISSUE'
    elif candidate in ('PASS','FAIL'):
        decision={'PASS':'N','FAIL':'STRESS'}[candidate]
        reason='ORIGINAL_EXIT_THRESHOLD_GATE'
    else:
        raise ThirdVoteError('THRESHOLD_DECISION_NOT_BINARY:'+repr(candidate))
    key=(country,episode_id,u.isoformat(),branch,horizon,anchor_id)
    return {'forecast_id':_hash(key)[:24],
            'country':country,'episode_id':episode_id,'issued_at':u.isoformat(),
            'target_at':add_months(u,horizon).isoformat(),'horizon_months':horizon,
            'branch':branch,'anchor_id':anchor_id,'track':'EXIT',
            'prediction':decision,'issue_status':issue_status,'threshold_decision':candidate,
            **regime_fields(rq01_regime,'current_rq01_regime'),'current_state_owner':'RQ01',
            'earliest_maturity_months':3,'reason':reason,
            'source_artifact_hash':_hash(threshold_decision),
            'policy':'RESEARCH_ONLY_NOT_CERTIFIED'}


def run_dual_exit_issue(*, issue, country, episode_id, onset, rq01_view,
                        state_events=(),first_n=None,follower_obligations=(),
                        exit_components_by_horizon=None,composition_policy='ANY_PASS',
                        state_conditioned_packets_by_horizon=None):
    """A and B are run IN PARALLEL for all EXIT_HORIZONS each issued month.

    A always runs after onset (even if there is no Peak); B runs once any RQ01
    progression Peak+ is confirmed. Every B issuance uses current highest-rank
    FIRST ANCHOR; older B issuance IDs remain preserved by append-only ledger.
    Neither can issue N unless OLD EXIT threshold family gate actually PASSes.
    """
    status=parallel_exit_controller(issue=issue,country=country,episode_id=episode_id,
             onset=onset,rq01_view=rq01_view,state_events=state_events,
             first_n=first_n,follower_obligations=follower_obligations)
    if not status.get('exit_forecasts_continue'):
        return {'status':status['status'],'controller':status,'issued':tuple()}
    u=_dev(issue)
    family_packets=exit_components_by_horizon or {}
    if not isinstance(family_packets,dict) or set(family_packets)-set(EXIT_HORIZONS):
        raise ValueError('EXIT_HORIZON_PACKET_KEYS_INVALID')
    issued=[]
    state_packets=state_conditioned_packets_by_horizon or {}
    if not isinstance(state_packets,dict) or set(state_packets)-set(EXIT_HORIZONS):
        raise ValueError('STATE_CONDITIONAL_HORIZON_REGISTER_INVALID')
    for h in EXIT_HORIZONS:
        threshold=exit_four_family_gate(issue=u,horizon_months=h,
               family_components=family_packets.get(h,{}),policy=composition_policy)
        issued.append(_issued_record(issue=u,country=country,episode_id=episode_id,
           horizon=h,branch='HORIZON_A',anchor_id=None,
           threshold_decision=threshold,rq01_regime=status['rq01_regime'],
           new_state_at_issue=status['new_state_at_issue']))
        anchor=status['selected_anchor']
        if anchor is not None:
            conditioned=state_conditioned_exit_gate(issue=u,horizon_months=h,
                   anchor=anchor,original_gate=threshold,
                   conditional_packet=state_packets.get(h))
            conditioned_record={k:v for k,v in threshold.items() if k!='candidate_N'}
            if 'candidate_N' in conditioned:
                conditioned_record['candidate_N']=conditioned['candidate_N']
            else:
                conditioned_record['reason']=conditioned.get('reason')
            conditioned_record['conditional_gate_status']=conditioned
            issued.append(_issued_record(issue=u,country=country,episode_id=episode_id,
              horizon=h,branch='STATE_B',anchor_id=anchor['anchor_id'],
              threshold_decision=conditioned_record,rq01_regime=status['rq01_regime'],
              new_state_at_issue=status['new_state_at_issue']))
    return {'status':'RESEARCH_ISSUED_NOT_CERTIFIED','controller':status,
            'issued':tuple(issued),'n_issued':len(issued)}


def append_only_forecast_ledger(*, previous, new_records):
    """No ex-post changes, even on delayed 20-month follower / first-N."""
    old=[dict(x) for x in previous];seen={}
    for r in old:
        if r['forecast_id'] in seen:raise ValueError('PREVIOUS_LEDGER_DUPLICATE_ID')
        seen[r['forecast_id']]=_hash(r)
    for r in new_records:
        rid=r['forecast_id']
        if rid in seen:
            if seen[rid]!=_hash(r):raise ValueError('ISSUED_FORECAST_REWRITE_ATTEMPT')
            continue
        if old and day(r['issued_at'])<day(old[-1]['issued_at']):
            raise ValueError('LEDGER_BACKDATING')
        old.append(dict(r));seen[rid]=_hash(r)
    return tuple(old)


def current_rq01_bridge(*, issue, rq01_view, first_exit_forecast_issued_at=None):
    """For t..t+3 (and each later issue): current state ALWAYS comes from RQ01.

    Does not fill future RQ01 states. No fabricated truth about months not yet
    available. The exit prediction is a separate frozen row evaluated at maturity.
    """
    u=_dev(issue)
    regime=_check_predicted_rq01_view(rq01_view,u)
    if first_exit_forecast_issued_at and day(first_exit_forecast_issued_at)>u:
        raise ValueError('FUTURE_EXIT_ISSUANCE')
    return {'issue':u.isoformat(),**regime_fields(regime,'current_regime'),'current_owner':'RQ01',
            'forecast_not_current_state':True,
            'future_issue_states':'NOT_OBSERVED','min_exit_horizon_months':3}


def retrospective_exit_horizon_samples(*, country,episode_id,first_n_at,
                                        eligible_riskset_rows,asof):
    """T-h sampling is RETROSPECTIVE LABEL JOIN ONLY, never production features.

    Each horizon needs its own eligible negatives, right-censoring and first-N
    validity. All DEV remains a single block; no chronology folds. A positive
    Exit anchor alone is a case-only sample and MUST NOT activate fitting.
    """
    t=_dev(first_n_at);a=_dev(asof)
    if t>a:return {'status':'NOT_MATURED','anchors':()}
    if not eligible_riskset_rows:return {'status':'BLOCKED_RISKSET_MISSING','anchors':()}
    rows=[]
    for r in eligible_riskset_rows:
        if r.get('country')!=country or r.get('episode_id')!=episode_id:continue
        if r.get('track')!='EXIT' or r.get('rq01_onset_issued_verified') is not True:
            raise ValueError('RETRO_EXIT_RISKSET_OWNER_REQUIRED')
        if r.get('label_independent') is not True or r.get('feature_asof_verified') is not True:
            raise ValueError('RETRO_FEATURE_OR_LABEL_NOT_VERIFIED')
        u=_dev(r['issued_at'])
        g=r.get('first_n_at')
        if g is not None and _dev(g)<=u:raise ValueError('EXIT_EVENT_NOT_IN_FUTURE_RISKSET')
        rows.append(r)
    anchors=[];n_neg=0;n_pos=0
    for h in EXIT_HORIZONS:
        u=add_months(t,-h);eligible=[]
        for r in rows:
            if day(r['issued_at'])!=u:continue
            if day(r['observed_through'])<add_months(u,h):continue
            g=r.get('first_n_at')
            y=g is not None and day(g)<=add_months(u,h)
            tm=add_months(u,h)
            y_exact=g is not None and (day(g).year,day(g).month)==(tm.year,tm.month)
            eligible.append({'source_key':r.get('source_key'), 'exit_by_h':y, 'exit_exact_month':y_exact})
            n_pos+=int(y);n_neg+=int(not y)
        anchors.append({'horizon':h,'issue':u.isoformat(),
            'status':'ELIGIBLE_RETROSPECTIVE_KEY' if eligible else 'NOT_EVALUATED_NO_ELIGIBLE_ISSUE',
            'eligible':tuple(eligible)})
    return {'status':'RETROSPECTIVE_RESEARCH_ONLY' if n_neg and n_pos else 'BLOCKED_NEEDS_MATCHED_POSITIVE_AND_NEGATIVE',
            'anchors':tuple(anchors),'n_matched_positive':n_pos,
            'n_matched_negative':n_neg,'n_rows_input':len(rows),
            'warning':'True Exit T only indexes labels, never production features; N_all controls required.'}


def score_first_n_exit_riskset(*, issued_forecast, first_n_at, observed_through,
                               label_matured_at, asof, target_definition='FIRST_N_EXACT_MONTH'):
    """First-N scoring.  PRIMARY = the first N falls in EXACTLY the target month; the by-horizon definition
    (first N within (issue, issue+h]) is the control and is never mixed into the same loss table.

    Exact first-N time MUST be future of issue; if known earlier, row is invalid.
    A censored horizon is NOT_MATURED (not negative; exact-month needs the whole target month observed).
    No post-issue events are predictors.
    """
    if target_definition not in ('FIRST_N_EXACT_MONTH','FIRST_N_BY_HORIZON'):
        raise ValueError('EXIT_TARGET_MODE_UNAUTHORIZED')
    if issued_forecast.get('track')!='EXIT':raise ValueError('EXIT_GOLD_SCORE_TRACK_ONLY')
    u=_dev(issued_forecast['issued_at']);t=_dev(issued_forecast['target_at'])
    if t!=add_months(u,issued_forecast['horizon_months']) or issued_forecast['horizon_months'] not in EXIT_HORIZONS:
        raise ValueError('FORECAST_HORIZON_TARGET_MISMATCH')
    exact=target_definition=='FIRST_N_EXACT_MONTH'
    need=date(t.year,t.month,monthrange(t.year,t.month)[1]) if exact else t
    if day(asof)<need or day(label_matured_at)>day(asof) or day(observed_through)<need:
        return {'status':'NOT_MATURED'}
    actual=False
    if first_n_at is not None:
        event=day(first_n_at)
        if event<=u:return {'status':'INVALID_PRE_ISSUE_FIRST_N'}
        actual=((event.year,event.month)==(t.year,t.month)) if exact else (event<=t)
    predicted=issued_forecast.get('prediction')
    if predicted not in ('N','STRESS'):return {'status':'NOT_EVALUATED_NO_ISSUED_PREDICTION'}
    return {'status':'EVALUATED_RESEARCH_ONLY','actual_first_n_by_h':actual,
        'predicted_first_n_by_h':predicted=='N',
        'correct':(predicted=='N')==actual,
        'target':t.isoformat(),'issue':u.isoformat(),'target_definition':target_definition,
        'branch':issued_forecast['branch'],'horizon_months':issued_forecast['horizon_months']}
