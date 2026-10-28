"""Function-first transfer of DEV-only five-asset trough-truth and issued-zone scorer.

Adapted from v8_w04_rq02/bottom_evaluation_rev357.py; intentionally standalone.
This function is post-event evaluation, NEVER a predictive feature or approval
of source authenticity. The original dataset is NOT altered or imported here.
"""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
from datetime import date, timedelta
from hashlib import sha256
import json
import math

RISKY_ASSETS = ('E1','E2','E3','B1','B2')
DEV_END = date(2019,12,31)
VERSION = 'V8_TRANSFER_BOTTOM_EVALUATION_FUNCTION_FIRST_V1'


def _hash(v):
    return sha256(json.dumps(v,sort_keys=True,default=str,separators=(',',':')).encode()).hexdigest()


def _seal(v):
    return {**v,'packet_sha':_hash(v)}


def _day(v):
    if isinstance(v,date): return v
    return date.fromisoformat(str(v)[:10])


def _key(v):
    return tuple(v[k] for k in ('country','asset','issue_u','target_id'))


def _finite(v):
    return type(v) in (int,float) and math.isfinite(v)


def build_trough_truth_panel(canonical_asset_levels, *, eval_windows, native_calendar, source_lock):
    """Register five weekly level-based targets on DEV; never use minimum return.

    Native levels and locked weekly calendars must be supplied, not synthesized.
    Each same issue/window group must cover all five assets. Immature windows
    receive IMMATURE, not a normal/negative/imputed trough label.
    """
    expected_assets = set(RISKY_ASSETS)
    if any(set(x)!=expected_assets for x in (canonical_asset_levels,native_calendar,source_lock)):
        raise ValueError('EXACT_FIVE_RISKY_ASSETS_REQUIRED')
    prices, calendars = {}, {}
    for asset in RISKY_ASSETS:
        rows, card = canonical_asset_levels[asset], source_lock[asset]
        if card.get('levels_sha') != _hash(rows) or not card.get('source_sha'):
            raise ValueError('SOURCE_LOCK_HASH_REQUIRED:'+asset)
        if card.get('native_frequency') != 'WEEKLY' or card.get('level_kind') not in (
                'NTR_INDEX_LEVEL','CANONICAL_SYNTHETIC_RESEARCH_INDEX_LEVEL'):
            raise ValueError('NATIVE_INDEX_LEVELS_REQUIRED:'+asset)
        cells = {}
        for row in rows:
            d=_day(row['economic_date']); v=row['level']
            if d > DEV_END or d in cells or not _finite(v) or v<=0:
                raise ValueError('DUPLICATE_OR_INVALID_DEV_LEVEL:'+asset)
            cells[d]=float(v)
        cal=sorted(_day(d) for d in native_calendar[asset])
        if not cal or len(set(cal))!=len(cal) or any(d > DEV_END for d in cal):
            raise ValueError('NATIVE_CALENDAR_DUPLICATE_OR_OOS')
        if any((b-a).days!=7 for a,b in zip(cal,cal[1:])) or not set(cells).issubset(cal):
            raise ValueError('WEEKLY_CALENDAR_OR_OBSERVATION_MISMATCH')
        prices[asset],calendars[asset]=cells,cal
    seen, groups, out = set(), {}, []
    for w in eval_windows:
        w=deepcopy(w)
        if w.get('asset') not in expected_assets:
            raise ValueError('UNKNOWN_BOTTOM_ASSET')
        key=_key(w)
        if key in seen:
            raise ValueError('DUPLICATE_BOTTOM_TARGET')
        seen.add(key)
        issue,start,end,register,cutoff=(_day(w[k]) for k in (
            'issue_date','start','end','registered_at','observed_through'))
        if issue < date(2005,1,1) or issue > DEV_END or issue.strftime('%Y-%m')!=w['issue_u']:
            raise ValueError('INVALID_DEV_ISSUE')
        if not issue<start<=end or register>issue or not w.get('window_policy_sha'):
            raise ValueError('STRICT_PREREGISTERED_FUTURE_WINDOW_REQUIRED')
        if cutoff>DEV_END:
            raise ValueError('OOS_SEALED_CUTOFF')
        cal=calendars[w['asset']]
        expected=[d for d in cal if start<=d<=min(end,cutoff)]
        missing=[d for d in expected if d not in prices[w['asset']]]
        reason=('WINDOW_NOT_MATURE_IN_DEV' if end>cutoff or end>DEV_END else
                'NATIVE_CALENDAR_DOES_NOT_COVER_WINDOW' if cal[0]>start or cal[-1]+timedelta(days=7)<=end else
                'NO_FUTURE_NATIVE_OBSERVATION' if not expected else
                'MISSING_EXPECTED_PRICE_OBSERVATIONS' if missing else None)
        trough=None if reason else min(expected,key=lambda d:(prices[w['asset']][d],d))
        out.append({**w,'truth_status':'IMMATURE' if reason else 'MATURE','truth_reason':reason,
                    'trough_week':trough.isoformat() if trough else None,
                    'trough_level':prices[w['asset']][trough] if trough else None,
                    'n_expected_observations':len(expected),'n_missing_price':len(missing),
                    'date_resolution':'OBSERVED_ECONOMIC_WEEK_NOT_INTRAWEEK_DAY',
                    'label_role':'EX_POST_TRUTH_ONLY_NEVER_FEATURE'})
        groups.setdefault((w['country'],w['issue_u'],w['start'],w['end']),set()).add(w['asset'])
    if not out or any(x != expected_assets for x in groups.values()):
        raise ValueError('FIVE_ASSETS_PER_WINDOW_REQUIRED')
    return _seal({'calc_version':VERSION,'rows':out,'development_block':'2005-2019_ONE_BLOCK',
                  'source_lock_sha':_hash(source_lock),'OOS':'SEALED'})


def _interval_score(lo, hi, truth, alpha):
    # Interval score, in weeks. Wider intervals incur a penalty.
    return (hi-lo)+(2/alpha)*max(0,lo-truth)+(2/alpha)*max(0,truth-hi)


def _summary(rows):
    ct=Counter(x['outcome'] for x in rows)
    scored=[x for x in rows if x['outcome'] in ('HIT','MISS')]
    issued=[x for x in rows if x['issued_zone']]
    # NO_ZONE rows are the MATURE opportunities without an issued zone (immature ones are IMMATURE); the issued
    # zone tape carries no non-evaluated decisions, so every NO_ZONE here is a real no-zone outcome (GLOBAL NO-U).
    eligible_matured=len(scored)+ct['NO_ZONE']
    # Spec RQ02-Bottom 16: HitRate = HIT/(HIT+MISS) is NOT Forecast Coverage.
    # ForecastCoverage = issued&matured / eligible&matured; a zone-everywhere policy must not look good.
    return {'n_opportunities':len(rows),'n_eligible_matured':eligible_matured,
            'n_issued_matured':len(scored),'n_hit':ct['HIT'],
            'n_miss':ct['MISS'],'n_no_zone':ct['NO_ZONE'],'n_immature':ct['IMMATURE'],
            'hit_rate':ct['HIT']/len(scored) if scored else None,
            'forecast_coverage':len(scored)/eligible_matured if eligible_matured else None,
            'no_zone_rate':ct['NO_ZONE']/eligible_matured if eligible_matured else None,
            'mean_interval_score_weeks':sum(x['interval_score_weeks'] for x in scored)/len(scored) if scored else None,
            'mean_width_calendar_days':sum(x['width_calendar_days'] for x in issued)/len(issued) if issued else None,
            'mean_lead_time_calendar_days':(sum(x['lead_time_calendar_days'] for x in scored)/len(scored)
                                            if scored else None),
            'n_country_issue_keys':len({(x['country'],x['issue_u']) for x in scored}),  # NOT independent episodes
            'uncertainty':'NOT_IDENTIFIABLE_WITHOUT_EPISODE_SUPPORT'}


def score_five_asset_bottom_zones(issued_zone_tape, trough_truth_panel, *, maturity_cutoff, alpha):
    """Matched HIT/MISS/NO_ZONE/IMMATURE truth scorer; no model fitting."""
    cutoff=_day(maturity_cutoff)
    if cutoff>DEV_END or not _finite(alpha) or not 0<alpha<1:
        raise ValueError('DEV_CUTOFF_AND_ALPHA_REQUIRED')
    body={k:v for k,v in trough_truth_panel.items() if k!='packet_sha'}
    if _hash(body)!=trough_truth_panel.get('packet_sha'):
        raise ValueError('TRUTH_PANEL_HASH_MISMATCH')
    zones={}
    for z in issued_zone_tape.records() if hasattr(issued_zone_tape,'records') else issued_zone_tape:
        if z.get('head')!='BOTTOM_3F':
            continue
        z=dict(z)
        if z.get('record_sha')!=_hash({k:v for k,v in z.items() if k!='record_sha'}):
            raise ValueError('ISSUED_ZONE_SHA_MISMATCH')
        key=_key(z)
        if key in zones:
            raise ValueError('DUPLICATE_ISSUED_ZONE')
        zones[key]=z
    keys={_key(x) for x in trough_truth_panel['rows']}
    if not set(zones).issubset(keys):
        raise ValueError('ZONE_WITHOUT_TRUTH_TARGET')
    result=[]
    for truth in trough_truth_panel['rows']:
        zone=zones.get(_key(truth))
        rec={k:truth[k] for k in ('country','asset','issue_u','target_id','start','end','n_missing_price')}
        rec.update(outcome='NO_ZONE',issued_zone=bool(zone),zone_start=None,zone_end=None,
                   width_calendar_days=None,interval_score_weeks=None,lead_time_calendar_days=None)
        if truth['truth_status']!='MATURE' or _day(truth['end'])>cutoff:
            rec['outcome']='IMMATURE'
        if zone:
            issue=_day(zone['issue_timestamp'])
            if issue!=_day(truth['issue_date']):
                raise ValueError('ISSUE_AND_TRUTH_MISMATCH')
            lo,hi=zone['zone_start_weeks'],zone['zone_end_weeks']
            if not _finite(lo) or not _finite(hi) or not 0<lo<=hi or (lo*7)%1 or (hi*7)%1:
                raise ValueError('STRICT_FUTURE_WEEK_ZONE_REQUIRED')
            left,right=issue+timedelta(days=7*lo),issue+timedelta(days=7*hi)
            rec.update(zone_start=left.isoformat(),zone_end=right.isoformat(),
                       width_calendar_days=(right-left).days)
            if truth['truth_status']=='MATURE' and _day(truth['end'])<=cutoff:
                actual=_day(truth['trough_week'])
                rec.update(outcome='HIT' if left<=actual<=right else 'MISS',
                           lead_time_calendar_days=(actual-issue).days,
                           interval_score_weeks=_interval_score(lo,hi,(actual-issue).days/7,alpha))
        result.append(rec)
    return _seal({'calc_version':VERSION,'rows':result,'summary':_summary(result),
                  'per_asset':{a:_summary([x for x in result if x['asset']==a]) for a in RISKY_ASSETS},
                  'truth_panel_sha':trough_truth_panel['packet_sha'], 'OOS':'SEALED',
                  'claim':'DEV_POST_EVENT_EVALUATION_ONLY_NOT_TRADING_AUTHORIZATION'})
