"""Function-first threshold primitives; no fitting or threshold selection.

The mask identity depends on *issued final decisions on the entire common
population*, not on the number of candidate parameter tuples. A non-executed key
(NOT_EVALUATED / NOT_APPLICABLE / BLOCKED) is different from FAIL and is part of the
mask identity, and issue/country/horizon keys cannot silently change.
GLOBAL NO-U: a row carries a binary `decision` (PASS/FAIL) OR an `execution_status`
non-vote record; the legacy third decision 'U' is rejected.
"""
from __future__ import annotations
from decimal import Decimal
from hashlib import sha256
import json

from .vote_contract import NON_VOTE_STATUSES

DECISIONS = frozenset(('PASS', 'FAIL'))
STATES = DECISIONS | frozenset(NON_VOTE_STATUSES)     # mask-identity tokens: two votes + three non-vote statuses


def mask_token(row) -> str:
    """'PASS'/'FAIL' for an executed key, the non-vote status for a non-executed key."""
    if 'decision' in row and row['decision'] is not None:
        v = str(row['decision'])
        if v not in DECISIONS:
            raise ValueError('UNREGISTERED_DECISION:' + v)
        return v
    st = row.get('execution_status')
    if st not in NON_VOTE_STATUSES:
        raise ValueError('NON_EXECUTED_ROW_REQUIRES_EXECUTION_STATUS:' + str(st))
    return str(st)


def inclusive_lu_grid(low, high, step):
    """Return exact decimal grid with both bounds included when reachable."""
    l,h,s=map(lambda x:Decimal(str(x)),(low,high,step))
    if not all(x.is_finite() for x in (l,h,s)) or h<l or s<=0:
        raise ValueError('INVALID_L_U_STEP')
    if (h-l)/s != ((h-l)/s).to_integral_value():
        raise ValueError('STEP_MUST_REACH_U_EXACTLY')
    n=int((h-l)/s)
    if n>100000:raise ValueError('GRID_TOO_LARGE')
    return [float(l+s*i) for i in range(n+1)]


def complete_decision_mask_sha(rows):
    """One canonical final decision per (country, issue, horizon, target)."""
    keys=[]
    for row in rows:
        if not isinstance(row,dict):raise ValueError('ROW_MUST_BE_MAPPING')
        k=tuple(str(row[x]) for x in ('country','issue','horizon','target'))
        keys.append((k,mask_token(row)))
    if not keys:raise ValueError('EMPTY_DECISION_MASK')
    if len(set(k for k,v in keys))!=len(keys):raise ValueError('DUPLICATE_DECISION_KEY')
    keys.sort(key=lambda x:x[0])
    return sha256(json.dumps(keys,separators=(',',':')).encode()).hexdigest()


def unique_complete_masks(candidate_rows):
    """Candidates [(params, decision_rows)]; no evaluation or threshold fit."""
    digest_to_first={}; records=[]
    for index,(params,rows) in enumerate(candidate_rows):
        k=complete_decision_mask_sha(rows)
        first=digest_to_first.setdefault(k,index)
        records.append({'candidate_index':index,'params':params,'decision_sha256':k,
                        'unique':index==first,'same_mask_as':None if index==first else first})
    return {'nominal_candidates':len(records),'unique_complete_masks':len(digest_to_first),
            'candidate_registry':records,'status':'RESEARCH_COMPONENT_ONLY_NOT_FITTED'}


def deduplicate_complete_timeline_decision_masks(results):
    """Transfer of archived REV261 API, repaired for keyed PASS/FAIL + non-executed timelines.

    Accepts research candidate mappings {'mask': [row...], ...}. Unlike the
    archived ``sha256(bytes(mask))`` implementation, keys and unavailable
    decisions belong to mask identity. All candidates MUST share a matched
    population; use another register for a different horizon or population.
    Does not choose thresholds, compute loss, or promote candidates.
    """
    results = list(results)
    if not results:
        raise ValueError('EMPTY_CANDIDATE_REGISTER')
    base_keys = None
    first = {}
    unique = []
    registry = []
    for index, candidate in enumerate(results):
        if not isinstance(candidate, dict) or 'mask' not in candidate:
            raise ValueError('MASK_ROWS_REQUIRED')
        rows = candidate['mask']
        digest = complete_decision_mask_sha(rows)
        keys = frozenset(tuple(str(row[k]) for k in ('country','issue','horizon','target')) for row in rows)
        if base_keys is None:
            base_keys = keys
        if keys != base_keys:
            raise ValueError('UNMATCHED_POPULATION_FOR_DEDUP')
        prior = first.setdefault(digest, index)
        if prior == index:
            unique.append(candidate)
        registry.append({'candidate_index': index, 'decision_sha256': digest,
                         'same_mask_as': None if prior == index else prior})
    return {'unique': unique, 'duplicates': len(results)-len(unique),
            'unique_masks': len(unique), 'nominal_candidates': len(results),
            'candidate_registry': registry, 'status': 'RESEARCH_COMPONENT_ONLY_NOT_FITTED'}
