"""W03 fixed scope: five RQ01 sources, ESRB four classes, EA7, DEV, binary votes + eligibility.

RQ01 four-M (M00/M01/M10/M11) is a separately registered RESEARCH track. File01
still freezes historical A/B owners (conflict C01), so every output carries
`SCIENTIFIC_CONTRACT_AMENDMENT_REQUIRED` and can never be an official RQ01 PASS.
Retired REV55/REV64 A/B/C thresholds are rejected as candidate owners.

GLOBAL NO-U contract (research_adapters.vote_contract): a model vote is PASS/FAIL only; a record
that carries no vote has status NOT_EVALUATED / NOT_APPLICABLE / BLOCKED plus a reason.  The
legacy third status 'U' is not registered and is rejected.
"""
from __future__ import annotations
import hashlib
import json
from datetime import date
from typing import Any

from . import CALC_VERSION
from ..vote_contract import (NON_VOTE_STATUSES, NonVote, ThirdVoteError, decide_and, decide_at_least,
                             decide_not, decide_or, is_nonvote, vote_token)

EA7 = ('BE', 'DE', 'ES', 'FI', 'FR', 'IT', 'NL')
FORBIDDEN_ACTIVE = frozenset(('PT', 'IE'))
SOURCES = ('CISS', 'CLIFS', 'DFC', 'GFC', 'VIX')
MONTHLY = ('CISS', 'CLIFS', 'GFC', 'VIX')
QUARTERLY = ('DFC',)
COMMON_SOURCES = ('CISS', 'GFC', 'VIX')
CLASSES = ('N', 'D', 'G', 'BOTH')
DEV_START, DEV_END = date(2005, 1, 1), date(2019, 12, 31)
DEV_FIRST_MONTH, DEV_LAST_MONTH = '2005-01', '2019-12'
# Record statuses: PASS/FAIL = executed record; the rest carry no model vote.
STATUSES = ('PASS', 'FAIL') + NON_VOTE_STATUSES
RETIRED_OWNER_TOKENS = ('REV55', 'REV64', 'REV_55', 'REV_64', 'OWNER_A', 'OWNER_B', 'OWNER_C')
CONFLICTS = {
    'C01_RQ01_OWNER': 'SCIENTIFIC_CONTRACT_AMENDMENT_REQUIRED: four-M research track vs File01 frozen A/B owners',
    'C02_EA7_EA8': 'EA7 research scope vs File01 EA8 (incl. IE); EA7 is not a certified EA8 benchmark',
}
CLAIM = 'CODE_TESTED_RESEARCH_TRACK_NOT_OFFICIAL_RQ01'


class ScopeError(ValueError):
    pass


class RetiredOwnerError(ValueError):
    pass


def canonical_sha256(x: Any) -> str:
    return hashlib.sha256(json.dumps(x, sort_keys=True, default=str, separators=(',', ':'),
                                     ensure_ascii=True).encode()).hexdigest()


def result(status: str, reason: str | None = None, *, source_ids=(), issue_times=(), **payload) -> dict:
    """Record constructor.  A `vote` payload must be binary and is only allowed on an executed record."""
    if status not in STATUSES:
        raise ValueError('UNREGISTERED_STATUS:' + str(status))
    if status != 'PASS' and not reason:
        raise ValueError('NON_PASS_REQUIRES_REASON')
    if 'vote' in payload:
        if status in NON_VOTE_STATUSES:
            raise ThirdVoteError('NON_EXECUTED_RECORD_CANNOT_CARRY_A_VOTE:' + status)
        if is_nonvote(payload['vote']):
            raise ThirdVoteError('NON_VOTE_PLACED_IN_VOTE_FIELD:' + payload['vote'].reason)
        payload['vote'] = vote_token(payload['vote'])
    execution_status = 'EXECUTED' if status in ('PASS', 'FAIL') else status
    if 'execution_status' in payload:
        want = payload.pop('execution_status')
        if want != execution_status:
            raise ValueError('EXECUTION_STATUS_CONFLICTS_WITH_VOTE:' + str(want))
    return {'status': status, 'reason': reason, 'execution_status': execution_status,
            'source_ids': list(source_ids),
            'issue_times': [str(t) for t in issue_times], 'calc_version': CALC_VERSION,
            'conflicts': ['C01_RQ01_OWNER'], **payload}


def decision_result(decision, *, source_ids=(), issue_times=(), components: dict | None = None,
                    **payload) -> dict:
    """Executed record with a binary `vote`, or a NOT_EVALUATED/BLOCKED record without any vote."""
    comp_votes, comp_missing = {}, {}
    for k, v in (components or {}).items():
        if is_nonvote(v):
            comp_missing[k] = v.reason
        else:
            comp_votes[k] = vote_token(v)
    extra = {'components': comp_votes, 'components_not_evaluated': comp_missing} if components is not None else {}
    if is_nonvote(decision):
        return result(decision.status, decision.reason, source_ids=source_ids, issue_times=issue_times,
                      non_vote_detail=list(decision.detail), **extra, **payload)
    return result('PASS', source_ids=source_ids, issue_times=issue_times, vote=decision, **extra, **payload)


def require_active_country(country: str) -> str:
    if country in FORBIDDEN_ACTIVE or country not in EA7:
        raise ScopeError('EA7_ACTIVE_COUNTRY_REQUIRED:' + str(country))
    return country


def require_dev_month(month: str) -> str:
    if not (isinstance(month, str) and len(month) == 7 and month[4] == '-'):
        raise ScopeError('MONTH_YYYY_MM_REQUIRED')
    try:
        yr, mo = int(month[:4]), int(month[5:7])
    except (TypeError, ValueError) as exc:
        raise ScopeError('MONTH_YYYY_MM_REQUIRED') from exc
    if not 1 <= mo <= 12:
        raise ScopeError('MONTH_INVALID')
    if month > DEV_LAST_MONTH:
        raise ScopeError('OOS_SEALED:' + month)
    if month < DEV_FIRST_MONTH:
        raise ScopeError('PRE_DEV_TARGET_KEY:' + month)
    return month


def reject_retired_owner(owner_id: str | None) -> None:
    text = str(owner_id or '').upper()
    if any(tok in text for tok in RETIRED_OWNER_TOKENS):
        raise RetiredOwnerError('RETIRED_REV55_REV64_ABC_OWNER_NOT_A_CANDIDATE:' + str(owner_id))


def month_index(month: str) -> int:
    return int(month[:4]) * 12 + int(month[5:7]) - 1


def month_from_index(i: int) -> str:
    return f'{i // 12:04d}-{i % 12 + 1:02d}'


def months_between(first: str, last: str) -> list[str]:
    return [month_from_index(i) for i in range(month_index(first), month_index(last) + 1)]


# Binary decisions with partial-input determination (work order section 2.4).  Inputs are 'PASS'/'FAIL'
# (or 0/1) for evaluated components and `NonVote` records for non-evaluable ones; the result is a binary
# vote or a NonVote (DECISION_NOT_DETERMINED_FROM_AVAILABLE_INPUTS).  The former Kleene 'U' token is
# rejected with ThirdVoteError.  These names are retained as the single shared engine (no second
# tri-state engine exists).
def k_not(a):
    return decide_not(a)


def k_and(*xs):
    return decide_and(*xs)


def k_or(*xs):
    return decide_or(*xs)


def k_at_least(votes, k: int):
    """>= k PASS: PASS if L >= k, FAIL if L + m < k, otherwise not determined (NonVote, no vote)."""
    return decide_at_least(votes, k)
