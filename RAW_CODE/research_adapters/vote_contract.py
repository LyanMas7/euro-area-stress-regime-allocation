"""MANDATORY ACTIVE VOTE CONTRACT — NO LEGACY U CLASS (work order V8-REV373-GLOBAL-NO-U-RQ01-RQ02-20261025-R01).

Every current, legitimately evaluable RQ01/RQ02 Individual, Joint, Pool, ENTRY, EXIT-A, EXIT-B and
Bottom model vote is exclusively binary (PASS/FAIL or 1/0).  No active model decision may emit U,
UNKNOWN, ABSTAIN, -1, NaN or None as a third forecast class.  Ineligible, unissued, blocked,
not-applicable, censored and not-activated records belong to separate data/evidence/authorization/
evaluation ledgers and are never silently coerced into FAIL or the N regime.

Two layers:
  * a model vote          -> 'PASS'/'FAIL' (or int 1/0); only an EXECUTED computation carries one;
  * a non-vote record     -> `NonVote(status, reason)` with status in NOT_EVALUATED / NOT_APPLICABLE /
                             BLOCKED.  It is a typed object, never a string vote token.

Boolean inference with incomplete inputs (work order section 2.4) never invents a vote: OR is PASS
on one proven PASS, AND is FAIL on one proven FAIL, and K-of-N with L proven PASS and m registered
non-evaluable components is PASS iff L >= K, FAIL iff L + m < K; otherwise the decision is NOT
issued and the caller records `DECISION_NOT_DETERMINED_FROM_AVAILABLE_INPUTS` (a pre-vote
identifiability status, not a renamed third vote).

Preserved meanings of the letter U that are NOT a vote: the motif/direction letter 'U' = UP
(`Direction`, `direction_from_legacy`), the numeric Upper threshold bound in L/U/STEP, File 02
section labels U.1-U.7, and immutable frozen historical owner records (Owner C U/UNAVAILABLE), which
are read only through `normalize_frozen_owner_c` and never rewritten.
"""
from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable

CONTRACT_ID = 'V8-REV373-GLOBAL-NO-U-RQ01-RQ02-20261025-R01'
DEPENDENCY_TOKEN = 'NO_LEGACY_U_ACTIVE_VOTE'
CONTRACT_VERSION = 'GLOBAL_NO_U_ACTIVE_VOTE_CONTRACT_V1'

MANDATORY_TEXT = (
    'MANDATORY ACTIVE VOTE CONTRACT — NO LEGACY U CLASS. Every current, legitimately evaluable '
    'RQ01/RQ02 Individual, Joint, Pool, ENTRY, EXIT-A, EXIT-B and Bottom model vote is exclusively '
    'binary (PASS/FAIL or 1/0). No active model decision may emit U, UNKNOWN, ABSTAIN, -1, NaN or None '
    'as a third forecast class. Ineligible, unissued, blocked, not-applicable, censored and '
    'not-activated records belong to separate data/evidence/authorization/evaluation ledgers and shall '
    'not be silently coerced into FAIL or the N regime. Preserve actual upward movements and numerical '
    'Upper threshold bounds. Preserve immutable historical frozen artifacts behind explicit versioned '
    'adapters and authorized owner approval. This design decision is standing: do not ask to '
    'reintroduce U.')

PASS, FAIL = 'PASS', 'FAIL'
VOTE_VALUES = (PASS, FAIL)
BINARY_INTS = (0, 1)


class ExecutionStatus(str, Enum):
    EXECUTED = 'EXECUTED'              # only this state carries a model vote
    NOT_EVALUATED = 'NOT_EVALUATED'    # no model vote was generated (data/readiness/identifiability)
    NOT_APPLICABLE = 'NOT_APPLICABLE'  # structurally inapplicable (e.g. H15 cross-horizon confirmation)
    BLOCKED = 'BLOCKED'                # owner/configuration/authorization prerequisite violated


NON_VOTE_STATUSES = (ExecutionStatus.NOT_EVALUATED.value, ExecutionStatus.NOT_APPLICABLE.value,
                     ExecutionStatus.BLOCKED.value)
EXECUTION_STATUSES = (ExecutionStatus.EXECUTED.value,) + NON_VOTE_STATUSES

# Label (gold) scoring status: a censored / immature future label is never a negative label.
LABEL_MATURE = 'MATURE'
LABEL_CENSORED = 'NOT_EVALUATED_CENSORED_GOLD'

DECISION_NOT_DETERMINED = 'DECISION_NOT_DETERMINED_FROM_AVAILABLE_INPUTS'

# Every encoding of a legacy third forecast class that must never appear as an active vote.
FORBIDDEN_VOTE_TOKENS = frozenset({'U', 'UNKNOWN', 'UNDECIDED', 'UNAVAILABLE', 'ABSTAIN', 'ABSTENTION',
                                   'FORECAST_U', 'TRISTATE_U', 'U_NOT_EVALUATED', 'NONE', 'NAN', ''})

# Report counters (work order section 9.6).  Only the first two are binary forecast outcomes.
REPORT_COUNTERS = ('N_ISSUED_PASS', 'N_ISSUED_FAIL', 'N_NOT_EVALUATED', 'N_NOT_APPLICABLE', 'N_BLOCKED',
                   'N_CENSORED_GOLD', 'N_UNAUTHORIZED_PUBLICATION')


class ThirdVoteError(ValueError):
    """A legacy third forecast class (U/UNKNOWN/ABSTAIN/-1/None/NaN) reached an active vote slot."""


@dataclass(frozen=True)
class NonVote:
    """A non-executed decision: carries a status and a reason, never a vote value."""
    status: str
    reason: str
    detail: tuple = ()

    def __post_init__(self):
        if self.status not in NON_VOTE_STATUSES:
            raise ValueError('NON_VOTE_STATUS_REQUIRED:' + str(self.status))
        if not self.reason or not isinstance(self.reason, str):
            raise ValueError('NON_VOTE_REASON_REQUIRED')

    def __bool__(self):
        raise TypeError('NonVote has no truth value: it is not a vote (' + self.reason + ')')

    def as_record(self) -> dict:
        return {'execution_status': self.status, 'reason': self.reason,
                'detail': [list(d) if isinstance(d, tuple) else d for d in self.detail]}


NOT_DETERMINED = NonVote(ExecutionStatus.NOT_EVALUATED.value, DECISION_NOT_DETERMINED)


def not_evaluated(reason: str, *detail) -> NonVote:
    return NonVote(ExecutionStatus.NOT_EVALUATED.value, reason, tuple(detail))


def not_applicable(reason: str, *detail) -> NonVote:
    return NonVote(ExecutionStatus.NOT_APPLICABLE.value, reason, tuple(detail))


def blocked(reason: str, *detail) -> NonVote:
    return NonVote(ExecutionStatus.BLOCKED.value, reason, tuple(detail))


def is_nonvote(x: Any) -> bool:
    return isinstance(x, NonVote)


def _is_nan(x: Any) -> bool:
    try:
        return isinstance(x, float) and math.isnan(x)
    except TypeError:
        return False


def is_vote(x: Any) -> bool:
    """True only for 'PASS'/'FAIL', Python bool, or an integral 0/1 (numpy ints included)."""
    if isinstance(x, NonVote) or x is None or _is_nan(x):
        return False
    if isinstance(x, str):
        return x in VOTE_VALUES
    if isinstance(x, bool):
        return True
    try:
        return float(x) in (0.0, 1.0) and float(x) == int(x)
    except (TypeError, ValueError):
        return False


def vote_token(x: Any) -> str:
    """Normalise a genuine vote to 'PASS'/'FAIL'; reject every third-class encoding."""
    if isinstance(x, str):
        if x in VOTE_VALUES:
            return x
        raise ThirdVoteError('THIRD_VOTE_TOKEN_REJECTED:' + repr(x))
    if is_vote(x):
        return PASS if int(x) == 1 else FAIL
    raise ThirdVoteError('THIRD_VOTE_VALUE_REJECTED:' + repr(x))


def vote_int(x: Any) -> int:
    return 1 if vote_token(x) == PASS else 0


def component(x: Any, reason_if_missing: str = 'COMPONENT_NOT_EVALUATED'):
    """Map an evaluated component to 'PASS'/'FAIL' and a non-evaluable one to a NonVote.

    `None` / NaN mean "this component was not evaluated" at an INPUT boundary (absence of evidence),
    never a vote; legacy third tokens ('U', 'UNKNOWN', ...) are refused loudly.
    """
    if isinstance(x, NonVote):
        return x
    if x is None or _is_nan(x):
        return not_evaluated(reason_if_missing)
    return vote_token(x)


def _norm(xs) -> list:
    out = []
    for x in xs:
        out.append(x if isinstance(x, NonVote) else vote_token(x))
    return out


def decide_not(a):
    a = _norm([a])[0]
    if isinstance(a, NonVote):
        return a
    return FAIL if a == PASS else PASS


def decide_and(*xs):
    """Binary AND with partial-input determination: one proven FAIL decides FAIL."""
    xs = _norm(xs)
    if FAIL in [x for x in xs if not isinstance(x, NonVote)]:
        return FAIL
    pending = [x for x in xs if isinstance(x, NonVote)]
    return _undetermined(pending) if pending else PASS


def decide_or(*xs):
    """Binary OR with partial-input determination: one proven PASS decides PASS."""
    xs = _norm(xs)
    if PASS in [x for x in xs if not isinstance(x, NonVote)]:
        return PASS
    pending = [x for x in xs if isinstance(x, NonVote)]
    return _undetermined(pending) if pending else FAIL


def decide_at_least(votes: Iterable, k: int):
    """K-of-N: PASS iff L >= K; FAIL iff L + m < K; otherwise not determined (no vote).

    K may be any integral type (int, numpy.int64 from a registered configuration row); bool/float are refused."""
    if isinstance(k, bool):
        raise ValueError('INTEGER_K_REQUIRED')
    try:
        k = operator.index(k)
    except TypeError:
        raise ValueError('INTEGER_K_REQUIRED') from None
    if k < 0:
        raise ValueError('INTEGER_K_REQUIRED')
    xs = _norm(list(votes))
    proven = sum(x == PASS for x in xs if not isinstance(x, NonVote))
    pending = [x for x in xs if isinstance(x, NonVote)]
    if proven >= k:
        return PASS
    if proven + len(pending) < k:
        return FAIL
    return _undetermined(pending)


def at_least_bounds(votes: Iterable, k: int) -> dict:
    xs = _norm(list(votes))
    proven = sum(x == PASS for x in xs if not isinstance(x, NonVote))
    m = sum(isinstance(x, NonVote) for x in xs)
    return {'L_proven_pass': proven, 'm_not_evaluated': m, 'H_possible_pass': proven + m, 'K': k}


def _undetermined(pending: list) -> NonVote:
    # A BLOCKED component that leaves the decision open blocks the decision; otherwise it is simply
    # not determined from the available inputs.
    reasons = tuple(sorted({p.reason for p in pending}))
    if any(p.status == ExecutionStatus.BLOCKED.value for p in pending):
        return NonVote(ExecutionStatus.BLOCKED.value, 'DECISION_BLOCKED_BY_COMPONENT', reasons)
    return NonVote(ExecutionStatus.NOT_EVALUATED.value, DECISION_NOT_DETERMINED, reasons)


def decision_fields(decision, *, vote_key: str = 'vote', as_int: bool = False) -> dict:
    """Serialise a decision for a result record: a vote only when executed, never None/U."""
    if isinstance(decision, NonVote):
        return {'execution_status': decision.status, 'non_vote_reason': decision.reason}
    tok = vote_token(decision)
    return {'execution_status': ExecutionStatus.EXECUTED.value,
            vote_key: (1 if tok == PASS else 0) if as_int else tok}


def split_components(components: dict) -> tuple[dict, dict]:
    """{name: vote|NonVote} -> (binary component votes, {name: non-vote reason})."""
    votes, missing = {}, {}
    for k, v in components.items():
        if isinstance(v, NonVote):
            missing[k] = v.reason
        else:
            votes[k] = vote_token(v)
    return votes, missing


def assert_binary_votes(values: Iterable, *, where: str = 'active vote column') -> None:
    """Every executed vote is exactly PASS/FAIL or 0/1 (machine invariant 1)."""
    bad = [v for v in values if not is_vote(v)]
    if bad:
        raise ThirdVoteError(f'NON_BINARY_ACTIVE_VOTE in {where}: {bad[:5]!r}')


def assert_no_third_vote(values: Iterable, *, where: str = 'active vote column') -> None:
    """Reject U/UNKNOWN/ABSTAIN/-1/None/NaN in an active vote array (machine invariant 3)."""
    for v in values:
        if isinstance(v, NonVote) or v is None or _is_nan(v):
            raise ThirdVoteError(f'NON_VOTE_IN_ACTIVE_VOTE_ARRAY in {where}: {v!r}')
        if isinstance(v, str) and v.strip().upper() in FORBIDDEN_VOTE_TOKENS:
            raise ThirdVoteError(f'LEGACY_THIRD_VOTE in {where}: {v!r}')
        if not is_vote(v):
            raise ThirdVoteError(f'NON_BINARY_ACTIVE_VOTE in {where}: {v!r}')


def assert_disjoint_keys(nonexecuted_keys: Iterable, issued_vote_keys: Iterable) -> None:
    """A key is either an executed vote or a non-executed record, never both (machine invariant 2)."""
    both = set(nonexecuted_keys) & set(issued_vote_keys)
    if both:
        raise ThirdVoteError('KEY_BOTH_VOTED_AND_NOT_EXECUTED:' + repr(sorted(both, key=repr)[:5]))


# ---------------------------------------------------------------- typed two-layer records (section 2.3)
@dataclass(frozen=True)
class EligibilityRecord:
    key: tuple
    status: ExecutionStatus
    reason_code: str | None
    source_version: str
    owner_id: str
    evidence_ref: str | None

    def __post_init__(self):
        st = ExecutionStatus(self.status)
        if st is ExecutionStatus.EXECUTED:
            raise ValueError('ELIGIBILITY_RECORD_IS_FOR_NON_EXECUTED_KEYS_ONLY')
        if not self.reason_code:
            raise ValueError('NON_EXECUTED_RECORD_REQUIRES_REASON')


@dataclass(frozen=True)
class ModelVote:
    key: tuple
    vote: int
    formula_id: str
    threshold_id: str
    inputs_hash: str
    issue_at: str

    def __post_init__(self):
        if type(self.vote) is not int or self.vote not in BINARY_INTS:
            raise ThirdVoteError('MODEL_VOTE_MUST_BE_INT_0_OR_1:' + repr(self.vote))


@dataclass(frozen=True)
class PublicationRecord:
    key: tuple
    publication_authorized: bool
    authorization_reason: str | None
    issued_artifact_ref: str | None

    def __post_init__(self):
        if type(self.publication_authorized) is not bool:
            raise ValueError('PUBLICATION_AUTHORIZED_MUST_BE_BOOL')
        if not self.publication_authorized and not self.authorization_reason:
            raise ValueError('UNAUTHORIZED_PUBLICATION_REQUIRES_REASON')


def emit(key: tuple, decision, *, formula_id: str, threshold_id: str, inputs_hash: str, issue_at: str,
         source_version: str, owner_id: str, evidence_ref: str | None = None):
    """Caller contract: either (EXECUTED, ModelVote) or (non-executed status, EligibilityRecord)."""
    if isinstance(decision, NonVote):
        return (ExecutionStatus(decision.status),
                EligibilityRecord(key, ExecutionStatus(decision.status), decision.reason, source_version,
                                  owner_id, evidence_ref))
    return (ExecutionStatus.EXECUTED,
            ModelVote(key, vote_int(decision), formula_id, threshold_id, inputs_hash, issue_at))


# ---------------------------------------------------------------- preserved non-vote meanings of 'U'
class Direction(str, Enum):
    UP = 'UP'
    DOWN = 'DOWN'
    SIDEWAY = 'SIDEWAY'


# Audited legacy ingestion: in motif identifiers ('UU', 'UD', ...) and historical movement columns the
# letter U means UP.  It is a direction code, never an unknown vote, and is preserved numerically.
LEGACY_DIRECTION_LETTER = {'U': Direction.UP, 'D': Direction.DOWN, 'S': Direction.SIDEWAY}
DIRECTION_LETTER = {v: k for k, v in LEGACY_DIRECTION_LETTER.items()}


def direction_from_legacy(letter: str) -> Direction:
    try:
        return LEGACY_DIRECTION_LETTER[letter]
    except KeyError as exc:
        raise ValueError('LEGACY_DIRECTION_LETTER_U_D_S_REQUIRED:' + repr(letter)) from exc


def legacy_letter(direction) -> str:
    return DIRECTION_LETTER[Direction(direction)]


def upper_bound_preserved(lower: float, upper: float) -> bool:
    """L/U/STEP: U is the numeric Upper threshold bound; L <= U must hold."""
    return math.isfinite(float(lower)) and math.isfinite(float(upper)) and float(lower) <= float(upper)


# ---------------------------------------------------------------- frozen owner compatibility boundary
FROZEN_OWNER_C_LEGACY_TOKENS = frozenset({'U', 'CONFLICT', 'U/CONFLICT', 'UNAVAILABLE'})
OWNER_CONFLICT_ID = 'DATA_CONTRACT_CONFLICT_FILE01_H5_OWNER_C_U_UNAVAILABLE'


def normalize_frozen_owner_c(record_value: str) -> dict:
    """Read-only view of a frozen Owner C (File 01 H5) historical output.

    File 01 H5 / File 04 V still REQUIRE Owner C to emit `U/CONFLICT` and `UNAVAILABLE`.  Those
    immutable records are never rewritten.  The current research view maps them to non-vote
    records (no N/D/G/BOTH is invented); a real N/D/G/BOTH output passes through as a class.
    """
    v = str(record_value).strip().upper()
    if v in ('N', 'D', 'G', 'BOTH'):
        return {'research_class': v, 'execution_status': ExecutionStatus.EXECUTED.value,
                'frozen_value': record_value}
    if v in FROZEN_OWNER_C_LEGACY_TOKENS:
        status = (ExecutionStatus.NOT_EVALUATED.value if v == 'UNAVAILABLE'
                  else ExecutionStatus.BLOCKED.value)
        return {'research_class': None, 'execution_status': status,
                'reason': ('FROZEN_OWNER_C_UNAVAILABLE_NATIVE_INPUT' if v == 'UNAVAILABLE'
                           else 'FROZEN_OWNER_C_CONFLICT_OR_UNRESOLVED_SOURCE'),
                'frozen_value': record_value, 'owner_conflict': OWNER_CONFLICT_ID}
    raise ValueError('UNREGISTERED_FROZEN_OWNER_C_VALUE:' + repr(record_value))


def tally(decisions: Iterable) -> dict:
    """Separate counters: only N_ISSUED_PASS/N_ISSUED_FAIL are forecast outcomes."""
    c = {k: 0 for k in REPORT_COUNTERS}
    for d in decisions:
        if isinstance(d, NonVote):
            c['N_' + d.status] += 1
        else:
            c['N_ISSUED_PASS' if vote_token(d) == PASS else 'N_ISSUED_FAIL'] += 1
    return c
