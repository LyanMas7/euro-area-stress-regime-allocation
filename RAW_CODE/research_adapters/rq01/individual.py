"""RQ01-14 / B02(individual): source-owned high/low Individual vote.

  Strong_j = 1[#{terminal h_b >= high_j} >= k_high]
  Weak_j   = 1[#{terminal h_b >= low_j}  >= k_low]
  Count_j  = 1[qualified native count >= k_j]
  I_j      = Strong_j OR (Weak_j AND Count_j),   low_j <= high_j
Binary vote only (GLOBAL NO-U): a non-computable terminal base or a missing native count is a
non-evaluable component (NonVote), never a FAIL.  Strong/Weak use the K-of-N bound rule (PASS iff
L >= k, FAIL iff L + m < k); when the decision cannot be proven from the evaluable components the
record is NOT_EVALUATED (DECISION_NOT_DETERMINED_FROM_AVAILABLE_INPUTS) and carries no vote.
This policy (IND_STRONG_OR_WEAK_AND_COUNT_V1) is distinct from the strict Joint gate
(STRICT_JOINT_GATE1_COUNT_MOTIF_LAG_TERM2_V1); no silent hybrid of the two.
Orientation: sign = +1 tests rises, -1 tests falls (the owner's documented direction).
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from itertools import product

from ..vote_contract import NOT_DETERMINED, not_evaluated
from .common import SOURCES, decision_result, k_and, k_at_least, k_or, reject_retired_owner

POLICY_ID = 'IND_STRONG_OR_WEAK_AND_COUNT_V1'


@dataclass(frozen=True)
class IndividualOwner:
    source: str
    owner_hash: str
    high: float
    low: float
    k_high: int
    k_low: int
    k_count: int
    sign: int = 1
    unit: str = 'native'

    def __post_init__(self):
        reject_retired_owner(self.owner_hash)
        if self.source not in SOURCES or not self.owner_hash:
            raise ValueError('SOURCE_OWNER_REQUIRED')
        if not (math.isfinite(self.high) and math.isfinite(self.low)) or self.low > self.high:
            raise ValueError('LOW_LE_HIGH_REQUIRED')
        if any(type(k) is not int or k < 1 for k in (self.k_high, self.k_low, self.k_count)):
            raise ValueError('POSITIVE_INTEGER_COUNTS_REQUIRED')
        if self.sign not in (1, -1):
            raise ValueError('SIGN_PLUS_OR_MINUS_ONE')


def individual_vote(terminals: list, qualified_native_count, owner: IndividualOwner) -> dict:
    """terminals: signed h_b list (None = base not computable); qualified_native_count: int or None.

    Returns an executed record with a binary `vote` (and binary strong/weak/count when determined), or a
    NOT_EVALUATED record without any vote.  Missing evidence is never converted into FAIL."""
    def votes(cut):
        # Spec B4: a terminal base counts only if D = sign*h is strictly positive AND >= cut,
        # so a zero/negative terminal can never confirm even when the owner cut is 0.
        out = []
        for h in terminals:
            if h is None or not math.isfinite(h):
                out.append(not_evaluated('TERMINAL_BASE_NOT_COMPUTABLE'))
            else:
                d = owner.sign * h
                out.append('PASS' if d > 0 and d >= cut else 'FAIL')
        return out
    if not terminals:
        # Zero native levels = no terminal bases exist.  Data absence is an eligibility record,
        # never a fabricated FAIL (work order section 8.1 test 2).
        return decision_result(not_evaluated('NO_NATIVE_TERMINAL_BASES'),
                               source_ids=[owner.source], policy_id=POLICY_ID,
                               owner_hash=owner.owner_hash)
    strong = k_at_least(votes(owner.high), owner.k_high)
    weak = k_at_least(votes(owner.low), owner.k_low)
    count = (not_evaluated('QUALIFIED_NATIVE_COUNT_NOT_AVAILABLE') if qualified_native_count is None
             else ('PASS' if qualified_native_count >= owner.k_count else 'FAIL'))
    vote = k_or(strong, k_and(weak, count))
    comps = {'strong': strong, 'weak': weak, 'count': count}
    determined = {k: v for k, v in comps.items() if isinstance(v, str)}
    return decision_result(vote, source_ids=[owner.source], components=comps, policy_id=POLICY_ID,
                           owner_hash=owner.owner_hash, **determined)


def truth_table() -> dict:
    """I = S OR (W AND C) over the 8 binary combinations of ELIGIBLE components (B02 proof object)."""
    vals = ('PASS', 'FAIL')
    return {(s, w, c): k_or(s, k_and(w, c)) for s, w, c in product(vals, repeat=3)}


def determination_table() -> dict:
    """Partial-input proof table: each component PASS, FAIL or not evaluable (NE).

    The output is a binary vote whenever the evaluable components prove it, otherwise
    DECISION_NOT_DETERMINED_FROM_AVAILABLE_INPUTS (a pre-vote status, not a third vote)."""
    vals = ('PASS', 'FAIL', 'NE')
    conv = {'PASS': 'PASS', 'FAIL': 'FAIL', 'NE': not_evaluated('COMPONENT_NOT_EVALUATED')}
    out = {}
    for s, w, c in product(vals, repeat=3):
        d = k_or(conv[s], k_and(conv[w], conv[c]))
        out[(s, w, c)] = d if isinstance(d, str) else NOT_DETERMINED.reason
    return out
