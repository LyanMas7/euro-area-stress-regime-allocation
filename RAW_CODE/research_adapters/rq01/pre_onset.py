"""RQ01 PRE-ONSET = universal Individual -> Joint -> Pool (spec RQ01 six-phase 3, Joint EXACT(L) lock).

Replaces ONLY the PRE detector.  The ENTRY..EXIT chain (five_state.py / phases.py) is untouched:
PRE is an auxiliary early warning, never a gate, never a quorum member and it never resets the
ENTRY streak.  All votes are binary 0/1 on ELIGIBLE inputs; anything that cannot be computed is
reported as NOT_EVALUATED with a reason code (never coerced to 0 / FAIL / N).

  I_PRE_j  = Strong OR (Weak AND Count)                      five sources, risk sign s_j (s_DFC = -1)
  J_PRE_jk = Gate1 AND Timing AND [Strong_J OR (Weak_J AND CountJoint)]
             ten unordered pairs, ONE registered EXACT(L, delta=0) lag per configuration
  Pool     = one of six policies, locked once per run (no default)
  PRE_conf = pre_vote(t) AND pre_vote(t-1) on consecutive monthly issues

Triangle / individual / joint / pool are NOT re-implemented here: they are the single kernel in
research_adapters/ijp_kernel.py.
"""
from __future__ import annotations

import calendar
from dataclasses import dataclass, field

from ..ijp_kernel import EligibilityError, POLICIES, NativeLevel, individual, joint, native_asof, pool
from ..vote_contract import NonVote, decision_fields, not_evaluated
from .common import SOURCES
from .pairs import PAIRS, LagPolicy

RISK_SIGN = {'CISS': 1, 'CLIFS': 1, 'DFC': -1, 'GFC': 1, 'VIX': 1}   # s_DFC = -1 (REV363)
STATUS_OK, STATUS_NE = 'EVALUATED', 'NOT_EVALUATED'
# GLOBAL NO-U: a NOT_EVALUATED record carries no `vote` key at all (eligibility ledger, not a vote).


@dataclass(frozen=True)
class PreIndividualOwner:
    """Source-owned PRE Individual parameters (same RAW unit as the source)."""
    source: str
    owner_hash: str
    theta_hi: object          # scalar or one value per Terminal base (n-1)
    theta_lo: object
    min_bases: int
    count_k: int
    window_native: int = 5    # newest distinct released native observations used
    sign: int = 0             # 0 -> registry sign

    def __post_init__(self):
        if self.source not in SOURCES or not self.owner_hash:
            raise ValueError('SOURCE_OWNER_REQUIRED')
        if type(self.window_native) is not int or self.window_native < 2:
            raise ValueError('WINDOW_NATIVE_GE_2')
        if self.sign not in (0, -1, 1):
            raise ValueError('SIGN_PLUS_MINUS_ONE')

    @property
    def s(self) -> int:
        return self.sign or RISK_SIGN[self.source]


@dataclass(frozen=True)
class PreJointOwner:
    """One registered Joint configuration: ordered (leader, follower) + EXACT lag in issue months."""
    leader: str
    follower: str
    lag: LagPolicy
    owner_hash: str
    high_a: object
    high_b: object
    low_a: object
    low_b: object
    gate1_a: float
    gate1_b: float
    min_bases: int
    count_k: int

    def __post_init__(self):
        if self.leader == self.follower or self.leader not in SOURCES or self.follower not in SOURCES:
            raise ValueError('TWO_DISTINCT_REGISTERED_SOURCES_REQUIRED')
        if tuple(sorted((self.leader, self.follower), key=SOURCES.index)) not in PAIRS:
            raise ValueError('UNREGISTERED_PAIR')
        if self.lag.kind != 'EXACT' or self.lag.delta != 0:
            raise ValueError('ONLY_EXACT_LAG_DELTA_ZERO_IS_REGISTERED')
        if not self.owner_hash:
            raise ValueError('JOINT_OWNER_HASH_REQUIRED')

    @property
    def pair(self) -> tuple:
        return tuple(sorted((self.leader, self.follower), key=SOURCES.index))


def _month_end(month: str) -> str:
    y, m = int(month[:4]), int(month[5:7])
    return f'{y:04d}-{m:02d}-{calendar.monthrange(y, m)[1]:02d}'


def _months_upto(issue_month: str, back: int) -> list[str]:
    y, m = int(issue_month[:4]), int(issue_month[5:7])
    base = y * 12 + m - 1
    out = []
    for k in range(back, -1, -1):
        yy, mm = divmod(base - k, 12)
        out.append(f'{yy:04d}-{mm + 1:02d}')
    return out


def _ne(reason: str, **extra) -> dict:
    return {'status': STATUS_NE, 'execution_status': 'NOT_EVALUATED', 'reason': reason, **extra}


def pre_individual(rows, issue_month: str, owner: PreIndividualOwner) -> dict:
    """I_PRE for one source at one monthly issue; `rows` are NativeLevel (period_id, end, available_at, value)."""
    issue_end = _month_end(issue_month)
    try:
        asof = native_asof(rows, issue_end, '0001-01-01', issue_end)
        if len(asof) < 2:
            raise EligibilityError('requires >= 2 genuine native levels')
        window = asof[-owner.window_native:]
        res = individual([r.value for r in window], owner.s, theta_hi=owner.theta_hi, theta_lo=owner.theta_lo,
                         min_bases=owner.min_bases, count_k=owner.count_k)
    except EligibilityError as exc:
        return _ne(str(exc), source=owner.source)
    return {'status': STATUS_OK, 'vote': res['vote'], 'source': owner.source, 'strong': res['strong'],
            'weak': res['weak'], 'count': res['count'], 'n_native': res['n'],
            'native_ids': [r.period_id for r in window], 'owner_hash': owner.owner_hash}


def _snapshot(rows, month_end: str):
    """Newest released native observation as of the end of a grid month (as-of snapshot)."""
    best = None
    for r in rows:
        if not getattr(r, 'available_at', None) or not getattr(r, 'period_end', None):
            continue
        if r.available_at <= month_end and r.period_end <= month_end:
            if best is None or (r.period_end, r.period_id) > (best.period_end, best.period_id):
                best = r
    return best


def align_exact_native(leader_rows, follower_rows, issue_month: str, lag_months: int, *, back: int = 60) -> dict:
    """Matched anchors under ONE fixed EXACT lag L: leader at m_i paired with follower at m_{i+L}.

    A monthly grid is only the time grid.  An anchor is kept only if BOTH sources show a native
    observation distinct from the previously kept anchor, so a carried quarter can never be
    counted twice.  Follower endpoints after the issue month are PENDING (not matured), never FAIL.
    """
    months = _months_upto(issue_month, back)
    last = len(months) - 1
    anchors, last_l, last_f, pending = [], None, None, 0
    for i in range(0, last + 1):
        j = i + lag_months
        if j > last:
            pending += 1
            continue
        a = _snapshot(leader_rows, _month_end(months[i]))
        b = _snapshot(follower_rows, _month_end(months[j]))
        if a is None or b is None:
            continue
        if last_l is not None and (a.period_id == last_l or b.period_id == last_f):
            continue
        anchors.append((a, b))
        last_l, last_f = a.period_id, b.period_id
    return {'anchors': anchors, 'pending_leader_anchors': pending, 'lag_months': lag_months}


def pre_joint(leader_rows, follower_rows, issue_month: str, owner: PreJointOwner, *, back: int = 60) -> dict:
    """J_PRE for one registered configuration.  Never I_j AND I_k, never min(Count_j, Count_k)."""
    al = align_exact_native(leader_rows, follower_rows, issue_month, owner.lag.lag, back=back)
    pairs = al['anchors']
    if len(pairs) < 2:
        return _ne('INSUFFICIENT_MATCHED_NATIVE_ANCHORS', pair=owner.pair, n_anchors=len(pairs),
                   pending_leader_anchors=al['pending_leader_anchors'])
    s = (RISK_SIGN[owner.leader], RISK_SIGN[owner.follower])
    try:
        res = joint([(a.value, b.value) for a, b in pairs], s, high_a=owner.high_a, high_b=owner.high_b,
                    low_a=owner.low_a, low_b=owner.low_b, gate1_a=owner.gate1_a, gate1_b=owner.gate1_b,
                    min_bases=owner.min_bases, count_k=owner.count_k, timing_valid=True)
    except EligibilityError as exc:
        return _ne(str(exc), pair=owner.pair, n_anchors=len(pairs))
    ids = sorted({a.period_id for a, _ in pairs} | {b.period_id for _, b in pairs})
    return {'status': STATUS_OK, 'vote': res['vote'], 'pair': owner.pair, 'leader': owner.leader,
            'follower': owner.follower, 'lag_months': owner.lag.lag, 'lag_kind': 'EXACT', 'delta': 0,
            'gate1': res['gate1'], 'gate2': res['gate2'], 'paired_count': res['paired_count'],
            'n_anchors': len(pairs), 'pending_leader_anchors': al['pending_leader_anchors'],
            'native_ids': ids, 'owner_hash': owner.owner_hash}


def pre_pools(individual_results: dict, joint_results: list, *, locked_policy: str | None) -> dict:
    """All six masks for comparison plus exactly ONE locked decision (no default policy)."""
    if locked_policy not in POLICIES:
        raise ValueError('LOCKED_POOL_POLICY_REQUIRED')
    inds = {s: r['vote'] for s, r in individual_results.items() if r.get('status') == STATUS_OK}
    joints = {}
    for r in joint_results:
        if r.get('status') == STATUS_OK:
            if r['pair'] in joints:
                raise ValueError('ONE_JOINT_CONFIGURATION_PER_PAIR_AT_AN_ISSUE')
            joints[r['pair']] = r['vote']
    masks, reasons = {}, {}
    for p in POLICIES:
        try:
            masks[p] = pool(inds, joints, p, registry=SOURCES)
        except EligibilityError as exc:
            reasons[p + '_reason'] = str(exc)          # no mask value: the policy is not derivable here
    out = {'locked_policy': locked_policy, 'masks': dict(masks),
           'masks_not_evaluated': [p for p in POLICIES if p not in masks],
           'reasons': reasons,
           'n_eligible_individuals': len(inds), 'n_eligible_joints': len(joints),
           'member_sources': sorted(inds)}
    if locked_policy in masks:
        return {'status': STATUS_OK, 'execution_status': 'EXECUTED', 'vote': masks[locked_policy], **out}
    return {'status': STATUS_NE, 'execution_status': 'NOT_EVALUATED',
            'reason': reasons.get(locked_policy + '_reason', 'POOL_NOT_DERIVABLE'), **out}


@dataclass
class PreOnsetDetector:
    """PRE_conf(t) = pre_vote(t)=1 AND pre_vote(t-1)=1 on consecutive months with NEW native evidence."""
    require_new_native: bool = True
    last_month: str | None = None
    last_vote: int | None = None
    last_ids: frozenset = field(default_factory=frozenset)
    trace: list = field(default_factory=list)

    @staticmethod
    def _idx(month: str) -> int:
        return int(month[:4]) * 12 + int(month[5:7]) - 1

    def observe(self, issue_month: str, vote: int | None, native_ids=()) -> dict:
        if self.last_month is not None and self._idx(issue_month) <= self._idx(self.last_month):
            raise ValueError('ISSUES_MUST_BE_STRICTLY_INCREASING_NO_BACKDATING')
        ids = frozenset(native_ids)
        consecutive = self.last_month is not None and self._idx(issue_month) == self._idx(self.last_month) + 1
        new_native = bool(ids - self.last_ids) if self.last_vote == 1 else True
        reason = None
        if vote is None:
            confirmed, reason = False, 'NOT_EVALUATED'
        elif vote == 1 and self.last_vote == 1 and consecutive:
            confirmed = new_native or not self.require_new_native
            reason = None if confirmed else 'NO_NEW_NATIVE_EVIDENCE_BETWEEN_ISSUES'
        else:
            confirmed = False
        # controller-facing confirmation: PASS/FAIL, or a NonVote that HOLDS the streak (never resets it,
        # never backfills a FAIL): not evaluated, or no new native confirmation since the last PASS.
        if vote is None:
            ctl = not_evaluated('PRE_VOTE_NOT_EVALUATED')
        elif vote == 0:
            ctl = 'FAIL'
        elif new_native or not self.require_new_native or self.last_vote != 1:
            ctl = 'PASS'
        else:
            ctl = not_evaluated('NO_NEW_NATIVE_CONFIRMATION')
        ser = decision_fields(ctl, vote_key='controller_vote')
        rec = {'issue_month': issue_month, 'confirmed': confirmed, 'reason': reason,
               'controller_confirmation': ctl,
               'controller_vote': ser.get('controller_vote'),
               'controller_execution_status': ser['execution_status'],
               'controller_non_vote_reason': ser.get('non_vote_reason')}
        if vote is not None:
            rec['vote'] = vote
        self.trace.append(rec)
        self.last_month, self.last_vote, self.last_ids = issue_month, vote, ids
        return rec
