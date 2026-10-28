"""RQ01 ENTRY -> PEAK -> DE_ESCALATION -> RECOVERY -> EXIT native five-state votes (REV352/REV353 law).

Spec RQ01 six-phase 4-7.  The native law is PRESERVED; only the percentiles become registry
parameters per (source, phase, reference class).  Votes are evaluated on the last THREE distinct
released native observations (a, b, d) of the risk-ORIENTED level z = s_j x (s_DFC = -1):

  ENTRY  E = [ d >= Q^N(p_E)  AND  d - b > 0 ]                                       p_E baseline .90
  PEAK   P = [ max(b,d) >= Q^S(p_P)  AND  b - a > 0  AND  d - b <= b - a ]           p_P baseline .75
  DE     D = [ d >= Q^S(p_D)  AND  d - b < 0 ]                                       p_D baseline .50
  RECOV  R = [ d <= Q^N(p_X)  AND  |d-m| <= B(p_L,p_H)  AND  |d-m| < |b-m| ]         m = Q^N(.5), B = (Q^N(p_H)-Q^N(p_L))/2
  EXIT   X = [ d <= Q^N(p_X)  AND  b <= Q^N(p_X) ]                                   p_X shared with RECOVERY

Consensus = number of the five sources voting 1 >= quorum (3,2,2,3,3); the controller needs
persistence 2 monthly issues.  A source that lacks the native levels a phase needs is
NOT_EVALUATED (absent from the vote), never 0.  PRE_ONSET (pre_onset.py) is a separate auxiliary
detector; this module has no PRE input, which is what makes PRE-on == PRE-off for the chain.
"""
from __future__ import annotations

import calendar
import math
from dataclasses import dataclass, field

import numpy as np

from ..shared_threshold_contract import inclusive_lu_grid
from ..vote_contract import DECISION_NOT_DETERMINED, NonVote, not_evaluated
from .common import SOURCES
from .phases import PhaseController
from .pre_onset import RISK_SIGN

PHASES5 = ('ENTRY', 'PEAK', 'DE_ESCALATION', 'RECOVERY', 'EXIT')
QUORUM = {'ENTRY': 3, 'PEAK': 2, 'DE_ESCALATION': 2, 'RECOVERY': 3, 'EXIT': 3}
PERSISTENCE = 2
BASELINE = {'p_E': .90, 'p_P': .75, 'p_D': .50, 'p_L': .25, 'p_H': .75, 'p_X': .75}
# L/U/STEP grids that the DEV selection procedure may search (illustrative design domain, not fitted)
GRID_DOMAIN = {'p_P': (.65, .90, .05), 'p_D': (.40, .70, .05), 'p_X': (.60, .85, .05),
               'p_L': (.15, .35, .05), 'p_H': (.65, .85, .05)}
# What a CONFIRMED state locks for the NEXT state (spec 6).  ENTRY also fixes the DE fallback so ENTRY->DE without
# a PEAK never uses a threshold learned from the DE event; p_X is locked once at DE and inherited by RECOVERY/EXIT.
LOCK_AFTER = {'ENTRY': ('p_P', 'p_D_fallback'), 'PEAK': ('p_D',), 'DE_ESCALATION': ('p_X', 'p_L', 'p_H'),
              'RECOVERY': ('p_X_inherited_shared_with_EXIT',), 'EXIT': ('EPISODE_CLOSED',)}


def percentile_grid(name: str) -> list:
    """Exact decimal L/U/STEP grid for one percentile parameter."""
    lo, hi, step = GRID_DOMAIN[name]
    return inclusive_lu_grid(lo, hi, step)


@dataclass(frozen=True)
class SourceReference:
    """Oriented DEV native values split by class N / Stress (labels are an input, never produced here)."""
    source: str
    values_N: tuple
    values_S: tuple
    unit: str
    quantile_method: str = 'linear'

    def __post_init__(self):
        if self.source not in SOURCES or not self.unit:
            raise ValueError('SOURCE_AND_UNIT_REQUIRED')
        for v in (self.values_N, self.values_S):
            if len(v) == 0 or not all(math.isfinite(float(x)) for x in v):
                raise ValueError('FINITE_NONEMPTY_REFERENCE_REQUIRED')

    def qN(self, p: float) -> float:
        return float(np.quantile(np.asarray(self.values_N, float), p, method=self.quantile_method))

    def qS(self, p: float) -> float:
        return float(np.quantile(np.asarray(self.values_S, float), p, method=self.quantile_method))


@dataclass(frozen=True)
class PercentileProfile:
    profile_id: str
    p_E: float = BASELINE['p_E']
    p_P: float = BASELINE['p_P']
    p_D: float = BASELINE['p_D']
    p_L: float = BASELINE['p_L']
    p_H: float = BASELINE['p_H']
    p_X: float = BASELINE['p_X']

    def __post_init__(self):
        if not self.profile_id:
            raise ValueError('PROFILE_ID_REQUIRED')
        vals = (self.p_E, self.p_P, self.p_D, self.p_L, self.p_H, self.p_X)
        if not all(0 < v < 1 for v in vals):
            raise ValueError('PERCENTILE_OPEN_UNIT_INTERVAL')
        if self.p_P < self.p_D:
            raise ValueError('PEAK_CUTOFF_MUST_NOT_BE_BELOW_DEESCALATION_CUTOFF')
        if not self.p_L < .5 < self.p_H:
            raise ValueError('RECOVERY_BAND_MUST_STRADDLE_MEDIAN')
        if self.p_E < self.p_X:
            raise ValueError('ENTRY_CEILING_MUST_NOT_BE_BELOW_EXIT_CEILING')


def build_snapshot(ref: SourceReference, prof: PercentileProfile) -> dict:
    """Cutoffs in the source's own oriented unit; quantile method and sample counts are recorded."""
    qe, qx = ref.qN(prof.p_E), ref.qN(prof.p_X)
    if qe < qx:
        raise ValueError('INFEASIBLE_Q_N_ENTRY_BELOW_Q_N_EXIT')
    band = 0.5 * (ref.qN(prof.p_H) - ref.qN(prof.p_L))
    if not band >= 0:
        raise ValueError('NEGATIVE_RECOVERY_BAND')
    return {'source': ref.source, 'profile_id': prof.profile_id, 'unit': ref.unit,
            'quantile_method': ref.quantile_method, 'n_N': len(ref.values_N), 'n_S': len(ref.values_S),
            'entry': qe, 'peak': ref.qS(prof.p_P), 'deesc': ref.qS(prof.p_D), 'exit_ceiling': qx,
            'median_N': ref.qN(.5), 'recovery_band': band}


def phase_votes(levels, snap: dict) -> dict:
    """levels = oriented (a, b, d) newest last; None = not available.

    Returns {phase: 0|1} for the EVALUATED phases only; a phase whose native levels are missing is
    absent (NOT_EVALUATED), never None/0 inside the vote map (GLOBAL NO-U)."""
    if len(levels) != 3:
        raise ValueError('EXACTLY_THREE_SLOTS_A_B_D_REQUIRED')
    a, b, d = levels
    out = {}
    if b is None or d is None:
        return out
    out['ENTRY'] = int(d >= snap['entry'] and (d - b) > 0)
    out['DE_ESCALATION'] = int(d >= snap['deesc'] and (d - b) < 0)
    m = snap['median_N']
    out['RECOVERY'] = int(d <= snap['exit_ceiling'] and abs(d - m) <= snap['recovery_band']
                          and abs(d - m) < abs(b - m))
    out['EXIT'] = int(d <= snap['exit_ceiling'] and b <= snap['exit_ceiling'])
    if a is not None:
        out['PEAK'] = int(max(b, d) >= snap['peak'] and (b - a) > 0 and (d - b) <= (b - a))
    return out


def oriented_last3(rows, issue_month: str, source: str) -> tuple:
    """Last three DISTINCT released native observations as-of the issue, risk-oriented."""
    y, m = int(issue_month[:4]), int(issue_month[5:7])
    end = f'{y:04d}-{m:02d}-{calendar.monthrange(y, m)[1]:02d}'
    seen = {}
    for r in rows:
        if r.available_at <= end and r.period_end <= end:
            if r.period_id in seen:
                raise ValueError('DUPLICATE_NATIVE_PERIOD_VINTAGE')
            seen[r.period_id] = r
    ordered = sorted(seen.values(), key=lambda r: (r.period_end, r.period_id))[-3:]
    s = RISK_SIGN[source]
    vals = tuple(s * r.value for r in ordered)
    return (None,) * (3 - len(vals)) + vals, tuple(r.period_id for r in ordered)


def consensus(votes: dict, phase: str) -> dict:
    """Quorum over the five DISTINCT sources (K-of-5 bound rule).  A source absent from `votes` is not
    evaluated and may still change the outcome: vote 1 iff L >= K, vote 0 iff L + m < K, otherwise the
    record is NOT_EVALUATED (DECISION_NOT_DETERMINED_FROM_AVAILABLE_INPUTS) and carries no vote."""
    if phase not in QUORUM:
        raise ValueError('UNREGISTERED_PHASE')
    k = QUORUM[phase]
    present = {s: votes[s] for s in SOURCES if s in votes and votes[s] is not None}
    if any(v not in (0, 1) for v in present.values()):
        raise ValueError('BINARY_SOURCE_PHASE_VOTES_REQUIRED')
    lower = sum(v == 1 for v in present.values())
    n_unevaluated = len(SOURCES) - len(present)
    upper = lower + n_unevaluated
    out = {'quorum': k, 'confirmed_min': lower, 'possible_max': upper, 'n_unevaluated': n_unevaluated}
    if lower >= k:
        return {'vote': 1, 'execution_status': 'EXECUTED', **out}
    if upper < k:
        return {'vote': 0, 'execution_status': 'EXECUTED', **out}
    return {'execution_status': 'NOT_EVALUATED', 'reason': DECISION_NOT_DETERMINED, **out}


@dataclass
class ThresholdLedger:
    """STATIC_BY_SOURCE_PHASE (primary): one frozen profile per source before the replay starts.

    A state's cutoff must exist before the first vote of its confirming streak; at a confirmation
    only the NEXT state's lock is written (audit trail).  ADAPTIVE re-selection is a separate
    challenger that needs its own activation id and is deliberately not implemented here.
    """
    profile_by_source: dict
    locked_at: str
    mode: str = 'STATIC_BY_SOURCE_PHASE'
    records: list = field(default_factory=list)

    def __post_init__(self):
        if self.mode != 'STATIC_BY_SOURCE_PHASE':
            raise NotImplementedError('ADAPTIVE_AT_PRIOR_STATE_CONFIRMATION_NEEDS_ITS_OWN_ACTIVATION_GATE')
        if set(self.profile_by_source) != set(SOURCES):
            raise ValueError('ALL_FIVE_SOURCE_PROFILES_REQUIRED')

    def lock_next(self, confirmed_state: str, issue_month: str) -> dict:
        names = LOCK_AFTER.get(confirmed_state, ())
        rec = {'confirmed_state': confirmed_state, 'confirmed_at': issue_month, 'locks_for_next': names,
               'effective_from': issue_month, 'mode': self.mode,
               'profile_ids': {s: p.profile_id for s, p in self.profile_by_source.items()}}
        self.records.append(rec)
        return rec


def compose_issue_record(country: str, step_rec: dict, controller: PhaseController, *, pre_individuals: dict | None = None,
                         pre_joints: list | None = None, pre_pool: dict | None = None, pre_obs: dict | None = None) -> dict:
    """The per-country/issue output schema of spec 10 (all PRE parts optional: PRE-off leaves them None)."""
    from .common import canonical_sha256
    from .pairs import PAIRS
    # evaluated PRE votes only; a source/pair without a vote is listed as not evaluated (never None in the map)
    ind = {s: (pre_individuals or {})[s]['vote'] for s in SOURCES
           if ((pre_individuals or {}).get(s) or {}).get('vote') is not None}
    jv = {}
    for j in pre_joints or []:
        if j.get('status') == 'EVALUATED':
            jv['|'.join(j['pair'])] = j['vote']
    pool = pre_pool or {}
    last = controller.trace[-1] if controller.trace else None
    transitioned = bool(last and last['issue_month'] == step_rec['issue_month'])
    start = None
    for t in reversed(controller.trace):
        if t['to'] == controller.phase:
            start = t['issue_month']
            break
    n_eval = {p: len(step_rec['source_votes'][p]) for p in PHASES5}
    pool_masks = pool.get('masks')
    rec = {'country': country, 'issue_at': step_rec['issue_month'],
           'PRE_I': ind, 'PRE_I_not_evaluated': [s for s in SOURCES if s not in ind],
           'PRE_J': jv, 'PRE_J_not_evaluated': ['|'.join(p) for p in PAIRS if '|'.join(p) not in jv],
           'PRE_Pool': ({p: v for p, v in pool_masks.items() if v is not None} if pool_masks is not None else None),
           'PRE_Pool_not_evaluated': ([p for p, v in pool_masks.items() if v is None] if pool_masks is not None
                                      else None),
           'selected_policy': pool.get('locked_policy'),
           'PRE_confirmed_at': step_rec['issue_month'] if (pre_obs or {}).get('confirmed') else None,
           'consensus': dict(step_rec['consensus']),
           'consensus_not_determined': list(step_rec.get('consensus_not_determined', [])),
           'state_before': step_rec['state_before'],
           'state_after': step_rec['state_after'],
           'transition_reason': last['reason'] if transitioned else None,
           'persistence_count': dict(controller.streak), 'last_native_periods': step_rec['native_ids'],
           'phase_start_at': start, 'source_coverage': {p: n_eval[p] / len(SOURCES) for p in PHASES5},
           'eligibility_coverage': {'pre_individual': len(ind) / len(SOURCES),
                                    'pre_joint': len(jv) / len(PAIRS)},
           'source_not_evaluated': dict(step_rec.get('source_not_evaluated', {})),
           'running_observed_peak': step_rec.get('running_observed_peak', {}),
           'issued_status': 'RESEARCH_REPLAY_NOT_ISSUER_RELEASED'}
    rec['provenance_hash'] = canonical_sha256(rec)
    return rec


@dataclass
class FiveStateEngine:
    """Monthly as-of replay: votes -> quorum -> PhaseController(persistence 2).  PRE is NOT an input."""
    snapshots: dict
    ledger: ThresholdLedger
    controller: PhaseController = field(default_factory=lambda: PhaseController(persistence=PERSISTENCE))
    trace: list = field(default_factory=list)
    running_peak: dict = field(default_factory=dict)      # observed-so-far oriented maximum per source
    peak_record: dict | None = None                        # written once when PEAK is confirmed

    def __post_init__(self):
        if set(self.snapshots) != set(SOURCES):
            raise ValueError('ALL_FIVE_SOURCE_SNAPSHOTS_REQUIRED')
        if self.controller.persistence != PERSISTENCE:
            raise ValueError('PERSISTENCE_MUST_BE_TWO_MONTHLY_ISSUES')

    def step(self, issue_month: str, native_rows: dict, *, pre_confirmation: str | None = None) -> dict:
        settled = False
        if self.controller.phase == 'EXIT':
            self.controller.settle_exit(issue_month)
            settled = True
        votes = {p: {} for p in PHASES5}
        ids, reasons, newest = {}, {}, {}
        for s in SOURCES:
            try:
                lv, pid = oriented_last3(native_rows.get(s, ()), issue_month, s)
            except ValueError as exc:
                if str(exc) != 'DUPLICATE_NATIVE_PERIOD_VINTAGE':
                    raise
                # two vintages of one native period: the source is NOT_EVALUATED at this issue (never 0)
                lv, pid, reasons[s] = (None, None, None), (), str(exc)
            ids[s] = pid
            if lv[-1] is not None:
                newest[s] = (lv[-1], pid[-1])
            v = phase_votes(lv, self.snapshots[s])
            for p in PHASES5:
                if p in v:                       # absent = NOT_EVALUATED for this source and phase
                    votes[p][s] = v[p]
        cons = {p: consensus(votes[p], p) for p in PHASES5}
        conf = {p: (('PASS' if c['vote'] == 1 else 'FAIL') if c['execution_status'] == 'EXECUTED'
                    else not_evaluated(c['reason'])) for p, c in cons.items()}
        if pre_confirmation is not None:
            conf['PRE_ONSET'] = pre_confirmation          # auxiliary; ENTRY never reads it
        before = self.controller.phase
        after = self.controller.step(issue_month, conf)
        if after != before and after in PHASES5:
            self.ledger.lock_next(after, issue_month)            # PRE_ONSET / NONE lock nothing
        if after == 'NONE':
            self.running_peak = {}
        elif after in PHASES5:
            for s, (val, per) in newest.items():
                prev = self.running_peak.get(s)
                if prev is None or val > prev[0]:
                    self.running_peak[s] = (val, per)
        if after == 'PEAK' and before != 'PEAK':
            # PEAK is a deceleration candidate confirmed at THIS issue: keep the running observed
            # maximum so far and its provenance; the episode's ultimate (ex-post) maximum is never used.
            self.peak_record = {'peak_confirmed_at': issue_month,
                                'running_observed_peak': {s: {'value': v, 'native_period': p}
                                                          for s, (v, p) in self.running_peak.items()},
                                'definition': 'RUNNING_MAX_SO_FAR_NOT_EXPOST_ARGMAX'}
        rec = {'issue_month': issue_month, 'state_before': before, 'state_after': after,
               'exit_settled_this_issue': settled,
               'running_observed_peak': {s: dict(value=v, native_period=p) for s, (v, p) in self.running_peak.items()},
               'consensus': {p: c['vote'] for p, c in cons.items() if c['execution_status'] == 'EXECUTED'},
               'consensus_not_determined': [p for p, c in cons.items() if c['execution_status'] != 'EXECUTED'],
               'native_ids': ids,
               'source_not_evaluated': reasons,
               'source_votes': votes}
        self.trace.append(rec)
        return rec
