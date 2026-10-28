"""RQ01-16..RQ01-20 / B07: six phases, state-start learning, rescue, as-of controller.

Phases 0..5: PRE_ONSET, ENTRY, PEAK, DE_ESCALATION, RECOVERY, EXIT (start-only labels;
PRE_ONSET is the lookback before ENTRY). Retrospective landmarks of one episode from
an ex-post reference path r (labels only, never as-of features), entry month e, first-N x:
  PEAK = first argmax r on [e, x)            (missing if at e: not a separate start)
  DE_ESCALATION = first m > PEAK with r_m <= r_e + (1 - f_d)(r_peak - r_e)
  RECOVERY = first m > DE_ESCALATION with r_m <= r_e + f_r (r_peak - r_e),  f_r < 1 - f_d
  EXIT = x (missing when right-censored after DEV)
State-start occurrence pi_{j,s} = #eligible sources present at the start / #eligible starts
(non-evaluated sources excluded from the denominator; zero support -> not fittable). Nominal 6^5 = 7776
state configurations per start are combinatorics, not fitted states.
Rescue: final = ENTRY OR (NOT ENTRY AND RESCUE) only where ENTRY predicts N, so
FN_new = FN - rescued_TP and FP_new = FP + rescued_FP on unchanged keys.
GLOBAL NO-U: confirmations/presence/rescue inputs are PASS/FAIL or NonVote; a non-evaluated key is
counted in a separate NOT_EVALUATED counter, never as a confusion-matrix class or as N.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field

from ..vote_contract import ThirdVoteError, is_nonvote, not_evaluated
from .common import SOURCES, k_not, month_from_index, month_index, reject_retired_owner, result

PHASES = ('PRE_ONSET', 'ENTRY', 'PEAK', 'DE_ESCALATION', 'RECOVERY', 'EXIT')
POST_ENTRY = PHASES[1:]
NOMINAL_VALUES_PER_SOURCE = 6


@dataclass(frozen=True)
class LandmarkPolicy:
    policy_id: str
    preonset_lookback: int
    deesc_fraction: float
    recovery_fraction: float

    def __post_init__(self):
        reject_retired_owner(self.policy_id)
        if type(self.preonset_lookback) is not int or self.preonset_lookback < 1:
            raise ValueError('PREONSET_LOOKBACK_POSITIVE_INT')
        if not (0 < self.deesc_fraction < 1 and 0 <= self.recovery_fraction < 1 - self.deesc_fraction):
            raise ValueError('FRACTIONS_REQUIRE_0<f_d<1_AND_0<=f_r<1-f_d')


def wilson95(k: int, n: int):
    if n == 0:
        return (None, None)
    z = 1.959963984540054
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def retrospective_landmarks(episode: dict, reference: dict, policy: LandmarkPolicy, *, dev_last='2019-12') -> dict:
    """episode: entry_month, exit_month, episode_id; reference: {month: value} ex-post path."""
    e, x = episode['entry_month'], episode['exit_month']
    lm = {p: None for p in PHASES}
    reasons = {}
    lm['PRE_ONSET'] = month_from_index(month_index(e) - policy.preonset_lookback)
    lm['ENTRY'] = e
    lm['EXIT'] = x if x <= dev_last else None
    if lm['EXIT'] is None:
        reasons['EXIT'] = 'RIGHT_CENSORED_AFTER_DEV'
    end = min(month_index(x), month_index(dev_last) + 1)
    path = [(month_from_index(i), reference.get(month_from_index(i))) for i in range(month_index(e), end)]
    if not path or any(v is None for _, v in path):
        for p in ('PEAK', 'DE_ESCALATION', 'RECOVERY'):
            reasons[p] = 'REFERENCE_PATH_INCOMPLETE'
        return result('NOT_EVALUATED', 'REFERENCE_PATH_INCOMPLETE', source_ids=[episode.get('episode_id', e)],
                      landmarks=lm, reasons=reasons, policy_id=policy.policy_id)
    r_e = path[0][1]
    peak_i = max(range(len(path)), key=lambda i: (path[i][1], -i))
    r_p = path[peak_i][1]
    if peak_i == 0:
        reasons['PEAK'] = 'PEAK_AT_ENTRY_NOT_A_SEPARATE_START'
    else:
        lm['PEAK'] = path[peak_i][0]
    de_cut = r_e + (1 - policy.deesc_fraction) * (r_p - r_e)
    rec_cut = r_e + policy.recovery_fraction * (r_p - r_e)
    de_i = next((i for i in range(peak_i + 1, len(path)) if path[i][1] <= de_cut), None)
    if de_i is None:
        reasons['DE_ESCALATION'] = 'NOT_OBSERVED_BEFORE_EXIT_OR_DEV_END'
    else:
        lm['DE_ESCALATION'] = path[de_i][0]
        rec_i = next((i for i in range(de_i + 1, len(path)) if path[i][1] <= rec_cut), None)
        if rec_i is None:
            reasons['RECOVERY'] = 'NOT_OBSERVED_BEFORE_EXIT_OR_DEV_END'
        else:
            lm['RECOVERY'] = path[rec_i][0]
    if de_i is None:
        reasons['RECOVERY'] = reasons.get('RECOVERY', 'NO_DE_ESCALATION_START')
    order = [lm[p] for p in PHASES if lm[p] is not None]
    if order != sorted(order) or len(set(order)) != len(order):
        raise AssertionError('LANDMARK_ORDER_VIOLATION')
    return result('PASS', source_ids=[episode.get('episode_id', e)], landmarks=lm, reasons=reasons,
                  policy_id=policy.policy_id, role='RETROSPECTIVE_LANDMARK_POLICY_NOT_ESRB_LABEL')


def state_start_events(landmarks_by_episode: dict) -> list[dict]:
    """Start-only events: one row per present (episode, phase); absent phases are not forced."""
    out = []
    for eid, lm in sorted(landmarks_by_episode.items()):
        country = eid.split('|')[0]
        for p in PHASES:
            if lm.get(p) is not None:
                out.append({'episode_id': eid, 'country': country, 'phase': p, 'month': lm[p]})
    return out


def source_state_occurrence(events: list[dict], presence: dict) -> dict:
    """presence: {(country, month, source): 'PASS'|'FAIL'} computed from as-of information; an absent key
    (or a NonVote value) is a not-evaluated start, excluded from the denominator."""
    table = {}
    for p in POST_ENTRY:
        evs = [e for e in events if e['phase'] == p]
        for s in SOURCES:
            votes = [presence.get((e['country'], e['month'], s)) for e in evs]
            k = sum(isinstance(v, str) and v == 'PASS' for v in votes)
            n = sum(isinstance(v, str) and v in ('PASS', 'FAIL') for v in votes)
            lo, hi = wilson95(k, n)
            table[(p, s)] = {'present': k, 'eligible': n, 'not_evaluated': len(votes) - n, 'n_starts': len(evs),
                             'pi': k / n if n else None, 'wilson95': (lo, hi),
                             'fittable': k > 0, 'note': None if k else 'ZERO_SUPPORT_NOT_FITTED'}
    return result('PASS', source_ids=list(SOURCES), table=table, kind='DESCRIPTIVE_STATE_START_INCIDENCE',
                  probability_claim='NOT_A_POSTERIOR_OR_TRANSITION_PROBABILITY')


def state_start_threshold(start_values: list, reference_values: list, *, fpr_cap: float = .15) -> dict:
    """Learn a source cutoff ONLY from values at true state starts (candidates = start values).

    Preregistered rule: maximise start capture (share of starts >= theta) subject to
    reference-row exceedance < fpr_cap; ties -> larger theta (fewer reference hits)."""
    sv = sorted(v for v in start_values if v is not None and math.isfinite(v))
    rv = [v for v in reference_values if v is not None and math.isfinite(v)]
    if not sv or not rv:
        return result('NOT_EVALUATED', 'NO_STATE_START_OR_REFERENCE_SUPPORT', n_starts=len(sv), n_reference=len(rv))
    best = None
    for theta in sorted(set(sv)):
        cap = sum(v >= theta for v in sv) / len(sv)
        exc = sum(v >= theta for v in rv) / len(rv)
        if exc < fpr_cap and (best is None or (cap, theta) > (best[1], best[0])):
            best = (theta, cap, exc)
    if best is None:
        # every candidate violates the reference cap: no admissible threshold is issued (not a vote)
        return result('NOT_EVALUATED', 'NO_START_CANDIDATE_UNDER_REFERENCE_CAP', candidates=sorted(set(sv)))
    return result('PASS', theta=best[0], start_capture=best[1], reference_exceedance=best[2],
                  candidates=sorted(set(sv)), n_starts=len(sv), learning='STATE_START_ONLY')


def nominal_configurations(n_sources: int = len(SOURCES), values: int = NOMINAL_VALUES_PER_SOURCE) -> int:
    return values ** n_sources


def preonset_rescue(entry: list, rescue: list, y: list, first_onset: list | None = None) -> dict:
    """entry, rescue: PASS/FAIL or NonVote per key; y: 0/1 stress; first_onset: bool per key.

    Only executed (binary) keys enter the confusion matrix; non-evaluated keys are a separate counter."""
    n = len(y)
    if not (len(entry) == len(rescue) == n):
        raise ValueError('SAME_KEYS_REQUIRED')
    final, new_tp, new_fp, gain_first = [], 0, 0, 0
    for i in range(n):
        if entry[i] == 'FAIL' and rescue[i] == 'PASS':
            final.append('PASS')
            new_tp += y[i] == 1
            new_fp += y[i] == 0
            gain_first += bool(first_onset and first_onset[i] and y[i] == 1)
        else:
            final.append(entry[i])

    def cm(pred):
        tp = sum(p == 'PASS' and t == 1 for p, t in zip(pred, y))
        fp = sum(p == 'PASS' and t == 0 for p, t in zip(pred, y))
        fn = sum(p == 'FAIL' and t == 1 for p, t in zip(pred, y))
        tn = sum(p == 'FAIL' and t == 0 for p, t in zip(pred, y))
        return {'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn, 'N_NOT_EVALUATED': n - tp - fp - fn - tn}
    before, after = cm(entry), cm(final)
    assert after['FN'] == before['FN'] - new_tp and after['FP'] == before['FP'] + new_fp
    assert after['N_NOT_EVALUATED'] == before['N_NOT_EVALUATED']
    return result('PASS', before=before, after=after, rescued_tp=new_tp, rescued_fp=new_fp,
                  first_onset_gain=gain_first, final=final, rule='ENTRY OR (NOT ENTRY AND RESCUE) where ENTRY=FAIL')


# ---- B07 / RQ01-18: as-of six-phase controller --------------------------------------
NEXT = {'NONE': ('PRE_ONSET', 'ENTRY'), 'PRE_ONSET': ('ENTRY',), 'ENTRY': ('PEAK', 'DE_ESCALATION'),
        'PEAK': ('DE_ESCALATION',), 'DE_ESCALATION': ('RECOVERY', 'PEAK', 'EXIT'),
        'RECOVERY': ('EXIT', 'PEAK'), 'EXIT': ('NONE',)}


@dataclass
class PhaseController:
    persistence: int = 2
    optional: tuple = ('PRE_ONSET', 'PEAK', 'RECOVERY')
    multiple_waves: bool = True
    phase: str = 'NONE'
    streak: dict = field(default_factory=dict)
    trace: list = field(default_factory=list)
    last_issue: str | None = None

    def __post_init__(self):
        if type(self.persistence) is not int or self.persistence < 1:
            raise ValueError('PERSISTENCE_POSITIVE_INT')

    def allowed(self) -> tuple:
        nxt = list(NEXT[self.phase])
        if self.phase == 'ENTRY' and 'PEAK' not in self.optional:
            nxt.remove('DE_ESCALATION')      # PEAK mandatory: no skip
        if self.phase == 'DE_ESCALATION' and 'RECOVERY' not in self.optional:
            nxt.remove('EXIT')
        if self.phase == 'NONE' and 'PRE_ONSET' not in self.optional:
            nxt.remove('ENTRY')
        if not self.multiple_waves:
            nxt = [p for p in nxt if not (p == 'PEAK' and self.phase in ('DE_ESCALATION', 'RECOVERY'))]
        return tuple(nxt)

    def settle_exit(self, issue_month: str) -> str:
        """EXIT is a transition state; the issue after its confirmation settles EXIT -> NONE."""
        if self.phase != 'EXIT':
            raise ValueError('SETTLE_EXIT_ONLY_FROM_EXIT')
        if self.last_issue is not None and issue_month <= self.last_issue:
            raise ValueError('ISSUES_MUST_BE_STRICTLY_INCREASING_NO_BACKDATING')
        self.trace.append({'issue_month': issue_month, 'from': 'EXIT', 'to': 'NONE',
                           'reason': 'EXIT_SETTLEMENT', 'skipped': []})
        self.phase, self.streak = 'NONE', {}
        return self.phase

    def step(self, issue_month: str, confirmations: dict) -> str:
        """confirmations: {phase: PASS/FAIL or NonVote} computed ONLY from information issued by issue_month.

        PASS extends the streak, FAIL resets it, an absent/non-evaluated confirmation holds it (no vote)."""
        if self.last_issue is not None and issue_month <= self.last_issue:
            raise ValueError('ISSUES_MUST_BE_STRICTLY_INCREASING_NO_BACKDATING')
        self.last_issue = issue_month
        cands = self.allowed()
        # PRE_ONSET is an auxiliary early warning, never a gate on ENTRY: every candidate
        # streak is updated, and ENTRY wins a same-issue tie with PRE_ONSET.
        order = [p for p in cands if p == 'ENTRY'] + [p for p in cands if p != 'ENTRY']
        fired = None
        for p in order:
            v = confirmations.get(p)
            if not (v is None or is_nonvote(v) or v in ('PASS', 'FAIL')):
                raise ThirdVoteError('PHASE_CONFIRMATION_MUST_BE_PASS_FAIL_OR_NONVOTE:' + repr(v))
            if v == 'PASS':
                self.streak[p] = self.streak.get(p, 0) + 1
            elif v == 'FAIL':
                self.streak[p] = 0
            if self.streak.get(p, 0) >= self.persistence and fired is None:
                fired = p
        for p in list(self.streak):
            if p not in cands:
                del self.streak[p]
        if fired:
            skipped = [] if fired in NEXT_STRICT.get(self.phase, ()) else ['SKIP_TO_' + fired]
            self.trace.append({'issue_month': issue_month, 'from': self.phase, 'to': fired,
                               'reason': 'PERSISTENT_ISSUED_CONFIRMATION', 'skipped': skipped})
            # NONE -> PRE_ONSET must not reset the running ENTRY streak (PRE-on == PRE-off for
            # the ENTRY..EXIT chain); every other transition starts a fresh streak.
            keep = {'ENTRY': self.streak['ENTRY']} if fired == 'PRE_ONSET' and 'ENTRY' in self.streak else {}
            self.phase = fired
            self.streak = keep
        return self.phase


NEXT_STRICT = {'NONE': ('PRE_ONSET',), 'PRE_ONSET': ('ENTRY',), 'ENTRY': ('PEAK',), 'PEAK': ('DE_ESCALATION',),
               'DE_ESCALATION': ('RECOVERY', 'PEAK'), 'RECOVERY': ('EXIT', 'PEAK'), 'EXIT': ('NONE',)}


def hysteresis_band(scores: list, enter: float, leave: float) -> dict:
    """Stress on a commensurable score: enter iff s >= enter, leave iff s <= leave, enter > leave."""
    if not enter > leave:
        raise ValueError('ENTRY_THRESHOLD_MUST_EXCEED_EXIT_ON_SAME_SCALE')
    state, out = False, []
    for s in scores:
        if s is None or not math.isfinite(s):
            # missing evidence holds an already-issued latch; before any latch it is not evaluated (no vote)
            out.append(not_evaluated('SCORE_MISSING_NO_ISSUED_LATCH') if not state else 'PASS')
            continue
        if not state and s >= enter:
            state = True
        elif state and s <= leave:
            state = False
        out.append('PASS' if state else 'FAIL')
    return result('PASS', states=out, enter=enter, leave=leave)


def invert(votes: list) -> list:
    return [k_not(v) for v in votes]
