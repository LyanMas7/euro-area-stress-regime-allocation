"""RQ01-07..RQ01-10, RQ01-13, B02(strict): 10 pairs, 9 ordered motifs, true Count, lags, Gate.

Roster (5 choose 2): CISS-CLIFS, CISS-DFC, CISS-GFC, CISS-VIX, CLIFS-DFC, CLIFS-GFC,
CLIFS-VIX, DFC-GFC, DFC-VIX, GFC-VIX. Motifs are ORDERED (first letter = leader a);
six unordered families are a display projection only.

Opportunities: on the issue window grid m_0..m_n (m_n = issue month) a comparison
(i,j), i<j, is a native movement of source s iff s has levels at m_i, m_j and a
different native observation at m_j than at m_i (a carried quarter never moves).
Lag policy lambda gives leader comparison (i,j) and follower comparison (i+l, j+l):
  EXACT(L, delta=0): l = L only.  AT_MOST / MINIMUM_WAIT / delta>0 are rejected at construction
  (RQ01 Joint design lock: exactly one registered lag per configuration, never an OR over lags).
A follower ending after the issue month is NOT matured (pending -> unknown).
  C_ab(m, lambda, u) = sum_{leader q} 1[pred_a(q, m_a) AND exists matured l: pred_b(q+l, m_b)]
reported as the interval [lower, upper]; never min(C_a, C_b) or date-only matching.
Lag 0 reduces to the simultaneous predicate intersection with Frechet bounds
max(0, C_a + C_b - |Q|) <= C_ab <= min(C_a, C_b) on the same opportunity set Q.
Strict pair gate: J_ab = Gate1 AND (C_ab >= k) AND lag-defined AND TermPass_a AND TermPass_b AND Elig.
Gate1 modes (spec RQ01 Joint EXACT(L) 6): STRICT is the REV363 audit baseline and the default; MOTIF_AWARE is an
opt-in EXTENSION (never called "the same strict baseline") that replaces only Gate1 by a per-motif predicate,
and `joint_pair_gate` answers BLOCKED (never PASS, never a vote) when the strict Gate1 has an empty domain for a
Sideway motif.
GLOBAL NO-U: motif letters U/D/S are DIRECTION codes (U = UP).  Every gate component is a binary PASS/FAIL
or a NonVote (non-evaluable: missing/carried native base, unmatured follower, missing stability proof); the
pair vote is binary when the components prove it and otherwise the record is NOT_EVALUATED with no vote.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from itertools import combinations, product

from ..vote_contract import NonVote, blocked, component, is_nonvote, not_evaluated
from .common import SOURCES, decision_result, k_and, reject_retired_owner, result
from .states import StateOwner, classify_path, step_predicate

PAIRS = tuple(combinations(SOURCES, 2))
MOTIFS = tuple(a + b for a, b in product('UDS', repeat=2))
DISPLAY_FAMILIES = {'BOTH_UP': ('UU',), 'BOTH_DOWN': ('DD',), 'BOTH_SIDEWAY': ('SS',),
                    'OPPOSITE': ('UD', 'DU'), 'UP_AND_SIDEWAY': ('US', 'SU'), 'DOWN_AND_SIDEWAY': ('DS', 'SD')}
LAG_KINDS = ('EXACT',)   # design lock: ONLY EXACT(L, delta=0); AT_MOST / MINIMUM_WAIT are not registered
STRICT_POLICY_ID = 'STRICT_JOINT_GATE1_COUNT_MOTIF_LAG_TERM2_V1'
MOTIF_AWARE_POLICY_ID = 'MOTIF_AWARE_GATE1_COUNT_MOTIF_LAG_TERM2_V1'
GATE1_MODES = ('STRICT', 'MOTIF_AWARE')

assert len(PAIRS) == 10 and len(MOTIFS) == 9
assert sorted(m for f in DISPLAY_FAMILIES.values() for m in f) == sorted(MOTIFS)


@dataclass(frozen=True)
class LagPolicy:
    kind: str
    lag: int
    delta: int = 0
    horizon: int | None = None
    owner_hash: str = ''

    def __post_init__(self):
        if self.kind not in LAG_KINDS or type(self.lag) is not int or self.lag < 0 or type(self.delta) is not int \
                or self.delta < 0:
            raise ValueError('LAG_POLICY_DOMAIN')
        if self.delta != 0:
            raise ValueError('EXACT_LAG_REQUIRES_DELTA_ZERO')
        if self.horizon is not None:
            raise ValueError('EXACT_LAG_HAS_NO_HORIZON')
        if not self.owner_hash:
            raise ValueError('LAG_OWNER_HASH_REQUIRED')
        reject_retired_owner(self.owner_hash)

    def lags(self) -> range:
        """The single matured lag L (EXACT with delta=0 is the only registered family)."""
        return range(self.lag, self.lag + 1)


@dataclass(frozen=True)
class TerminalOwner:
    source: str
    owner_hash: str
    base: int
    up_min: float
    down_min: float
    side_max: float

    def __post_init__(self):
        reject_retired_owner(self.owner_hash)
        if not self.owner_hash or type(self.base) is not int or self.base < 1:
            raise ValueError('TERMINAL_OWNER_BASE_HASH_REQUIRED')
        if self.up_min <= 0 or self.down_min <= 0 or self.side_max < 0 or \
                self.side_max >= min(self.up_min, self.down_min):
            raise ValueError('TERMINAL_OWNER_DOMAIN')


def display_projection(ordered_counts: dict) -> dict:
    if set(ordered_counts) != set(MOTIFS):
        raise ValueError('NINE_ORDERED_MOTIFS_REQUIRED')
    return {f: sum(ordered_counts[m] for m in ms) for f, ms in DISPLAY_FAMILIES.items()}


def _movement(series: list, i: int, j: int):
    """series: [(native_id, value) | None] on the window grid -> delta or None."""
    a, b = series[i], series[j]
    if a is None or b is None or a[1] is None or b[1] is None:
        return None
    if a[0] == b[0]:
        return 'CARRIED'
    return b[1] - a[1]



def _distinct_native_matching(edges):
    """Maximum one-to-one matching of distinct SOURCE-NATIVE transitions.

    A carried quarterly observation cannot be reused across different monthly
    triangular slots to manufacture independent Count confirmation.  Calendar
    contrast opportunities remain auditable separately from this evidence count.
    """
    graph = {}
    for left, right in edges:
        graph.setdefault(left, set()).add(right)
    assigned = {}
    def augment(left, visited):
        for right in sorted(graph[left], key=repr):
            if right in visited:
                continue
            visited.add(right)
            if right not in assigned or augment(assigned[right], visited):
                assigned[right] = left
                return True
        return False
    return sum(augment(left, set()) for left in sorted(graph, key=repr))


def _native_id_pair(series, i, j):
    a, b = series[i], series[j]
    if a is None or b is None or a[0] is None or b[0] is None or a[0] == b[0]:
        return None
    return (str(a[0]), str(b[0]))

def joint_count(series_a: list, series_b: list, motif: str, lag: LagPolicy, owner_a: StateOwner,
                owner_b: StateOwner) -> dict:
    """Interval Count of ordered motif `motif` with leader a and follower b."""
    if motif not in MOTIFS:
        raise ValueError('ORDERED_MOTIF_REQUIRED')
    if len(series_a) != len(series_b) or len(series_a) < 2:
        raise ValueError('SAME_WINDOW_GRID_REQUIRED')
    n = len(series_a)
    last = n - 1
    lower = upper = n_opp = c_a = c_b = 0
    hits, pending_ids = [], []
    proven_native_edges, possible_native_edges = [], []
    pending_native_leaders = set()
    for i, j in combinations(range(n), 2):
        da = _movement(series_a, i, j)
        if da is None or da == 'CARRIED':
            continue
        pa = step_predicate(da, motif[0], owner_a)
        followers, pending = [], False
        for ell in lag.lags():
            if j + ell > last:
                pending = True
                continue
            db = _movement(series_b, i + ell, j + ell)
            if db is None or db == 'CARRIED':
                continue
            followers.append(step_predicate(db, motif[1], owner_b))
        if not followers and not pending:
            continue
        n_opp += 1
        c_a += pa == 'PASS'
        c_b += 'PASS' in followers
        hit = pa == 'PASS' and 'PASS' in followers
        # upper bound: the comparison may still confirm if the leader is not a proven FAIL and a follower is
        # PASS, non-evaluable, or not yet matured (pending).  This is interval accounting, not a vote.
        possible = pa != 'FAIL' and ('PASS' in followers or any(is_nonvote(f) for f in followers) or pending)
        lower += hit
        upper += possible
        if hit:
            hits.append((i, j))
        elif possible:
            pending_ids.append((i, j))
        # For a single FIXED exact lag the counted object is a distinct pair
        # of native transitions, not one of many carried-calendar contrasts.
        # The leader confirmation must be in the observable history and the
        # follower's ending index must be <= the current issue index.
        if lag.kind == 'EXACT' and lag.delta == 0:
            lid = _native_id_pair(series_a, i, j)
            ell = lag.lag
            if lid is not None and j + ell <= last:
                rid = _native_id_pair(series_b, i + ell, j + ell)
                if rid is not None:
                    if hit:
                        proven_native_edges.append((lid, rid))
                    if possible:
                        possible_native_edges.append((lid, rid))
            elif lid is not None and j + ell > last and pa != 'FAIL':
                pending_native_leaders.add(lid)
    raw_lower, raw_upper = lower, upper
    exact_fixed = lag.kind == 'EXACT' and lag.delta == 0
    if exact_fixed:
        lower = _distinct_native_matching(proven_native_edges)
        upper = min(len(set(a for a, _ in possible_native_edges) | pending_native_leaders),
                    _distinct_native_matching(possible_native_edges) + len(pending_native_leaders))
        upper = max(lower, upper)
        if lower > upper:
            raise AssertionError('NATIVE_MATCHED_COUNT_BOUNDS_INVALID')
    frechet = (max(0, c_a + c_b - n_opp), min(c_a, c_b))
    if exact_fixed and lag.lag == 0 and raw_lower == raw_upper:
        assert frechet[0] <= raw_lower <= frechet[1], 'RAW_CALENDAR_FRECHET_BOUND_VIOLATED'
    return result('PASS', source_ids=[motif], count_lower=lower, count_upper=upper, n_opportunities=n_opp,
                  marginal_a=c_a, marginal_b=c_b, frechet_bounds=frechet, hit_comparisons=hits,
                  unresolved_comparisons=pending_ids, lag_policy=(lag.kind, lag.lag, lag.delta, lag.horizon),
                  native_distinct_count_applied=exact_fixed, raw_calendar_lower=raw_lower,
                  raw_calendar_upper=raw_upper, native_repeated_slots_removed=raw_lower-lower,
                  pending_distinct_leader_transitions=len(pending_native_leaders),
                  fixed_exact_matured_lag=(lag.lag if exact_fixed else None))


def terminal_pass(series: list, state: str, owner: TerminalOwner):
    """Binary terminal predicate for motif direction letter `state` (U=UP, D=DOWN, S=SIDEWAY).

    A missing, carried (same native id) or non-finite terminal base is a NonVote, never FAIL."""
    if state not in ('U', 'D', 'S'):
        raise ValueError('TERMINAL_STATE_U_D_S')
    if len(series) <= owner.base or series[-1] is None or series[-1 - owner.base] is None:
        return not_evaluated('TERMINAL_BASE_MISSING')
    if (series[-1][0] is None or series[-1-owner.base][0] is None
            or series[-1][0] == series[-1-owner.base][0]):
        return not_evaluated('TERMINAL_BASE_NOT_DISTINCT_NATIVE')
    h = series[-1][1] - series[-1 - owner.base][1]
    if not math.isfinite(h):
        return not_evaluated('TERMINAL_DELTA_NONFINITE')
    if state == 'U':
        return 'PASS' if h >= owner.up_min else 'FAIL'
    if state == 'D':
        return 'PASS' if -h >= owner.down_min else 'FAIL'
    return 'PASS' if abs(h) <= owner.side_max else 'FAIL'


def gate1(series_a: list, series_b: list, base_a: int, base_b: int, scale_a: float, scale_b: float,
          rho: float):
    """Source-normalised unsigned terminal magnitude vs owner N-clean scale, both >= rho.

    Binary when determined (one proven FAIL decides FAIL); otherwise a NonVote."""
    if not (scale_a > 0 and scale_b > 0 and rho >= 0):
        raise ValueError('GATE1_OWNER_SCALE_RATIO_DOMAIN')
    out = []
    for s, b, sc in ((series_a, base_a, scale_a), (series_b, base_b, scale_b)):
        if (len(s) <= b or s[-1] is None or s[-1-b] is None
                or s[-1][0] is None or s[-1-b][0] is None
                or s[-1][0] == s[-1-b][0]):
            out.append(not_evaluated('GATE1_TERMINAL_BASE_NOT_DISTINCT_NATIVE'))
        else:
            out.append('PASS' if abs(s[-1][1] - s[-1 - b][1]) / sc >= rho else 'FAIL')
    return k_and(*out)


def _terminal_delta(series: list, base: int):
    """Newest-minus-base native delta; None unless both ends are DISTINCT, finite native observations."""
    if (len(series) <= base or series[-1] is None or series[-1 - base] is None
            or series[-1][0] is None or series[-1 - base][0] is None
            or series[-1][0] == series[-1 - base][0]):
        return None
    h = series[-1][1] - series[-1 - base][1]
    return h if math.isfinite(h) else None


def path_stability(levels: list, owner: StateOwner):
    """PathStability_j for a Sideway terminal (spec 3): the native sample must classify as an
    equilibrium-bounded SIDEWAY path.  Fewer than n_min natives (or missing levels) is a NonVote
    (non-evaluable proof), never FAIL."""
    vals = list(levels)
    if len(vals) < owner.n_min or any(v is None or not math.isfinite(v) for v in vals):
        return not_evaluated('PATH_STABILITY_FEWER_THAN_N_MIN_NATIVE_LEVELS')
    res = classify_path(vals, owner)
    if res['status'] != 'PASS':
        return not_evaluated('PATH_STABILITY_' + str(res.get('reason')))
    return 'PASS' if res['motif_letter'] == 'S' else 'FAIL'


def strict_gate1_domain(motif: str, term_a: TerminalOwner, term_b: TerminalOwner, scale_a: float, scale_b: float,
                        rho: float) -> dict:
    """Can ANY terminal satisfy the strict unsigned Gate1 (|T|/scale >= rho) AND the motif terminal predicate?

    U/D letters: T >= up_min (or -T >= down_min) AND |T| >= rho*scale is always satisfiable.  A Sideway
    letter needs |T| <= side_max AND |T| >= rho*scale, i.e. side_max >= rho*scale; otherwise the strict
    Gate1 has an EMPTY domain and a Joint PASS for that configuration is impossible, so it is BLOCKED
    (a configuration error), not an ordinary FAIL."""
    if motif not in MOTIFS:
        raise ValueError('ORDERED_MOTIF_REQUIRED')
    if not (scale_a > 0 and scale_b > 0 and rho >= 0):
        raise ValueError('GATE1_OWNER_SCALE_RATIO_DOMAIN')
    per = {}
    for letter, owner, sc in ((motif[0], term_a, scale_a), (motif[1], term_b, scale_b)):
        floor = rho * sc
        if letter == 'S':
            ok = owner.side_max >= floor - 1e-12 * max(1.0, abs(floor))
            per[owner.source] = {'letter': 'S', 'terminal_predicate': '|T| <= side_max', 'side_max': owner.side_max,
                                 'gate1_floor': floor, 'feasible': bool(ok)}
        else:
            per[owner.source] = {'letter': letter, 'terminal_predicate': 'directional minimum',
                                 'gate1_floor': floor, 'feasible': True}
    blocked = [s for s, r in per.items() if not r['feasible']]
    return {'feasible': not blocked, 'blocked_sources': blocked, 'per_source': per,
            'reason': None if not blocked else 'STRICT_GATE1_EMPTY_DOMAIN_FOR_SIDE_MOTIF'}


def gate1_motif_aware(series_a: list, series_b: list, motif: str, owner_a: TerminalOwner, owner_b: TerminalOwner, *,
                      stability_a=None, stability_b=None):
    """OPT-IN Sideway-aware Gate1 (spec 6.2); an extension, NOT the strict REV363 baseline.

      G1^U = [T >= theta_min_U]   G1^D = [-T >= theta_min_D]   G1^S = [|T| <= theta_max_S] AND PathStability
    Source scale/step/terminal owners stay separate.  A Sideway letter without a PathStability proof is a
    NonVote (non-evaluable), never PASS and never FAIL."""
    if motif not in MOTIFS:
        raise ValueError('ORDERED_MOTIF_REQUIRED')
    out = []
    for s, letter, owner, stab in ((series_a, motif[0], owner_a, stability_a),
                                   (series_b, motif[1], owner_b, stability_b)):
        h = _terminal_delta(s, owner.base)
        if h is None:
            out.append(not_evaluated('GATE1_TERMINAL_BASE_NOT_DISTINCT_NATIVE'))
        elif letter == 'U':
            out.append('PASS' if h >= owner.up_min else 'FAIL')
        elif letter == 'D':
            out.append('PASS' if -h >= owner.down_min else 'FAIL')
        elif not abs(h) <= owner.side_max:
            out.append('FAIL')
        elif is_nonvote(stab) or stab in ('PASS', 'FAIL'):
            out.append(stab)
        else:
            out.append(not_evaluated('SIDEWAY_PATH_STABILITY_PROOF_MISSING'))
    return k_and(*out)


def strict_pair_gate(*, pair: tuple, series_a: list, series_b: list, motif: str, lag: LagPolicy,
                     state_a: StateOwner, state_b: StateOwner, term_a: TerminalOwner, term_b: TerminalOwner,
                     k: int, gate1_ratio: float, scale_a: float, scale_b: float, eligible: str = 'PASS') -> dict:
    """J_ab strict policy (the REV363 audit baseline, behaviour unchanged).  Count, terminals and lag are
    pair/interaction owned; raw step and terminal cutoffs stay source owned."""
    return _pair_gate_core(pair=pair, series_a=series_a, series_b=series_b, motif=motif, lag=lag, state_a=state_a,
                           state_b=state_b, term_a=term_a, term_b=term_b, k=k, gate1_ratio=gate1_ratio,
                           scale_a=scale_a, scale_b=scale_b, eligible=eligible, gate1_mode='STRICT')


def joint_pair_gate(*, pair: tuple, series_a: list, series_b: list, motif: str, lag: LagPolicy, state_a: StateOwner,
                    state_b: StateOwner, term_a: TerminalOwner, term_b: TerminalOwner, k: int, gate1_ratio: float,
                    scale_a: float, scale_b: float, eligible: str = 'PASS', gate1_mode: str = 'STRICT',
                    stability_a=None, stability_b=None) -> dict:
    """Front end with an explicit Gate1 mode.  Default STRICT keeps the audit baseline; before running it the
    strict Gate1 domain is checked and an empty domain is reported as a BLOCKED configuration record that
    carries no vote (never PASS, never FAIL)."""
    if gate1_mode not in GATE1_MODES:
        raise ValueError('UNREGISTERED_GATE1_MODE')
    if gate1_mode == 'STRICT':
        dom = strict_gate1_domain(motif, term_a, term_b, scale_a, scale_b, gate1_ratio)
        if not dom['feasible']:
            return result('BLOCKED', dom['reason'], source_ids=list(pair),
                          components={}, components_not_evaluated={'GATE1_DOMAIN': 'EMPTY'}, domain=dom,
                          policy_id=STRICT_POLICY_ID, motif=motif, k=k, gate1_mode='STRICT')
    return _pair_gate_core(pair=pair, series_a=series_a, series_b=series_b, motif=motif, lag=lag, state_a=state_a,
                           state_b=state_b, term_a=term_a, term_b=term_b, k=k, gate1_ratio=gate1_ratio,
                           scale_a=scale_a, scale_b=scale_b, eligible=eligible, gate1_mode=gate1_mode,
                           stability_a=stability_a, stability_b=stability_b)


def gate1_ablation(cases: list) -> dict:
    """Strict vs motif-aware on the SAME native masks (each case = kwargs of joint_pair_gate without gate1_mode)."""
    rows, strict_c, motif_c = [], {}, {}

    def outcome(r):
        # binary vote when executed; otherwise the non-vote execution status (separate counter, not a vote)
        return r['vote'] if r['execution_status'] == 'EXECUTED' else r['execution_status']
    for i, case in enumerate(cases):
        s = joint_pair_gate(gate1_mode='STRICT', **case)
        m = joint_pair_gate(gate1_mode='MOTIF_AWARE', **case)
        strict_c[outcome(s)] = strict_c.get(outcome(s), 0) + 1
        motif_c[outcome(m)] = motif_c.get(outcome(m), 0) + 1
        rows.append({'case': i, 'motif': case['motif'], 'strict_status': s['status'],
                     'strict_vote': s.get('vote'), 'strict_execution_status': s['execution_status'],
                     'motif_aware_vote': m.get('vote'), 'motif_aware_execution_status': m['execution_status']})
    return {'n_cases': len(rows), 'strict_votes': strict_c, 'motif_aware_votes': motif_c,
            'only_strict_pass': sum(r['strict_vote'] == 'PASS' and r['motif_aware_vote'] != 'PASS' for r in rows),
            'only_motif_aware_pass': sum(r['motif_aware_vote'] == 'PASS' and r['strict_vote'] != 'PASS' for r in rows),
            'strict_blocked': sum(r['strict_status'] == 'BLOCKED' for r in rows), 'rows': rows,
            'claim': 'ABLATION_ONLY_MOTIF_AWARE_IS_NOT_EQUIVALENT_TO_STRICT_REV363'}


def _pair_gate_core(*, pair: tuple, series_a: list, series_b: list, motif: str, lag: LagPolicy,
                    state_a: StateOwner, state_b: StateOwner, term_a: TerminalOwner, term_b: TerminalOwner,
                    k: int, gate1_ratio: float, scale_a: float, scale_b: float, eligible='PASS',
                    gate1_mode: str = 'STRICT', stability_a=None, stability_b=None) -> dict:
    if gate1_mode not in GATE1_MODES:
        raise ValueError('UNREGISTERED_GATE1_MODE')
    if pair not in PAIRS and pair[::-1] not in PAIRS:
        raise ValueError('UNREGISTERED_PAIR')
    if type(k) is not int or k < 1:
        raise ValueError('COUNT_K_POSITIVE_INT')
    if state_a.source != pair[0] or state_b.source != pair[1] or term_a.source != pair[0] or term_b.source != pair[1]:
        raise ValueError('SOURCE_OWNER_PAIR_MISMATCH')
    cnt = joint_count(series_a, series_b, motif, lag, state_a, state_b)
    if k > cnt['n_opportunities'] and cnt['n_opportunities'] > 0:
        count_vote = 'FAIL'
    elif cnt['count_lower'] >= k:
        count_vote = 'PASS'
    elif cnt['count_upper'] < k:
        count_vote = 'FAIL'
    else:
        count_vote = not_evaluated('COUNT_INTERVAL_STRADDLES_K')
    if cnt['n_opportunities'] == 0:
        count_vote = not_evaluated('NO_MATCHED_NATIVE_COMPARISON_OPPORTUNITY')
    # A leading indicator confirmed at issue t-lag cannot use its value at
    # issue t as its terminal; that would silently move the leader endpoint.
    # The single registered EXACT(L, delta=0) lag owns the anchor.  LagPolicy rejects every
    # other family at construction, so a flexible lag can never reach this point.
    if lag.kind != 'EXACT' or lag.delta != 0:
        raise ValueError('ONLY_EXACT_LAG_DELTA_ZERO_IS_REGISTERED')
    anchor = series_a
    terminal_clock = 'ISSUE_T'
    if lag.lag > 0:
        anchor = series_a[:-lag.lag]
        terminal_clock = 'LEADER_T_MINUS_L'
    if gate1_mode == 'STRICT':
        g1 = gate1(anchor, series_b, term_a.base, term_b.base, scale_a, scale_b, gate1_ratio)
    else:
        g1 = gate1_motif_aware(anchor, series_b, motif, term_a, term_b, stability_a=stability_a,
                               stability_b=stability_b)
    comp = {'GATE1': g1,
            'COUNT': count_vote, 'TERM_A': terminal_pass(anchor, motif[0], term_a),
            'TERM_B': terminal_pass(series_b, motif[1], term_b),
            'ELIG': component(eligible, 'PAIR_ELIGIBILITY_NOT_PROVEN')}
    vote = k_and(*comp.values())
    if gate1_mode == 'STRICT':
        return decision_result(vote, source_ids=list(pair), components=comp, count=cnt,
                               policy_id=STRICT_POLICY_ID, motif=motif, k=k, terminal_clock=terminal_clock)
    return decision_result(vote, source_ids=list(pair), components=comp, count=cnt,
                           policy_id=MOTIF_AWARE_POLICY_ID, motif=motif, k=k, terminal_clock=terminal_clock,
                           gate1_mode='MOTIF_AWARE', baseline_policy_id=STRICT_POLICY_ID,
                           equivalence_claim='NONE_EXTENSION_NOT_THE_STRICT_REV363_BASELINE')


def pair_survey_and_choice(pair_masks: dict, individual_mask: list, y: list, *, fp_budget: int,
                           source_cap: int = 2) -> dict:
    """RQ01-08 HEURISTIC: survey all 10 pairs on matched keys, then a GREEDY conditional choice.

    The primary selector is the exact enumeration of pair SETS (<= 2^10) in pair_selection.select_pair_set;
    this greedy rule can be strictly worse (source cap / FP budget interactions) and is kept only as a
    comparator and for backwards compatibility.
    pair_masks: {(a,b): [PASS/FAIL or NonVote per key]}; individual_mask: same per key;
    y: 0/1 stress per key. Conditional value of a pair = FN rescued among keys where
    Individual is a proven FAIL and the pair a proven PASS, versus new FP there (preregistered:
    maximise rescued subject to cumulative new FP <= fp_budget, then fewer FP). Non-evaluable keys
    never count as FAIL or PASS. Source overlap cap per source."""
    if set(pair_masks) - set(PAIRS):
        raise ValueError('UNREGISTERED_PAIR')
    n = len(y)
    survey = []
    for p in PAIRS:
        m = pair_masks.get(p)
        if m is None or len(m) != n:
            survey.append({'pair': p, 'status': 'NOT_EVALUATED', 'reason': 'PAIR_MASK_ABSENT'})
            continue
        resc = [i for i in range(n) if individual_mask[i] == 'FAIL' and m[i] == 'PASS' and y[i] == 1]
        newfp = [i for i in range(n) if individual_mask[i] == 'FAIL' and m[i] == 'PASS' and y[i] == 0]
        coex = sum(individual_mask[i] == 'PASS' and m[i] == 'PASS' for i in range(n))
        survey.append({'pair': p, 'status': 'PASS', 'rescued_fn': len(resc), 'new_fp': len(newfp),
                       'coexceedance_with_individual': coex, '_r': set(resc), '_f': set(newfp)})
    chosen, used_r, used_f, per_source = [], set(), set(), {s: 0 for s in SOURCES}
    while True:
        best = None
        for s in survey:
            if s['status'] != 'PASS' or s['pair'] in chosen:
                continue
            if any(per_source[x] >= source_cap for x in s['pair']):
                continue
            add_r, add_f = len(s['_r'] - used_r), len(s['_f'] - used_f)
            if add_r == 0 or len(used_f | s['_f']) > fp_budget:
                continue
            key = (-add_r, add_f, PAIRS.index(s['pair']))
            if best is None or key < best[0]:
                best = (key, s)
        if best is None:
            break
        s = best[1]
        chosen.append(s['pair'])
        used_r |= s['_r']
        used_f |= s['_f']
        for x in s['pair']:
            per_source[x] += 1
    clean = [{k: v for k, v in s.items() if not k.startswith('_')} for s in survey]
    return result('PASS', source_ids=['|'.join(p) for p in chosen], survey=clean, chosen=chosen,
                  rescued_fn=len(used_r), new_fp=len(used_f), fp_budget=fp_budget, source_cap=source_cap,
                  method='GREEDY_HEURISTIC_NOT_PRIMARY_SEE_pair_selection.select_pair_set',
                  rule='max conditional rescued FN s.t. new FP <= budget; coexceedance is not independent information')
