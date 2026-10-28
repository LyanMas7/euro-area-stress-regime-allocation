"""RQ01 Joint pair registry: configuration identity, N counters, five pair gates, EXACT pair-SET selection.

Spec 'RQ01 Joint pair selection: EXACT(L) only' sections 2, 4, 7, 8 and the IJP matched M00/M01/M10/M11 rule.

* A pair (unordered, 10 of them) and a CONFIGURATION (pair, leader, follower, motif, EXACT lag L, delta=0, unit/sign,
  threshold profile, release rule, owner version) are different objects.  At L=0 there is no lead direction, so
  (j,k,US,0) and (k,j,SU,0) are ONE configuration and share one `joint_id`.
* Four counters are kept apart: N_nominal, N_eligible, N_unique_decision_masks, N_executed.  Duplicates are merged on
  the COMPLETE final decision mask, never on the Individual part alone: a profile whose Individual mask coincides with
  another but whose Joint mask differs is kept.
* Pair registry gates G0..G4 return a status structure.  `pair_selected` is a research-registry decision and does NOT
  imply Joint PASS at any issue.
* Pair SETS are selected by exact enumeration of the (at most 2^10 = 1024) subsets, with
  Rescue(A) = |U_p R_p| and NewFP(A) = |U_p F_p| (never the sum over pairs).  The OR-rescue arithmetic is valid for the
  IND_OR_JOINT pool only; every other pool is judged on its FINAL pool mask.  The greedy rule in pairs.py is a heuristic.
GLOBAL NO-U: a per-key mask entry is a binary vote ('PASS'/'FAIL' or 1/0) or a non-evaluated key (a NonVote
record or None at the ingestion boundary).  Non-evaluated keys are isolated (never read as FAIL or PASS) and
are counted separately; the legacy token 'U' is rejected as an invalid mask entry.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from ..ijp_kernel import geometry
from ..vote_contract import ThirdVoteError, is_nonvote, vote_token
from .common import SOURCES, canonical_sha256
from .pairs import MOTIFS, PAIRS, LagPolicy, pair_survey_and_choice
from .pools import OPERATORS

MAX_PAIR_SETS = 2 ** len(PAIRS)                     # 1024
GATES = ('G0_MECHANISM', 'G1_NATIVE_CLOCK', 'G2_EXACT_MOTIF', 'G3_JOINT_GATE', 'G4_ADDED_VALUE')
GATE_STATUSES = ('PASS', 'FAIL', 'BLOCKED', 'INELIGIBLE', 'RESEARCH_ONLY', 'NOT_EVALUATED')
COUNT_METHOD_OK = 'DISTINCT_NATIVE_ONE_TO_ONE'
PAIR_VETO = ('INELIGIBLE', 'BLOCKED', 'REJECTED')


# ------------------------------------------------------------------ configuration identity
@dataclass(frozen=True)
class JointConfig:
    leader: str
    follower: str
    motif: str                      # ordered: motif[0] = leader state, motif[1] = follower state
    lag: LagPolicy                  # EXACT, delta = 0 (LagPolicy rejects everything else)
    unit_sign: str
    threshold_profile_id: str
    release_rule: str
    owner_version: str

    def __post_init__(self):
        if self.leader == self.follower or self.leader not in SOURCES or self.follower not in SOURCES:
            raise ValueError('TWO_DISTINCT_REGISTERED_SOURCES_REQUIRED')
        if self.motif not in MOTIFS:
            raise ValueError('ORDERED_MOTIF_REQUIRED')
        if not isinstance(self.lag, LagPolicy) or self.lag.kind != 'EXACT' or self.lag.delta != 0:
            raise ValueError('ONLY_EXACT_LAG_DELTA_ZERO_IS_REGISTERED')
        for f in (self.unit_sign, self.threshold_profile_id, self.release_rule, self.owner_version):
            if not isinstance(f, str) or not f:
                raise ValueError('CONFIG_IDENTITY_FIELDS_REQUIRED')

    @property
    def pair(self) -> tuple:
        return tuple(sorted((self.leader, self.follower), key=SOURCES.index))

    def canonical(self) -> 'JointConfig':
        """At L=0 the orientation is not a causal direction: store leader = first source of the sorted pair."""
        if self.lag.lag == 0 and self.leader != self.pair[0]:
            return replace(self, leader=self.follower, follower=self.leader, motif=self.motif[::-1])
        return self


def joint_id(cfg: JointConfig) -> str:
    c = cfg.canonical()
    return canonical_sha256({'pair': list(c.pair), 'leader': c.leader, 'follower': c.follower, 'motif': c.motif,
                             'lag_kind': 'EXACT', 'lag_months': c.lag.lag, 'delta': 0, 'unit_sign': c.unit_sign,
                             'threshold_profile_id': c.threshold_profile_id, 'release_rule': c.release_rule,
                             'owner_version': c.owner_version})


def register_exact_pair(leader: str, follower: str, motif: str, lag_months, *, owner_hash: str, kind: str = 'EXACT',
                        delta: int = 0, unit_sign: str, threshold_profile_id: str, release_rule: str,
                        owner_version: str) -> tuple:
    """rq01_register_exact_pair: reject kind != EXACT or delta != 0 at once; return (canonical config, stable id)."""
    if kind != 'EXACT' or delta != 0:
        raise ValueError('ONLY_EXACT_LAG_DELTA_ZERO_IS_REGISTERED')
    cfg = JointConfig(leader, follower, motif, LagPolicy('EXACT', lag_months, 0, None, owner_hash), unit_sign,
                      threshold_profile_id, release_rule, owner_version).canonical()
    return cfg, joint_id(cfg)


def nominal_config_count(lmax: int) -> int:
    """10 pairs x 9 ordered motifs x (1 orientation at L=0 + 2 orientations for each L=1..Lmax) = 90 (1 + 2 Lmax)."""
    if type(lmax) is not int or lmax < 0:
        raise ValueError('LMAX_NONNEGATIVE_INT')
    return 90 * (1 + 2 * lmax)


def enumerate_nominal_configs(lmax: int, *, lag_owner_hash: str, unit_sign: str, threshold_profile_id: str,
                              release_rule: str, owner_version: str) -> list:
    out = []
    for lag in range(lmax + 1):
        policy = LagPolicy('EXACT', lag, 0, None, lag_owner_hash)
        for pair in PAIRS:
            orientations = [pair] if lag == 0 else [pair, pair[::-1]]
            for lead, follow in orientations:
                for motif in MOTIFS:
                    out.append(JointConfig(lead, follow, motif, policy, unit_sign, threshold_profile_id,
                                           release_rule, owner_version).canonical())
    if len(out) != nominal_config_count(lmax) or len({joint_id(c) for c in out}) != len(out):
        raise AssertionError('NOMINAL_CONFIG_REGISTRY_INCONSISTENT')
    return out


def config_counters(configs, *, eligible: dict, masks: dict, executed) -> dict:
    """N_nominal, N_eligible, N_executed and N_unique_decision_masks kept as FOUR different numbers.

    eligible: {joint_id: bool | (bool, reason)}; masks: {joint_id: complete final decision mask} for executed
    configurations (a mapping with 'individual'/'joint'/'pool' parts, or any JSON-able object); executed: ids run.
    The unique-mask count is over the COMPLETE mask, so equal Individual parts with different Joint parts stay apart."""
    cfgs = list(configs)
    by_id = {}
    for c in cfgs:
        by_id.setdefault(joint_id(c), c)
    ex = set(executed)
    elig = {}
    for cid in by_id:
        v = eligible.get(cid, (False, 'NOT_ASSESSED'))
        elig[cid] = bool(v[0] if isinstance(v, tuple) else v)
    if not ex <= {c for c, ok in elig.items() if ok}:
        raise ValueError('EXECUTED_CONFIGURATION_MUST_BE_ELIGIBLE')
    if any(cid not in masks for cid in ex):
        raise ValueError('EXECUTED_CONFIGURATION_WITHOUT_COMPLETE_MASK')
    complete, individual = {}, {}
    for cid in sorted(ex):
        m = masks[cid]
        complete.setdefault(canonical_sha256(m), []).append(cid)
        part = m.get('individual') if isinstance(m, dict) else None
        individual.setdefault(canonical_sha256(part), []).append(cid)
    out = {'N_nominal': len(by_id), 'N_input_rows': len(cfgs), 'N_canonical_duplicates_merged': len(cfgs) - len(by_id),
           'N_eligible': sum(elig.values()), 'N_executed': len(ex), 'N_unique_decision_masks': len(complete),
           'N_unique_individual_parts': len(individual) if any(isinstance(masks[c], dict) and 'individual' in masks[c] for c in ex) else None,
           'duplicate_groups': [g for g in complete.values() if len(g) > 1],
           'dedupe_rule': 'COMPLETE_FINAL_MASK_ONLY_INDIVIDUAL_COINCIDENCE_IS_NOT_A_DUPLICATE'}
    out['invariant_ok'] = out['N_unique_decision_masks'] <= out['N_executed'] <= out['N_eligible'] <= out['N_nominal']
    if not out['invariant_ok']:
        raise AssertionError('COUNTER_ORDER_VIOLATED')
    return out


# ------------------------------------------------------------------ five pair gates
def joint_threshold_domain(n_anchors: int, min_bases: int, count_k: int, theta_hi, theta_lo) -> dict:
    """Is the Joint threshold domain non-empty?  (1<=b_min<=n-1, 0<=K<=M(n), 0<=weak<=strong everywhere)."""
    reasons = []
    try:
        m, b = geometry(n_anchors)
    except ValueError:
        return {'nonempty': False, 'reasons': ['FEWER_THAN_TWO_MATCHED_ANCHORS']}
    if not 1 <= min_bases <= b:
        reasons.append('MIN_BASES_OUTSIDE_1_TO_N_MINUS_1')
    if not 0 <= count_k <= m:
        reasons.append('COUNT_K_OUTSIDE_0_TO_M')
    hi = list(theta_hi) if isinstance(theta_hi, (list, tuple)) else [theta_hi] * b
    lo = list(theta_lo) if isinstance(theta_lo, (list, tuple)) else [theta_lo] * b
    if len(hi) != b or len(lo) != b:
        reasons.append('THRESHOLD_VECTOR_LENGTH_NOT_N_MINUS_1')
    elif any(l < 0 or l > h for h, l in zip(hi, lo)):
        reasons.append('WEAK_ABOVE_STRONG_OR_NEGATIVE')
    return {'nonempty': not reasons, 'reasons': reasons}


def _g0(e):
    if not e or not isinstance(e.get('mechanism'), str) or not e['mechanism'].strip():
        return 'RESEARCH_ONLY', 'NO_TESTABLE_MECHANISM_HYPOTHESIS'
    if e.get('testable') is not True:
        return 'RESEARCH_ONLY', 'MECHANISM_NOT_TESTABLE'
    return 'PASS', None


def _g1(e):
    if not e:
        return 'INELIGIBLE', 'NATIVE_CLOCK_EVIDENCE_MISSING'
    need = int(e.get('min_matched_pairs', 2))
    if e.get('native_ids_verified') is not True or e.get('release_verified') is not True:
        return 'INELIGIBLE', 'NATIVE_ID_OR_RELEASE_NOT_VERIFIED'
    if int(e.get('n_matched_native_pairs', 0)) < need:
        return 'INELIGIBLE', 'INSUFFICIENT_MATCHED_NATIVE_PAIRS'
    return 'PASS', None


def _g2(e):
    if not e:
        return 'NOT_EVALUATED', 'MOTIF_EVIDENCE_MISSING'
    lag = e.get('lag')
    kind = getattr(lag, 'kind', None) if lag is not None and not isinstance(lag, dict) else (lag or {}).get('kind')
    delta = getattr(lag, 'delta', None) if lag is not None and not isinstance(lag, dict) else (lag or {}).get('delta')
    if kind != 'EXACT' or delta != 0:
        return 'FAIL', 'LAG_NOT_EXACT_DELTA_ZERO'
    if e.get('motif') not in MOTIFS:
        return 'FAIL', 'MOTIF_NOT_ONE_OF_NINE_ORDERED'
    if e.get('count_method') != COUNT_METHOD_OK:
        return 'FAIL', 'COUNT_NOT_DISTINCT_NATIVE_ONE_TO_ONE:' + str(e.get('count_method'))
    return 'PASS', None


def _g3(e):
    if not e:
        return 'NOT_EVALUATED', 'JOINT_GATE_EVIDENCE_MISSING'
    if e.get('owner_present') is not True:
        return 'BLOCKED', 'JOINT_OWNER_MISSING'
    if e.get('gate1_feasible') is False:
        return 'BLOCKED', 'GATE1_DOMAIN_EMPTY'
    if e.get('threshold_domain_nonempty') is False:
        return 'BLOCKED', 'THRESHOLD_DOMAIN_EMPTY'
    if e.get('gate1_feasible') is None or e.get('threshold_domain_nonempty') is None:
        return 'NOT_EVALUATED', 'FEASIBILITY_NOT_CHECKED'
    return 'PASS', None


def _g4(e):
    if not e or e.get('n_labelled', 0) <= 0:
        return 'NOT_EVALUATED', 'NO_MATCHED_LABELLED_POOL_COMPARISON'
    if e.get('fp_budget_ok') is False:
        return 'FAIL', 'NEW_FP_EXCEEDS_BUDGET'
    if e.get('final_mask_equals_individual') is True:
        return 'FAIL', 'OVERLAP_ONLY_NO_INCREMENT_OVER_INDIVIDUAL'
    # AND / complement pools cannot rescue a key the aggregate Individual already
    # turns PASS, so they must be judged on the FINAL pool mask (spec 7.3):
    # strict TP gain, or equal TP with strictly fewer FP.
    if e.get('comparison') == 'FINAL_POOL_MASK':
        if e.get('added_value') is True:
            return 'PASS', None
        if e.get('added_value') is False:
            return 'FAIL', e.get('added_value_reason') or 'FINAL_POOL_MASK_NOT_BETTER_THAN_INDIVIDUAL'
        return 'NOT_EVALUATED', 'FINAL_POOL_MASK_NOT_COMPARED'
    if e.get('rescue') is None:
        return 'NOT_EVALUATED', 'NO_MATCHED_LABELLED_POOL_COMPARISON'
    if e['rescue'] == 0:
        return 'FAIL', 'OVERLAP_ONLY_NO_INCREMENT_OVER_INDIVIDUAL'
    return 'PASS', None


def evaluate_pair_gates(evidence: dict) -> dict:
    """G0 mechanism, G1 native clock (hard veto INELIGIBLE), G2 exact motif, G3 joint gate (hard veto BLOCKED),
    G4 added value.  Registry decision only: `joint_pass_at_issue` is never implied."""
    fns = (_g0, _g1, _g2, _g3, _g4)
    keys = ('mechanism', 'native_clock', 'motif', 'joint_gate', 'added_value')
    gates, reasons = {}, {}
    for name, key, fn in zip(GATES, keys, fns):
        st, why = fn(evidence.get(key))
        gates[name], reasons[name] = st, why
    vals = list(gates.values())
    if gates['G1_NATIVE_CLOCK'] == 'INELIGIBLE':
        status = 'INELIGIBLE'
    elif gates['G3_JOINT_GATE'] == 'BLOCKED':
        status = 'BLOCKED'
    elif 'FAIL' in vals:
        status = 'REJECTED'
    elif 'NOT_EVALUATED' in vals:
        status = 'NOT_EVALUATED'
    elif gates['G0_MECHANISM'] == 'RESEARCH_ONLY':
        status = 'RESEARCH_ONLY'
    else:
        status = 'SELECTED_CANDIDATE'
    return {'gates': gates, 'reasons': {k: v for k, v in reasons.items() if v}, 'pair_status': status,
            'pair_selected': status == 'SELECTED_CANDIDATE', 'research_only_candidate': status == 'RESEARCH_ONLY',
            'joint_pass_at_issue': 'NOT_IMPLIED_BY_PAIR_SELECTION'}


# ------------------------------------------------------------------ exact pair-SET selection
def _norm_vote(v):
    """'PASS'/'FAIL' for a binary vote, None for a non-evaluated key; an invalid mask entry is rejected."""
    if v is None or is_nonvote(v):
        return None
    if isinstance(v, float) and v != v:            # NaN at the ingestion boundary = not evaluated
        return None
    try:
        return vote_token(v.upper() if isinstance(v, str) else v)
    except ThirdVoteError as exc:
        raise ValueError('INVALID_PAIR_MASK_ENTRY_NOT_A_BINARY_VOTE:' + repr(v)) from exc


def _bits(vec) -> tuple:
    """(PASS bitset, FAIL bitset); non-evaluated keys are in neither set."""
    p = f = 0
    for i, v in enumerate(vec):
        t = _norm_vote(v)
        if t == 'PASS':
            p |= 1 << i
        elif t == 'FAIL':
            f |= 1 << i
    return p, f


def _or(a, b):
    return a[0] | b[0], a[1] & b[1]


def _and(a, b):
    return a[0] & b[0], a[1] | b[1]


def _final_mask(op: str, base, j_by_pair: dict, c_by_pair: dict, chosen: tuple, full: int) -> tuple:
    fail = (0, full)
    jany = fail
    for p in chosen:
        jany = _or(jany, j_by_pair[p])
    if op == 'JOINT_ONLY':
        return jany
    if op == 'IND_AND_JOINT':
        return _and(base, jany)
    if op == 'IND_OR_JOINT':
        return _or(base, jany)
    acc = fail
    for p in chosen:
        term = _and(j_by_pair[p], c_by_pair[p]) if op == 'COMPLEMENT_AND_JOINT' else _or(j_by_pair[p], c_by_pair[p])
        acc = _or(acc, term)
    return acc


def _confusion(mask, pos: int, neg: int, full: int) -> dict:
    p, f = mask
    not_evaluated = full & ~(p | f)           # keys without a final vote: separate counters, never a class
    return {'tp': (p & pos).bit_count(), 'fp': (p & neg).bit_count(), 'fn': (f & pos).bit_count(),
            'tn': (f & neg).bit_count(), 'ne_pos': (not_evaluated & pos).bit_count(),
            'ne_neg': (not_evaluated & neg).bit_count()}


def select_pair_set(pair_masks: dict, y: list, *, fp_budget: int, source_cap: int = 2,
                    pool_operator: str = 'IND_OR_JOINT', individual_votes: dict | None = None,
                    individual_mask: list | None = None, pair_status: dict | None = None,
                    pair_configs=None) -> dict:
    """Exact enumeration of every subset A of the registered pairs (<= 1024).

    pair_masks   {pair: per-key mask} - ONE registered configuration per pair (its Joint decision on every key)
    y            per-key 0/1 stress label (None = unlabelled: excluded from every count, never an imputed 0)
    fp_budget    max growth of FALSE POSITIVES of the final pool mask over the Individual-only mask
    source_cap   max number of chosen pairs a single source may belong to (REV363 default 2)
    pool_operator one of the six pool policies; the union formulas Rescue(A)=|U R_p|, NewFP(A)=|U F_p| are exactly the
                 IND_OR_JOINT arithmetic.  JOINT_ONLY / IND_AND_JOINT / COMPLEMENT_* are compared on their FINAL mask.
    individual_votes {source: per-key mask} (all five; needed by the COMPLEMENT pools) or individual_mask (aggregate).
    Objective (deterministic): max TP of the final mask, then min FP, then fewer pairs, then lexicographic pair index.
    """
    if pool_operator not in OPERATORS or pool_operator == 'IND_ONLY':
        raise ValueError('POOL_OPERATOR_WITH_A_JOINT_COMPONENT_REQUIRED')
    if type(fp_budget) is not int or fp_budget < 0 or type(source_cap) is not int or source_cap < 1:
        raise ValueError('FP_BUDGET_NONNEGATIVE_INT_AND_SOURCE_CAP_POSITIVE_INT_REQUIRED')
    y = list(y)
    n = len(y)
    if n == 0 or any(v not in (0, 1, None) for v in y):
        raise ValueError('BINARY_OR_MISSING_LABELS_REQUIRED')
    if set(pair_masks) - set(PAIRS):
        raise ValueError('UNREGISTERED_PAIR')
    if any(len(m) != n for m in pair_masks.values()):
        raise ValueError('SAME_KEYS_REQUIRED')
    needs_votes = pool_operator in ('COMPLEMENT_AND_JOINT', 'COMPLEMENT_OR_JOINT')
    if individual_votes is not None:
        if set(individual_votes) != set(SOURCES) or any(len(v) != n for v in individual_votes.values()):
            raise ValueError('FIVE_SOURCE_INDIVIDUAL_VOTES_ON_THE_SAME_KEYS_REQUIRED')
        base = (0, (1 << n) - 1)
        for s in SOURCES:
            base = _or(base, _bits(individual_votes[s]))
        # an OR over sources that are all FAIL is FAIL; unknown stays unknown (handled by the bitsets)
    elif individual_mask is not None and not needs_votes:
        if len(individual_mask) != n:
            raise ValueError('SAME_KEYS_REQUIRED')
        base = _bits(individual_mask)
    else:
        raise ValueError('INDIVIDUAL_VOTES_REQUIRED_FOR_COMPLEMENT_POOLS' if needs_votes else 'INDIVIDUAL_MASK_REQUIRED')
    full = (1 << n) - 1
    pos = sum(1 << i for i, v in enumerate(y) if v == 1)
    neg = sum(1 << i for i, v in enumerate(y) if v == 0)
    if pos | neg == 0:
        return {'status': 'NOT_EVALUATED', 'reason': 'NO_LABELLED_KEYS'}

    cfgs = {}
    if pair_configs:
        items = pair_configs.items() if isinstance(pair_configs, dict) else [(c.pair, c) for c in pair_configs]
        for p, c in items:
            if c.pair != p or p in cfgs:
                raise ValueError('ONE_REGISTERED_CONFIG_PER_PAIR_REQUIRED')
            cfgs[p] = c
        if set(cfgs) - set(pair_masks):
            raise ValueError('CONFIG_WITHOUT_MASK')
    excluded = {p: 'PAIR_MASK_ABSENT' for p in PAIRS if p not in pair_masks}
    for p, st in (pair_status or {}).items():
        if st in PAIR_VETO and p in pair_masks:
            excluded[p] = 'PAIR_GATE_' + st
    cands = [p for p in PAIRS if p in pair_masks and p not in excluded]
    j_by_pair = {p: _bits(pair_masks[p]) for p in cands}
    c_by_pair = {}
    if needs_votes:
        for p in cands:
            acc = (0, full)
            for s in SOURCES:
                if s not in p:
                    acc = _or(acc, _bits(individual_votes[s]))
            c_by_pair[p] = acc
    base_c = _confusion(base, pos, neg, full)
    rescue_each = {p: (j_by_pair[p][0] & pos & base[1]).bit_count() for p in cands}
    m = len(cands)
    rows, best_key = [], None
    for sel in range(1 << m):
        chosen = tuple(cands[i] for i in range(m) if sel >> i & 1)
        cnt = {s: 0 for s in SOURCES}
        for p in chosen:
            for s in p:
                cnt[s] += 1
        viol = []
        if any(c > source_cap for c in cnt.values()):
            viol.append('SOURCE_CAP')
        fin = _final_mask(pool_operator, base, j_by_pair, c_by_pair, chosen, full)
        cm = _confusion(fin, pos, neg, full)
        fp_inc = cm['fp'] - base_c['fp']
        if fp_inc > fp_budget:
            viol.append('FP_BUDGET')
        row = {'pairs': chosen, 'size': len(chosen), 'valid': not viol, 'violations': viol, **cm,
               'rescued_fn': (fin[0] & pos & base[1]).bit_count(), 'new_fp': (fin[0] & neg & base[1]).bit_count(),
               'lost_tp': (pos & base[0] & ~fin[0]).bit_count(), 'avoided_fp': (neg & base[0] & ~fin[0]).bit_count(),
               'fp_increase': fp_inc, 'sum_pair_rescues': sum(rescue_each[p] for p in chosen),
               'final_mask_sha': canonical_sha256([fin[0], fin[1], n]), 'duplicate_of': None,
               '_idx': tuple(PAIRS.index(p) for p in chosen)}
        rows.append(row)
    valid = sorted((r for r in rows if r['valid']), key=lambda r: (-r['tp'], r['fp'], r['size'], r['_idx']))
    seen = {}
    for r in valid:
        if r['final_mask_sha'] in seen:
            r['duplicate_of'] = seen[r['final_mask_sha']]['pairs']
        else:
            seen[r['final_mask_sha']] = r
    best = valid[0]                       # the empty set is always valid for the OR pool; AND pools may be empty too
    ledger = [{k: v for k, v in r.items() if k != '_idx'} for r in rows]
    return {'status': 'PASS', 'pool_operator': pool_operator, 'chosen': best['pairs'],
            'rescued_fn': best['rescued_fn'], 'new_fp': best['new_fp'], 'lost_tp': best['lost_tp'],
            'avoided_fp': best['avoided_fp'], 'final_confusion': {k: best[k] for k in ('tp', 'fp', 'fn', 'tn', 'ne_pos', 'ne_neg')},
            'baseline_confusion': base_c, 'fp_increase': best['fp_increase'],
            'sum_pair_rescues': best['sum_pair_rescues'],
            'overlap_double_count': best['sum_pair_rescues'] - best['rescued_fn'] if pool_operator == 'IND_OR_JOINT' else None,
            'rescue_union_formula_applies': pool_operator == 'IND_OR_JOINT',
            'comparison': 'OR_RESCUE_UNION' if pool_operator == 'IND_OR_JOINT' else 'FINAL_POOL_MASK',
            'n_candidate_pairs': m, 'n_subsets_enumerated': 1 << m, 'n_valid_subsets': len(valid),
            'n_unique_final_masks': len(seen), 'n_duplicate_final_masks': len(valid) - len(seen),
            'n_unlabelled_keys': y.count(None), 'excluded_pairs': excluded, 'fp_budget': fp_budget,
            'source_cap': source_cap, 'config_ids': {p: joint_id(c) for p, c in cfgs.items()},
            'ledger': ledger, 'improves_on_individual': best['tp'] > base_c['tp'] or best['fp'] < base_c['fp'],
            'objective': 'MAX_TP_FINAL_THEN_MIN_FP_FINAL_THEN_FEWER_PAIRS_THEN_LEX_PAIR_INDEX',
            'selection_claim': 'RESEARCH_REGISTRY_DECISION_NOT_JOINT_PASS_AT_ANY_ISSUE'}


def compare_with_greedy(pair_masks: dict, individual_mask: list, y: list, *, fp_budget: int,
                        source_cap: int = 2) -> dict:
    """Exact IND_OR_JOINT selector vs the legacy greedy heuristic on the same keys."""
    exact = select_pair_set(pair_masks, y, fp_budget=fp_budget, source_cap=source_cap,
                            pool_operator='IND_OR_JOINT', individual_mask=individual_mask)
    greedy = pair_survey_and_choice(pair_masks, individual_mask, y, fp_budget=fp_budget, source_cap=source_cap)
    return {'exact_chosen': exact['chosen'], 'exact_rescued': exact['rescued_fn'], 'exact_new_fp': exact['new_fp'],
            'greedy_chosen': tuple(greedy['chosen']), 'greedy_rescued': greedy['rescued_fn'],
            'greedy_new_fp': greedy['new_fp'], 'greedy_gap_rescued': exact['rescued_fn'] - greedy['rescued_fn'],
            'greedy_is_optimal': exact['rescued_fn'] == greedy['rescued_fn'] and exact['new_fp'] == greedy['new_fp']}


# ------------------------------------------------------------------ matched M00 / M01 / M10 / M11
FOUR_M = ('M00', 'M01', 'M10', 'M11')


def _loss(kind, pred, y):
    import math
    n = len(y)
    if callable(kind):
        return float(kind(pred, y))
    if kind == 'ZERO_ONE':
        return sum((1 if p >= 0.5 else 0) != t for p, t in zip(pred, y)) / n
    if kind == 'BRIER':
        return sum((p - t) ** 2 for p, t in zip(pred, y)) / n
    if kind == 'LOGLOSS':
        eps = 1e-12
        return -sum(t * math.log(min(max(p, eps), 1 - eps)) + (1 - t) * math.log(min(max(1 - p, eps), 1 - eps))
                    for p, t in zip(pred, y)) / n
    raise ValueError('UNREGISTERED_LOSS')


def matched_four_m(forecasts: dict, y: list, keys: list, *, loss='LOGLOSS', joint_off: bool = False,
                   tol: float = 0.0) -> dict:
    """M00 benchmark; M01 +Individual; M10 +Pure Joint; M11 +both, on IDENTICAL keys/denominator/labels/target/fit.

    forecasts = {M: {'pred': [...], 'target': str, 'fit_artifact': str}}.  Delta-loss values are only reported when
    the four forecasts are genuinely matched.  With joint_off=True the invariants M10 == M00 and M11 == M01 must hold
    (no other change may sneak in), otherwise the helper reports INVARIANT_VIOLATION instead of a gain."""
    if set(forecasts) != set(FOUR_M):
        return {'status': 'NOT_MATCHED', 'reason': 'ALL_FOUR_M_REQUIRED'}
    n = len(keys)
    if len(set(keys)) != n or len(y) != n or n == 0:
        return {'status': 'NOT_MATCHED', 'reason': 'UNIQUE_KEYS_AND_LABELS_ON_THE_SAME_KEYS_REQUIRED'}
    if any(len(forecasts[m]['pred']) != n for m in FOUR_M):
        return {'status': 'NOT_MATCHED', 'reason': 'FORECAST_LENGTH_DIFFERS_FROM_KEYS'}
    for field in ('target', 'fit_artifact'):
        if len({forecasts[m].get(field) for m in FOUR_M}) != 1:
            return {'status': 'NOT_MATCHED', 'reason': field.upper() + '_DIFFERS_ACROSS_MODELS'}
    avail = [all(forecasts[m]['pred'][i] is not None for m in FOUR_M) and y[i] in (0, 1) for i in range(n)]
    per_model_missing = {m: sum(forecasts[m]['pred'][i] is None for i in range(n)) for m in FOUR_M}
    if len(set(per_model_missing.values())) != 1 or any(
            (forecasts[m]['pred'][i] is None) != (forecasts['M00']['pred'][i] is None)
            for m in FOUR_M for i in range(n)):
        return {'status': 'NOT_MATCHED', 'reason': 'DENOMINATOR_DIFFERS_ACROSS_MODELS',
                'missing_by_model': per_model_missing}
    idx = [i for i, ok in enumerate(avail) if ok]
    if not idx:
        return {'status': 'NOT_EVALUATED', 'reason': 'NO_COMMON_LABELLED_KEYS'}
    P = {m: [forecasts[m]['pred'][i] for i in idx] for m in FOUR_M}
    yy = [y[i] for i in idx]
    if joint_off:
        for a, b in (('M10', 'M00'), ('M11', 'M01')):
            bad = [keys[idx[j]] for j in range(len(idx)) if abs(P[a][j] - P[b][j]) > tol]
            if bad:
                return {'status': 'INVARIANT_VIOLATION', 'reason': f'{a}_MUST_EQUAL_{b}_WHEN_JOINT_IS_OFF',
                        'first_mismatch_key': bad[0], 'n_mismatch': len(bad)}
    L = {m: _loss(loss, P[m], yy) for m in FOUR_M}
    return {'status': 'MATCHED', 'n_matched': len(idx), 'n_dropped_unlabelled': n - len(idx),
            'n_positive': int(sum(yy)), 'loss': L, 'joint_off': joint_off,
            'delta_loss_individual': L['M00'] - L['M01'], 'delta_loss_joint_alone': L['M00'] - L['M10'],
            'delta_loss_joint_given_individual': L['M01'] - L['M11'],
            'sign_convention': 'POSITIVE_DELTA_MEANS_THE_RICHER_MODEL_HAS_LOWER_LOSS',
            'dependence_caveat': 'KEYS_OF_ONE_EPISODE_ARE_NOT_INDEPENDENT'}
