"""RQ02 ENTRY / EXIT: the universal binary Individual -> Joint -> Pool applied to the native RQ02 geometry.

Spec: 'Individual_Joint_Pool ap dung RQ01, RQ02' sections E2 (Pre-Onset), E3 (EXIT), G2, H, I3 and
'RQ02_2 nhanh exit' 3.8-3.9.  The arithmetic (triangle, Strong OR (Weak AND Count), paired Joint, pair-indexed
Pool) lives ONLY in research_adapters/ijp_kernel.py; this module decides WHICH native levels, anchors, pairs
and branches are fed to it:

  * ENTRY horizons H15/H12/H10/H8/H6 = T-45/-36/-30/-24/-18 months (the number after H counts QUARTERS).
    Long (CG, DSR-P) uses the anchor snapshots H15..h: n = 1..5 levels -> confirmations 0..4, Count pairs
    0,1,3,6,10, Terminal bases 0..4.  H15 has no cross-horizon confirmation: NOT_APPLICABLE, never 0.
  * Short stages S1..S4 are used only at their END horizon (S1 H15->H12 = 9 months, S2..S4 = 6 months);
    S5 is retrospective and never feeds an issued forecast.  Monthly levels n = 10 / 7, quarterly 4 / 3 when
    complete; they SHRINK to the genuine released levels (no forward fill, no invented baseline quarter).
  * 7 Short EWI: CISS, CLIFS, GFC, VIX are monthly-confirmation sources (weekly / daily provenance is kept, the
    monthly level is the newest native observation of that month); CG, DSR-P, DFC are quarterly.
  * 21 Short Joint pairs = 6 (M x M, monthly grid) + 12 (M x Q, QUARTERLY master anchor) + 3 (Q x Q); the Long
    Joint is the single pair CG x DSR-P.  Anchors are ECONOMIC periods that are both released by the issue.
  * Pool: IL, IS, JL, JS -> three role policies (Sensitive = IND_OR_JOINT, Confirmed = IND_AND_JOINT,
    ComplementConfirmed = COMPLEMENT_AND_JOINT) on the pair-indexed kernel pool.
  * EXIT: anchors T-18/-12/-6/-3 (confirmations 0/1/2/3, Count 0/1/3/6, Terminal 0/1/2/3), Short S1..S4 only
    (no S5, no portfolio, no VaR/ES, no RoundBack), EXIT-A is K-of-4 over IL/IS/JL/JS, and the EXIT nominal
    budget is NOT the ENTRY 9,175,534.

Votes are strictly 0/1 on ELIGIBLE inputs.  Whatever cannot be computed is reported NOT_EVALUATED (with a
reason code) in the CoverageLedger; it is never coerced to 0, FAIL or N.  No OOS data, no fitting here.
"""
from __future__ import annotations

import calendar
import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime
from itertools import combinations
from typing import Mapping, Sequence

from .ijp_kernel import (EligibilityError, NativeLevel, geometry, individual, joint, native_asof, pool)

EA7 = ('BE', 'DE', 'ES', 'FI', 'FR', 'IT', 'NL')
SHORT_SOURCES = ('CISS', 'CLIFS', 'DFC', 'GFC', 'VIX', 'CG', 'DSR-P')
LONG_SOURCES = ('CG', 'DSR-P')
NATIVE_FREQUENCY = {'CISS': 'W', 'CLIFS': 'M', 'DFC': 'Q', 'GFC': 'M', 'VIX': 'D', 'CG': 'Q', 'DSR-P': 'Q'}
MONTHLY_CONFIRMATION = ('CISS', 'CLIFS', 'GFC', 'VIX')
QUARTERLY_CONFIRMATION = ('DFC', 'CG', 'DSR-P')

ENTRY_HORIZON_MONTHS = {'H15': 45, 'H12': 36, 'H10': 30, 'H8': 24, 'H6': 18}
ENTRY_HORIZONS = tuple(ENTRY_HORIZON_MONTHS)
ENTRY_STAGE_BOUNDS = {'S1': ('H15', 'H12'), 'S2': ('H12', 'H10'), 'S3': ('H10', 'H8'), 'S4': ('H8', 'H6')}
STAGE_ENDING_AT = {end: stage for stage, (_, end) in ENTRY_STAGE_BOUNDS.items()}
ENTRY_ROLE_POLICY = {'Sensitive': 'IND_OR_JOINT', 'Confirmed': 'IND_AND_JOINT',
                     'ComplementConfirmed': 'COMPLEMENT_AND_JOINT'}

EXIT_ANCHOR_MONTHS = (18, 12, 6, 3)                       # months before the target month T
EXIT_HORIZONS = EXIT_ANCHOR_MONTHS
EXIT_SLOT_NAMES = {18: 'H6', 12: 'H4', 6: 'H2', 3: 'H1'}  # quarter-named slots used by rq02_source native_bridge
EXIT_STAGE_BOUNDS = {'S1': (18, 12), 'S2': (12, 6), 'S3': (6, 3), 'S4': (3, 0)}   # NO S5 in EXIT
EXIT_FAMILIES = ('IL', 'IS', 'JL', 'JS')
EXIT_BUDGET_STATUS = 'NOT_REGISTERED'

SAMPLING_RULE = 'LAST_NATIVE_OBSERVATION_OF_MONTH'        # registered rule for weekly / daily -> monthly snapshot
MAX_PROFILES = {'S1-S4': 80, 'S5': 40}


# ------------------------------------------------------------------------------------------ month arithmetic
def month_index(x) -> int:
    if isinstance(x, (date, datetime)):
        return x.year * 12 + x.month - 1
    s = str(x)
    if len(s) < 7 or s[4] != '-':
        raise ValueError('MONTH_OR_ISO_DATE_REQUIRED')
    return int(s[:4]) * 12 + int(s[5:7]) - 1


def month_label(idx: int) -> str:
    return f'{idx // 12:04d}-{idx % 12 + 1:02d}'


def month_end(idx: int) -> str:
    y, m0 = divmod(idx, 12)
    return f'{y:04d}-{m0 + 1:02d}-{calendar.monthrange(y, m0 + 1)[1]:02d}'


def is_quarter_end_month(idx: int) -> bool:
    return idx % 12 in (2, 5, 8, 11)


def _ne(reason: str, **extra) -> dict:
    return {'status': 'NOT_EVALUATED', 'vote': None, 'reason': reason, **extra}


def _na(reason: str, **extra) -> dict:
    return {'status': 'NOT_APPLICABLE', 'vote': None, 'reason': reason, **extra}


# ------------------------------------------------------------------------------------------ ENTRY geometry
def build_entry_horizons(onset_T) -> list:
    """Retrospective scheduling of the five issue keys for an Onset target T (T is never a feature)."""
    t = month_index(onset_T)
    rows = []
    for i, (h, months) in enumerate(ENTRY_HORIZON_MONTHS.items()):
        n_anchors = i + 1
        pairs, terminal = geometry(n_anchors) if n_anchors >= 2 else (0, 0)
        rows.append({'horizon': h, 'quarters': int(h[1:]), 'months_before_T': months,
                     'issue_month': month_label(t - months), 'issue_date': month_end(t - months),
                     'long_anchors_known': n_anchors, 'long_confirmations': i, 'long_count_pairs': pairs,
                     'long_terminal_bases': terminal,
                     'cross_horizon_confirmation': 'NOT_APPLICABLE' if i == 0 else 'APPLICABLE',
                     'short_stage_completed': None if i == 0 else f'S{i}',
                     'scope': 'RETROSPECTIVE_SCHEDULING_ONLY'})
    return rows


def stage_window(stage: str, onset_T) -> dict:
    if stage not in ENTRY_STAGE_BOUNDS:
        raise ValueError('ENTRY_STAGE_S1_TO_S4_ONLY_S5_IS_RETROSPECTIVE')
    a, b = ENTRY_STAGE_BOUNDS[stage]
    t = month_index(onset_T)
    m0, m1 = t - ENTRY_HORIZON_MONTHS[a], t - ENTRY_HORIZON_MONTHS[b]
    return {'stage': stage, 'start_idx': m0, 'end_idx': m1, 'start': month_label(m0), 'end': month_label(m1),
            'length_months': m1 - m0, 'issue_horizon': b, 'issue_date': month_end(m1)}


def calendar_levels(group: str, m_start: int, m_end: int, *, allow_prior_quarter_baseline: bool = False) -> int:
    """Expected calendar levels of a closed month window (the 'planned' count, logged next to the eligible one)."""
    if m_end < m_start:
        raise ValueError('WINDOW_END_BEFORE_START')
    if group == 'M':
        return m_end - m_start + 1
    if group == 'Q':
        n = sum(1 for m in range(m_start, m_end + 1) if is_quarter_end_month(m))
        if allow_prior_quarter_baseline and not is_quarter_end_month(m_start):
            n += 1
        return n
    raise ValueError('GROUP_M_OR_Q')


def source_group(source: str) -> str:
    if source not in SHORT_SOURCES:
        raise ValueError('UNKNOWN_EWI')
    return 'M' if source in MONTHLY_CONFIRMATION else 'Q'


def expected_stage_geometry(stage: str, group: str, onset_T, *, allow_prior_quarter_baseline: bool = False) -> dict:
    w = stage_window(stage, onset_T)
    n = calendar_levels(group, w['start_idx'], w['end_idx'], allow_prior_quarter_baseline=allow_prior_quarter_baseline)
    pairs, terminal = geometry(n) if n >= 2 else (0, 0)
    return {'stage': stage, 'group': group, 'n_levels': n, 'count_pairs': pairs, 'terminal_bases': terminal,
            'length_months': w['length_months']}


def short_pairs() -> tuple:
    return tuple(combinations(SHORT_SOURCES, 2))


def pair_group(a: str, b: str) -> str:
    ga, gb = source_group(a), source_group(b)
    return 'MM' if ga == gb == 'M' else 'QQ' if ga == gb == 'Q' else 'MQ'


def pair_groups() -> dict:
    out = {'MM': [], 'MQ': [], 'QQ': []}
    for a, b in short_pairs():
        out[pair_group(a, b)].append((a, b))
    return out


def profile_count(n_levels: int, cap: int) -> int:
    """Threshold profiles per source/sign/stage: 3^n cells (1 Count + (n-1) Terminal), capped by the budget."""
    if type(n_levels) is not int or n_levels < 2:
        raise EligibilityError('requires >= 2 genuine native levels')
    return min(3 ** n_levels, cap)


def derived_stage_K() -> dict:
    """K_L, K_M, K_Q per stage derived from the geometry (spec H3 table) - cross-check of the nominal budget."""
    out = {}
    for stage, long_n in (('S1', 2), ('S2', 3), ('S3', 4), ('S4', 5)):
        g = {'M': 10 if stage == 'S1' else 7, 'Q': 4 if stage == 'S1' else 3}
        cap = MAX_PROFILES['S1-S4']
        out[stage] = (profile_count(long_n, cap), profile_count(g['M'], cap), profile_count(g['Q'], cap))
    return out


def stage_budget(k_long: int, k_monthly: int, k_quarterly: int, *, gate2_variants: int = 12,
                 pool_bundles: int = 28, n_policies: int = 3) -> dict:
    """Spec H4 module-stage-sign counts for one S1..S4 stage (two signs)."""
    il = 2 * 2 * k_long
    is_ = 2 * (4 * k_monthly + 3 * k_quarterly)
    jl = 2 * k_long * k_long * gate2_variants
    js = 2 * gate2_variants * (6 * k_monthly ** 2 + 12 * k_monthly * k_quarterly + 3 * k_quarterly ** 2)
    po = 2 * pool_bundles * pool_bundles * n_policies
    return {'IL': il, 'IS': is_, 'JL': jl, 'JS': js, 'Pool': po, 'total': il + is_ + jl + js + po}


def s5_budget(k_short: int = 40, *, gate2_variants: int = 12, pool_bundles: int = 28, n_policies: int = 3) -> dict:
    is_ = 2 * 7 * k_short
    js = 2 * 21 * k_short * k_short * gate2_variants
    po = 2 * pool_bundles * pool_bundles * n_policies
    return {'IL': 0, 'IS': is_, 'JL': 0, 'JS': js, 'Pool': po, 'total': is_ + js + po}


# ------------------------------------------------------------------------------------------ native levels
def _released(rows: Sequence[NativeLevel], issue_iso: str) -> list:
    return native_asof(rows, issue_iso, '0001-01-01', issue_iso)


def level_map(rows: Sequence[NativeLevel], source: str, m_start: int, m_end: int, issue_iso: str) -> dict:
    """{month index -> NativeLevel}: only RELEASED natives; no carry, no fill, one observation per month/quarter."""
    freq = NATIVE_FREQUENCY[source]
    out: dict = {}
    for r in _released(rows, issue_iso):
        mi = month_index(r.period_end)
        if not m_start <= mi <= m_end:
            continue
        if freq == 'Q':
            if not is_quarter_end_month(mi):
                raise EligibilityError('NATIVE_QUARTER_PERIOD_MISMATCH')
            if mi in out:
                raise EligibilityError('MULTIPLE_NATIVE_OBSERVATIONS_IN_ONE_QUARTER')
            out[mi] = r
        elif freq == 'M':
            if mi in out:
                raise EligibilityError('MULTIPLE_NATIVE_OBSERVATIONS_IN_ONE_MONTH')
            out[mi] = r
        else:           # weekly / daily provenance kept; monthly SNAPSHOT = newest native observation of the month
            cur = out.get(mi)
            if cur is None or (r.period_end, r.period_id) > (cur.period_end, cur.period_id):
                out[mi] = r
    return out


def stage_levels(rows: Sequence[NativeLevel], source: str, m_start: int, m_end: int, issue_iso: str, *,
                 allow_prior_quarter_baseline: bool = False) -> dict:
    group = source_group(source)
    lm = level_map(rows, source, m_start, m_end, issue_iso)
    baseline = None
    if group == 'Q' and allow_prior_quarter_baseline and not is_quarter_end_month(m_start):
        prev = m_start - 1
        while not is_quarter_end_month(prev):
            prev -= 1
        baseline = level_map(rows, source, prev, prev, issue_iso).get(prev)
    levels = ([baseline] if baseline is not None else []) + [lm[k] for k in sorted(lm)]
    expected = calendar_levels(group, m_start, m_end, allow_prior_quarter_baseline=allow_prior_quarter_baseline)
    if group == 'M':
        wanted = list(range(m_start, m_end + 1))
    else:
        wanted = [m for m in range(m_start, m_end + 1) if is_quarter_end_month(m)]
    missing = [month_label(m) for m in wanted if m not in lm]
    return {'levels': levels, 'group': group, 'expected_calendar_levels': expected, 'n_eligible': len(levels),
            'shrunk': len(levels) < expected, 'missing_periods': missing,
            'native_frequency': NATIVE_FREQUENCY[source],
            'sampling': SAMPLING_RULE if NATIVE_FREQUENCY[source] in ('W', 'D') else 'NATIVE',
            'native_ids': [r.period_id for r in levels]}


def _snapshot_asof(rows: Sequence[NativeLevel], issue_iso: str):
    best = None
    for r in _released(rows, issue_iso):
        if best is None or (r.period_end, r.period_id) > (best.period_end, best.period_id):
            best = r
    return best


def anchor_levels(rows: Sequence[NativeLevel], anchors: Sequence[tuple]) -> dict:
    """Snapshot per anchor: the newest native released AS OF the anchor's own issue date (immutable replay).

    `anchors` = [(label, month_idx)].  A missing snapshot or a repeated native period (carried quarter) makes
    the anchor ineligible; nothing is filled.
    """
    got, reasons = [], []
    for label, mi in anchors:
        try:
            snap = _snapshot_asof(rows, month_end(mi))
        except EligibilityError as exc:
            reasons.append(f'{label}:{exc}')
            continue
        if snap is None:
            reasons.append(f'{label}:NO_RELEASED_NATIVE_AT_ANCHOR')
            continue
        if got and snap.period_id == got[-1][1].period_id:
            reasons.append(f'{label}:QUARTER_CARRY_NOT_FRESH')
            continue
        got.append((label, snap))
    return {'anchors': got, 'missing': reasons}


def aligned_native_pair(rows_a, rows_b, a: str, b: str, m_start: int, m_end: int, issue_iso: str) -> dict:
    """Matched anchors of two sources.  MM: monthly grid; MQ / QQ: QUARTERLY master anchors (quarters released by
    the issue); the monthly / weekly / daily source supplies its snapshot of that quarter-end month."""
    if a == b or a not in SHORT_SOURCES or b not in SHORT_SOURCES:
        raise ValueError('TWO_DISTINCT_REGISTERED_EWI_REQUIRED')
    group = pair_group(a, b)
    la = level_map(rows_a, a, m_start, m_end, issue_iso)
    lb = level_map(rows_b, b, m_start, m_end, issue_iso)
    months = sorted(set(la) & set(lb))
    quarterly = group in ('MQ', 'QQ')
    if quarterly:
        months = [m for m in months if is_quarter_end_month(m)]
        expected = [m for m in range(m_start, m_end + 1) if is_quarter_end_month(m)]
    else:
        expected = list(range(m_start, m_end + 1))
    ids_a, ids_b = [la[m].period_id for m in months], [lb[m].period_id for m in months]
    if len(set(ids_a)) != len(ids_a) or len(set(ids_b)) != len(ids_b):
        raise EligibilityError('CARRIED_NATIVE_OBSERVATION_REUSED')
    return {'group': group, 'master_grid': 'QUARTERLY' if quarterly else 'MONTHLY', 'anchors': [month_label(m) for m in months],
            'matched_levels': [(la[m].value, lb[m].value) for m in months], 'native_ids_a': ids_a,
            'native_ids_b': ids_b, 'n_anchors': len(months), 'expected_anchors': len(expected),
            'shrunk': len(months) < len(expected),
            'provenance': {a: NATIVE_FREQUENCY[a], b: NATIVE_FREQUENCY[b]}}


# ------------------------------------------------------------------------------------------ threshold cards
def _require_text(**fields) -> None:
    for k, v in fields.items():
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f'THRESHOLD_CARD_FIELD_REQUIRED:{k}')


@dataclass(frozen=True)
class ThresholdCard:
    """Individual threshold registry entry (spec G1 input contract) for one source/branch/horizon-or-stage/sign."""
    source: str
    branch: str                       # LONG | SHORT
    horizon_or_interval: str          # H12.. (Long) | S1.. (Short)
    sign: int
    unit: str
    raw_transformation: str
    owner_id: str
    registry_version: str
    theta_hi: object
    theta_lo: object
    min_bases: int
    count_k: int

    def __post_init__(self):
        _require_text(unit=self.unit, raw_transformation=self.raw_transformation, owner_id=self.owner_id,
                      registry_version=self.registry_version)
        if self.source not in SHORT_SOURCES or self.branch not in ('LONG', 'SHORT'):
            raise ValueError('SOURCE_OR_BRANCH_UNREGISTERED')
        if self.branch == 'LONG' and self.source not in LONG_SOURCES:
            raise ValueError('LONG_BRANCH_IS_CG_AND_DSR_P_ONLY')
        if self.sign not in (-1, 1):
            raise ValueError('SIGN_PLUS_OR_MINUS_ONE')

    @property
    def config_id(self) -> str:
        return f'{self.source}|{self.branch}|{self.horizon_or_interval}|{self.sign:+d}|{self.owner_id}|{self.registry_version}'


@dataclass(frozen=True)
class JointCard:
    pair: tuple
    branch: str
    horizon_or_interval: str
    signs: tuple
    unit_a: str
    unit_b: str
    raw_transformation: str
    owner_id: str
    registry_version: str
    high_a: object
    high_b: object
    low_a: object
    low_b: object
    gate1_a: float
    gate1_b: float
    min_bases: int
    count_k: int

    def __post_init__(self):
        _require_text(unit_a=self.unit_a, unit_b=self.unit_b, raw_transformation=self.raw_transformation,
                      owner_id=self.owner_id, registry_version=self.registry_version)
        legal = set(short_pairs()) if self.branch == 'SHORT' else {('CG', 'DSR-P')}
        if self.branch not in ('LONG', 'SHORT') or tuple(self.pair) not in legal:
            raise ValueError('UNREGISTERED_JOINT_PAIR_FOR_BRANCH')
        if len(self.signs) != 2 or any(s not in (-1, 1) for s in self.signs):
            raise ValueError('SIGNED_MODE_PLUS_MINUS_ONE_PAIR')

    @property
    def config_id(self) -> str:
        return (f'{"|".join(self.pair)}|{self.branch}|{self.horizon_or_interval}|{self.signs[0]:+d}{self.signs[1]:+d}'
                f'|{self.owner_id}|{self.registry_version}')


def _check_card(card, *, source=None, pair=None, branch, tag):
    if source is not None and (card.source != source or card.branch != branch or card.horizon_or_interval != tag):
        raise ValueError('THRESHOLD_CARD_DOES_NOT_MATCH_SOURCE_BRANCH_HORIZON')
    if pair is not None and (tuple(card.pair) != tuple(pair) or card.branch != branch or card.horizon_or_interval != tag):
        raise ValueError('JOINT_CARD_DOES_NOT_MATCH_PAIR_BRANCH_HORIZON')


# ------------------------------------------------------------------------------------------ Individual / Joint
def _individual_from_levels(levels: Sequence[NativeLevel], card: ThresholdCard, info: dict) -> dict:
    try:
        if len(levels) < 2:
            raise EligibilityError('requires >= 2 genuine native levels')
        res = individual([lv.value for lv in levels], card.sign, theta_hi=card.theta_hi, theta_lo=card.theta_lo,
                         min_bases=card.min_bases, count_k=card.count_k)
    except EligibilityError as exc:
        return _ne(str(exc), source=card.source, branch=card.branch, n_levels=len(levels), **info)
    terminal = list(res.get('terminal') or [])
    terminal_signed = [card.sign * d for d in terminal]
    return {'status': 'EXECUTED', 'vote': res['vote'], 'source': card.source, 'branch': card.branch,
            'strong': res['strong'], 'weak': res['weak'], 'count': res['count'], 'signed_count': res['signed_count'],
            'n_levels': res['n'], 'count_pairs': res['n'] * (res['n'] - 1) // 2, 'terminal_bases': res['n'] - 1,
            'terminal_mean': (sum(terminal) / len(terminal) if terminal else None),
            'terminal_latest': (terminal[-1] if terminal else None),
            'terminal_mean_signed': (sum(terminal_signed) / len(terminal_signed) if terminal_signed else None),
            'terminal_latest_signed': (terminal_signed[-1] if terminal_signed else None),
            'native_ids': [lv.period_id for lv in levels], 'config_id': card.config_id, **info}


def short_interval_confirm(rows, source: str, stage: str, onset_T, card: ThresholdCard | None, *,
                           allow_prior_quarter_baseline: bool = False) -> dict:
    """Individual Short for stage S1..S4, evaluated at the stage's END horizon issue."""
    if stage not in ENTRY_STAGE_BOUNDS:
        raise ValueError('ENTRY_STAGE_S1_TO_S4_ONLY_S5_IS_RETROSPECTIVE')
    w = stage_window(stage, onset_T)
    base = {'stage': stage, 'issue_date': w['issue_date']}
    if card is None:
        return _ne('THRESHOLD_OWNER_MISSING', source=source, branch='SHORT', **base)
    _check_card(card, source=source, branch='SHORT', tag=stage)
    try:
        sl = stage_levels(rows, source, w['start_idx'], w['end_idx'], w['issue_date'],
                          allow_prior_quarter_baseline=allow_prior_quarter_baseline)
    except EligibilityError as exc:
        return _ne(str(exc), source=source, branch='SHORT', **base)
    info = {**base, 'expected_calendar_levels': sl['expected_calendar_levels'], 'shrunk': sl['shrunk'],
            'missing_periods': sl['missing_periods'], 'native_frequency': sl['native_frequency'],
            'sampling': sl['sampling']}
    return _individual_from_levels(sl['levels'], card, info)


def long_horizon_confirm(rows, source: str, onset_T, horizon: str, card: ThresholdCard | None) -> dict:
    """Individual Long on the anchor snapshots H15..horizon.  H15 -> NOT_APPLICABLE (not a 0)."""
    if source not in LONG_SOURCES:
        raise ValueError('LONG_IS_CG_AND_DSR_P_ONLY')
    if horizon not in ENTRY_HORIZON_MONTHS:
        raise ValueError('ENTRY_HORIZON_UNREGISTERED')
    upto = ENTRY_HORIZONS.index(horizon)
    base = {'horizon': horizon}
    if upto == 0:
        return _na('H15_HAS_NO_CROSS_HORIZON_CONFIRMATION', source=source, branch='LONG', **base)
    if card is None:
        return _ne('THRESHOLD_OWNER_MISSING', source=source, branch='LONG', **base)
    _check_card(card, source=source, branch='LONG', tag=horizon)
    t = month_index(onset_T)
    anchors = [(h, t - ENTRY_HORIZON_MONTHS[h]) for h in ENTRY_HORIZONS[:upto + 1]]
    got = anchor_levels(rows, anchors)
    if got['missing']:
        return _ne('INSUFFICIENT_VERIFIED_NATIVE_ANCHORS', source=source, branch='LONG', missing=got['missing'],
                   n_anchors_expected=len(anchors), n_anchors_eligible=len(got['anchors']), **base)
    return _individual_from_levels([s for _, s in got['anchors']], card,
                                   {**base, 'anchors': [h for h, _ in got['anchors']]})


def _joint_from_matched(matched: dict, card: JointCard, info: dict) -> dict:
    try:
        res = joint(matched['matched_levels'], card.signs, high_a=card.high_a, high_b=card.high_b,
                    low_a=card.low_a, low_b=card.low_b, gate1_a=card.gate1_a, gate1_b=card.gate1_b,
                    min_bases=card.min_bases, count_k=card.count_k, timing_valid=True)
    except EligibilityError as exc:
        return _ne(str(exc), pair=tuple(card.pair), branch=card.branch, n_anchors=matched['n_anchors'], **info)
    va = [float(z[0]) for z in matched['matched_levels']]
    vb = [float(z[1]) for z in matched['matched_levels']]
    ta, tb = va[-1] - va[0], vb[-1] - vb[0]
    return {'status': 'EXECUTED', 'vote': res['vote'], 'pair': tuple(card.pair), 'branch': card.branch,
            'gate1': res['gate1'], 'gate2': res['gate2'], 'paired_count': res['paired_count'],
            'paired_hi': res['paired_hi'], 'paired_lo': res['paired_lo'], 'n_anchors': matched['n_anchors'],
            'terminal_a_latest': float(ta), 'terminal_b_latest': float(tb),
            'terminal_a_signed': float(card.signs[0] * ta), 'terminal_b_signed': float(card.signs[1] * tb),
            'expected_anchors': matched.get('expected_anchors'), 'shrunk': matched.get('shrunk'),
            'native_ids_a': matched['native_ids_a'], 'native_ids_b': matched['native_ids_b'],
            'config_id': card.config_id, **info}


def joint_long_confirm(rows_cg, rows_dsrp, onset_T, horizon: str, card: JointCard | None) -> dict:
    """Joint Long CG x DSR-P on MATCHED anchors (both snapshots exist, both fresh) - not I_CG AND I_DSR-P."""
    if horizon not in ENTRY_HORIZON_MONTHS:
        raise ValueError('ENTRY_HORIZON_UNREGISTERED')
    upto = ENTRY_HORIZONS.index(horizon)
    base = {'horizon': horizon}
    if upto == 0:
        return _na('H15_HAS_NO_CROSS_HORIZON_CONFIRMATION', pair=('CG', 'DSR-P'), branch='LONG', **base)
    if card is None:
        return _ne('THRESHOLD_OWNER_MISSING', pair=('CG', 'DSR-P'), branch='LONG', **base)
    _check_card(card, pair=('CG', 'DSR-P'), branch='LONG', tag=horizon)
    t = month_index(onset_T)
    anchors = [(h, t - ENTRY_HORIZON_MONTHS[h]) for h in ENTRY_HORIZONS[:upto + 1]]
    ga, gb = anchor_levels(rows_cg, anchors), anchor_levels(rows_dsrp, anchors)
    da, db = dict(ga['anchors']), dict(gb['anchors'])
    labels = [h for h, _ in anchors if h in da and h in db]
    matched = {'matched_levels': [(da[h].value, db[h].value) for h in labels], 'n_anchors': len(labels),
               'expected_anchors': len(anchors), 'shrunk': len(labels) < len(anchors),
               'native_ids_a': [da[h].period_id for h in labels], 'native_ids_b': [db[h].period_id for h in labels]}
    return _joint_from_matched(matched, card, {**base, 'anchors': labels,
                                               'unmatched_a': ga['missing'], 'unmatched_b': gb['missing']})


def joint_short_confirm(rows_a, rows_b, pair: tuple, stage: str, onset_T, card: JointCard | None) -> dict:
    """Joint Short for one of the 21 pairs on a stage window with the group-specific master grid."""
    if stage not in ENTRY_STAGE_BOUNDS:
        raise ValueError('ENTRY_STAGE_S1_TO_S4_ONLY_S5_IS_RETROSPECTIVE')
    a, b = pair
    if tuple(pair) not in set(short_pairs()):
        raise ValueError('UNREGISTERED_SHORT_PAIR')
    w = stage_window(stage, onset_T)
    base = {'stage': stage, 'issue_date': w['issue_date']}
    if card is None:
        return _ne('THRESHOLD_OWNER_MISSING', pair=tuple(pair), branch='SHORT', **base)
    _check_card(card, pair=pair, branch='SHORT', tag=stage)
    try:
        matched = aligned_native_pair(rows_a, rows_b, a, b, w['start_idx'], w['end_idx'], w['issue_date'])
    except EligibilityError as exc:
        return _ne(str(exc), pair=tuple(pair), branch='SHORT', **base)
    return _joint_from_matched(matched, card, {**base, 'group': matched['group'], 'master_grid': matched['master_grid'],
                                               'anchors': matched['anchors']})


# ------------------------------------------------------------------------------------------ ENTRY pool
def _any_eligible(votes: Sequence, registered: int):
    """OR of eligible 0/1 branch votes: 1 if any is 1; 0 only if EVERY registered branch is eligible and 0."""
    if any(v == 1 for v in votes):
        return 1
    return 0 if len(votes) == registered else None


def merge_branches(il: Mapping, is_: Mapping, jl, js: Mapping, *, long_applicable: bool = True) -> dict:
    """Branch results -> source-level / pair-level eligible votes for the kernel pool.

    CG and DSR-P appear in both the Long and the Short branch (different windows and owner values): they are
    NOT independent evidence, so they merge into ONE source vote; the pair (CG, DSR-P) likewise merges JL with
    its Short Joint.  A source/pair whose registered branches are not all eligible and none is 1 stays ABSENT
    (unknown) - the kernel then refuses a decision that depends on it.
    """
    individuals, joints, per_source = {}, {}, {}
    for s in SHORT_SOURCES:
        votes, registered = [], 1
        r = is_.get(s)
        if r and r.get('status') == 'EXECUTED':
            votes.append(r['vote'])
        if s in LONG_SOURCES and long_applicable:
            registered += 1
            r = il.get(s)
            if r and r.get('status') == 'EXECUTED':
                votes.append(r['vote'])
        v = _any_eligible(votes, registered)
        per_source[s] = {'registered_branches': registered, 'eligible_votes': list(votes), 'merged': v}
        if v is not None:
            individuals[s] = v
    for p in short_pairs():
        votes, registered = [], 1
        r = js.get(p)
        if r and r.get('status') == 'EXECUTED':
            votes.append(r['vote'])
        if p == ('CG', 'DSR-P') and long_applicable:
            registered += 1
            if jl and jl.get('status') == 'EXECUTED':
                votes.append(jl['vote'])
        v = _any_eligible(votes, registered)
        if v is not None:
            joints[p] = v
    return {'individuals': individuals, 'joints': joints, 'per_source': per_source}


def entry_pool(il: Mapping, is_: Mapping, jl, js: Mapping, role: str, *, long_applicable: bool = True) -> dict:
    """ENTRY Pool_r = G_r(IL, IS, JL, JS): one of the THREE role policies on the pair-indexed kernel pool."""
    if role not in ENTRY_ROLE_POLICY:
        raise ValueError('ENTRY_ROLE_NOT_REGISTERED_ONLY_SENSITIVE_CONFIRMED_COMPLEMENTCONFIRMED')
    merged = merge_branches(il, is_, jl, js, long_applicable=long_applicable)
    branch_agg = {}
    for name, items in (('IL', list(il.values())), ('IS', list(is_.values())),
                        ('JS', list(js.values())), ('JL', [jl] if jl else [])):
        ex = [x['vote'] for x in items if x and x.get('status') == 'EXECUTED']
        branch_agg[name] = 1 if any(v == 1 for v in ex) else (0 if ex and len(ex) == len(items) else None)
    policy = ENTRY_ROLE_POLICY[role]
    try:
        vote = pool(merged['individuals'], merged['joints'], policy, registry=SHORT_SOURCES)
    except EligibilityError as exc:
        return {'status': 'NOT_EVALUATED', 'vote': None, 'reason': str(exc), 'role': role, 'policy': policy,
                'branch_aggregates': branch_agg, 'n_eligible_sources': len(merged['individuals']),
                'n_eligible_pairs': len(merged['joints'])}
    return {'status': 'EXECUTED', 'vote': vote, 'role': role, 'policy': policy, 'branch_aggregates': branch_agg,
            'n_eligible_sources': len(merged['individuals']), 'n_eligible_pairs': len(merged['joints']),
            'member_sources': sorted(merged['individuals']), 'member_pairs': sorted(merged['joints'])}


# ------------------------------------------------------------------------------------------ coverage ledger
@dataclass
class CoverageLedger:
    """N_nominal / N_unique / N_attempted / N_executed with separate denominators (spec H1)."""
    auto_register: bool = True
    nominal: dict = field(default_factory=dict)
    _configs: dict = field(default_factory=dict)
    rows: list = field(default_factory=list)

    def register_nominal(self, module: str, stage: str, sign, n_configs: int) -> None:
        if type(n_configs) is not int or n_configs < 0:
            raise ValueError('NOMINAL_COUNT_NONNEGATIVE_INT')
        self.nominal[(module, stage, str(sign))] = self.nominal.get((module, stage, str(sign)), 0) + n_configs

    def record(self, module: str, stage: str, sign, config_id: str, key, result: Mapping) -> None:
        group = (module, stage, str(sign))
        known = {c for (m, s, g, c) in self._configs if (m, s, g) == group}
        if config_id not in known:
            if group not in self.nominal:
                if not self.auto_register:
                    raise ValueError('CONFIG_GROUP_NOT_REGISTERED_IN_NOMINAL')
                self.nominal[group] = 0
            if len(known) + 1 > self.nominal[group]:
                if not self.auto_register:
                    raise ValueError('CONFIG_COUNT_EXCEEDS_REGISTERED_NOMINAL')
                self.nominal[group] = len(known) + 1
        status = result.get('status')
        if status not in ('EXECUTED', 'NOT_EVALUATED', 'NOT_APPLICABLE'):
            raise ValueError('LEDGER_STATUS_EXECUTED_NOT_EVALUATED_OR_NOT_APPLICABLE')
        vote = result.get('vote')
        if status == 'EXECUTED' and vote not in (0, 1):
            raise ValueError('EXECUTED_ROW_NEEDS_A_BINARY_VOTE')
        if status != 'EXECUTED' and vote is not None:
            raise ValueError('NO_VOTE_WITHOUT_ELIGIBILITY')
        self._configs.setdefault((module, stage, str(sign), config_id), {})[str(key)] = (vote if status == 'EXECUTED' else 'NE')
        self.rows.append({'module': module, 'stage': stage, 'sign': str(sign), 'config_id': config_id, 'key': str(key),
                          'status': status, 'vote': vote, 'reason': result.get('reason')})

    def summary(self) -> dict:
        attempted_cfg = {c for c in self._configs if any(r['status'] != 'NOT_APPLICABLE' for r in self.rows
                                                         if (r['module'], r['stage'], r['sign'], r['config_id']) == c)}
        executed_cfg = {c for c in self._configs if any(v != 'NE' for v in self._configs[c].values())}
        masks = {hashlib.sha256(json.dumps(sorted(self._configs[c].items()), default=str).encode()).hexdigest()
                 for c in executed_cfg}
        by_status = Counter(r['status'] for r in self.rows)
        reasons = Counter(r['reason'] for r in self.rows if r['status'] == 'NOT_EVALUATED')
        n_attempt_keys = by_status['EXECUTED'] + by_status['NOT_EVALUATED']
        return {'N_nominal': sum(self.nominal.values()), 'N_unique_decision_masks': len(masks),
                'N_attempted': len(attempted_cfg), 'N_executed': len(executed_cfg),
                'n_key_executed': by_status['EXECUTED'], 'n_key_not_evaluated': by_status['NOT_EVALUATED'],
                'n_key_not_applicable': by_status['NOT_APPLICABLE'],
                'key_coverage': by_status['EXECUTED'] / n_attempt_keys if n_attempt_keys else None,
                'not_evaluated_reasons': dict(reasons),
                'config_coverage': len(executed_cfg) / len(attempted_cfg) if attempted_cfg else None}


def entry_issue(country: str, onset_T, horizon: str, native_rows: Mapping, owners: Mapping, *,
                roles: Sequence[str] = tuple(ENTRY_ROLE_POLICY), ledger: CoverageLedger | None = None,
                allow_prior_quarter_baseline: bool = False) -> dict:
    """One RQ02 ENTRY issue (country, T, horizon): IL, IS, JL, JS(21) -> three role pools.

    `native_rows[source]` = NativeLevel rows of that EWI for this country.  `owners` =
    {'IL': {source: ThresholdCard}, 'IS': {source: ThresholdCard}, 'JL': JointCard,
     'JS': {pair: JointCard}}; a missing card -> NOT_EVALUATED(THRESHOLD_OWNER_MISSING), never a 0.
    """
    if country not in EA7:
        raise ValueError('EA7_ACTIVE_COUNTRY_REQUIRED')
    if horizon not in ENTRY_HORIZON_MONTHS:
        raise ValueError('ENTRY_HORIZON_UNREGISTERED')
    key = (country, month_label(month_index(onset_T)), horizon)
    if horizon == 'H15':
        return {'status': 'INITIALIZATION_REQUIRED', 'reason': 'H15_HAS_NO_COMPLETED_STAGE_OR_CROSS_HORIZON_CONFIRMATION',
                'key': key, 'pools': {}}
    stage = STAGE_ENDING_AT[horizon]
    rows_of = lambda s: native_rows.get(s, ())
    il = {s: long_horizon_confirm(rows_of(s), s, onset_T, horizon, owners.get('IL', {}).get(s)) for s in LONG_SOURCES}
    is_ = {s: short_interval_confirm(rows_of(s), s, stage, onset_T, owners.get('IS', {}).get(s),
                                     allow_prior_quarter_baseline=allow_prior_quarter_baseline) for s in SHORT_SOURCES}
    jl = joint_long_confirm(rows_of('CG'), rows_of('DSR-P'), onset_T, horizon, owners.get('JL'))
    js = {p: joint_short_confirm(rows_of(p[0]), rows_of(p[1]), p, stage, onset_T, owners.get('JS', {}).get(p))
          for p in short_pairs()}
    pools = {r: entry_pool(il, is_, jl, js, r) for r in roles}
    if ledger is not None:
        for s, r in il.items():
            card = owners.get('IL', {}).get(s)
            ledger.record('IL', horizon, card.sign if card else '?', card.config_id if card else f'{s}|LONG|MISSING', key, r)
        for s, r in is_.items():
            card = owners.get('IS', {}).get(s)
            ledger.record('IS', stage, card.sign if card else '?', card.config_id if card else f'{s}|SHORT|MISSING', key, r)
        card = owners.get('JL')
        ledger.record('JL', horizon, ''.join(f'{x:+d}' for x in card.signs) if card else '?',
                      card.config_id if card else 'CG|DSR-P|LONG|MISSING', key, jl)
        for p, r in js.items():
            card = owners.get('JS', {}).get(p)
            ledger.record('JS', stage, ''.join(f'{x:+d}' for x in card.signs) if card else '?',
                          card.config_id if card else '|'.join(p) + '|SHORT|MISSING', key, r)
        for role, r in pools.items():
            ledger.record('POOL', horizon, role, 'POOL|' + role, key, r)
    return {'status': 'EVALUATED', 'key': key, 'stage': stage, 'IL': il, 'IS': is_, 'JL': jl, 'JS': js, 'pools': pools}


# ------------------------------------------------------------------------------------------ EXIT
def build_exit_anchors(first_n_T) -> list:
    """T-18/-12/-6/-3 with cumulative confirmations 0/1/2/3, Count 0/1/3/6, Terminal 0/1/2/3 (retrospective)."""
    t = month_index(first_n_T)
    rows = []
    for i, m in enumerate(EXIT_ANCHOR_MONTHS):
        n = i + 1
        pairs, terminal = geometry(n) if n >= 2 else (0, 0)
        rows.append({'anchor': f'T-{m}', 'months_before_T': m, 'slot': EXIT_SLOT_NAMES[m], 'issue_month': month_label(t - m),
                     'issue_date': month_end(t - m), 'long_anchors_known': n, 'long_confirmations': i,
                     'long_count_pairs': pairs, 'long_terminal_bases': terminal,
                     'cross_anchor_confirmation': 'NOT_APPLICABLE' if i == 0 else 'APPLICABLE',
                     'scope': 'RETROSPECTIVE_LABEL_JOIN_ONLY__T_IS_NOT_A_FEATURE'})
    return rows


def exit_stage_status(horizon_months: int) -> dict:
    """Which Short stages are complete at a forecast issue h months before the target T.  There is no S5."""
    if horizon_months not in EXIT_HORIZONS:
        raise ValueError('EXIT_HORIZON_NOT_IN_18_12_6_3')
    out = {}
    for stage, (start, end) in EXIT_STAGE_BOUNDS.items():
        if horizon_months <= end:
            out[stage] = 'COMPLETE'
        elif horizon_months < start:
            out[stage] = 'IN_PROGRESS_PARTIAL'
        else:
            out[stage] = 'NOT_STARTED'
    return out


def exit_long_confirm(rows, source: str, first_n_T, issue_month, card: ThresholdCard | None) -> dict:
    """EXIT Long on the anchors T-18/-12/-6/-3 that already exist at the issue."""
    if source not in LONG_SOURCES:
        raise ValueError('LONG_IS_CG_AND_DSR_P_ONLY')
    t, u = month_index(first_n_T), month_index(issue_month)
    anchors = [(f'T-{m}', t - m) for m in EXIT_ANCHOR_MONTHS if t - m <= u]
    if len(anchors) < 2:
        return _na('T18_HAS_NO_CROSS_ANCHOR_CONFIRMATION', source=source, branch='LONG', n_anchors=len(anchors))
    if card is None:
        return _ne('THRESHOLD_OWNER_MISSING', source=source, branch='LONG')
    _check_card(card, source=source, branch='LONG', tag='EXIT')
    got = anchor_levels(rows, anchors)
    if got['missing']:
        return _ne('INSUFFICIENT_VERIFIED_NATIVE_ANCHORS', source=source, branch='LONG', missing=got['missing'],
                   n_anchors_expected=len(anchors), n_anchors_eligible=len(got['anchors']))
    return _individual_from_levels([s for _, s in got['anchors']], card, {'anchors': [a for a, _ in got['anchors']]})


def exit_short_confirm(rows, source: str, stage: str, first_n_T, issue_month, card: ThresholdCard | None) -> dict:
    """EXIT Short for S1..S4 (no S5).  A stage still running at the issue only uses levels released by then."""
    if stage not in EXIT_STAGE_BOUNDS:
        raise ValueError('EXIT_STAGE_S1_TO_S4_ONLY')
    b0, b1 = EXIT_STAGE_BOUNDS[stage]
    t, u = month_index(first_n_T), month_index(issue_month)
    m_start, m_end = t - b0, t - b1
    if u < m_start:
        return _na('STAGE_NOT_STARTED_AT_ISSUE', source=source, branch='SHORT', stage=stage)
    if card is None:
        return _ne('THRESHOLD_OWNER_MISSING', source=source, branch='SHORT', stage=stage)
    _check_card(card, source=source, branch='SHORT', tag=stage)
    partial = u < m_end
    eff_end = min(m_end, u)
    try:
        sl = stage_levels(rows, source, m_start, eff_end, month_end(u))
    except EligibilityError as exc:
        return _ne(str(exc), source=source, branch='SHORT', stage=stage, partial=partial)
    info = {'stage': stage, 'partial': partial, 'expected_calendar_levels': sl['expected_calendar_levels'],
            'shrunk': sl['shrunk'], 'missing_periods': sl['missing_periods'], 'native_frequency': sl['native_frequency'],
            'sampling': sl['sampling']}
    return _individual_from_levels(sl['levels'], card, info)


def exit_joint_confirm(rows_a, rows_b, pair: tuple, stage: str, first_n_T, issue_month, card: JointCard | None) -> dict:
    """EXIT Joint Short (21 pairs, same group-specific master grids as ENTRY) for S1..S4."""
    if stage not in EXIT_STAGE_BOUNDS:
        raise ValueError('EXIT_STAGE_S1_TO_S4_ONLY')
    if tuple(pair) not in set(short_pairs()):
        raise ValueError('UNREGISTERED_SHORT_PAIR')
    b0, b1 = EXIT_STAGE_BOUNDS[stage]
    t, u = month_index(first_n_T), month_index(issue_month)
    m_start, m_end = t - b0, t - b1
    if u < m_start:
        return _na('STAGE_NOT_STARTED_AT_ISSUE', pair=tuple(pair), branch='SHORT', stage=stage)
    if card is None:
        return _ne('THRESHOLD_OWNER_MISSING', pair=tuple(pair), branch='SHORT', stage=stage)
    _check_card(card, pair=pair, branch='SHORT', tag=stage)
    try:
        matched = aligned_native_pair(rows_a, rows_b, pair[0], pair[1], m_start, min(m_end, u), month_end(u))
    except EligibilityError as exc:
        return _ne(str(exc), pair=tuple(pair), branch='SHORT', stage=stage)
    return _joint_from_matched(matched, card, {'stage': stage, 'partial': u < m_end, 'group': matched['group'],
                                               'master_grid': matched['master_grid'], 'anchors': matched['anchors']})


def exit_pool_A(family_votes: Mapping, k: int, *, k_owner_id: str | None = None) -> dict:
    """EXIT-A = 1[IL + IS + JL + JS >= K]; a family without eligible evidence is MISSING, never a 0.

    K = 1 (ANY) and K = 4 (ALL) are structural; K = 2 or 3 needs an owner id.  Raises EligibilityError when the
    outcome cannot be derived from the eligible families (-> NOT_EVALUATED in the ledger).
    """
    if set(family_votes) - set(EXIT_FAMILIES):
        raise ValueError('EXIT_FAMILY_UNREGISTERED')
    if type(k) is not int or not 1 <= k <= 4:
        raise ValueError('K_OUTSIDE_1_TO_4')
    if k in (2, 3) and not k_owner_id:
        raise EligibilityError('K2_K3_THRESHOLD_OWNER_REQUIRED')
    votes = {f: family_votes.get(f) for f in EXIT_FAMILIES}
    if any(v not in (0, 1, None) for v in votes.values()):
        raise EligibilityError('family votes must be binary or missing')
    lower = sum(v == 1 for v in votes.values())
    upper = lower + sum(v is None for v in votes.values())
    if lower >= k:
        vote = 1
    elif upper < k:
        vote = 0
    else:
        raise EligibilityError('EXIT_A_K_OF_4_NOT_DERIVABLE_FROM_ELIGIBLE_FAMILIES')
    return {'vote': vote, 'K': k, 'confirmed_min': lower, 'possible_max': upper, 'unresolved': upper - lower}


def exit_nominal_budget(registry: Mapping | None = None) -> dict:
    """EXIT has NO official total until its own candidate counts per interval / mode / policy are registered.

    The ENTRY figure 9,175,534 (S1..S5, VaR/ES S5) is never reused.  `registry` = {stage: {'K_long','K_monthly',
    'K_quarterly'}, 'gate2_variants', 'pool_bundles', 'n_policies'} for S1..S4 only.
    """
    if registry is None:
        raise EligibilityError('EXIT_BUDGET_NOT_REGISTERED: do not copy the ENTRY 9,175,534')
    stages = {s: v for s, v in registry.items() if s in EXIT_STAGE_BOUNDS or str(s).startswith('S')}
    if 'S5' in registry:
        raise ValueError('EXIT_HAS_NO_S5')
    if set(stages) != set(EXIT_STAGE_BOUNDS):
        raise EligibilityError('EXIT_BUDGET_NEEDS_S1_TO_S4_REGISTRATION')
    g2, bundles, pols = (registry.get(k) for k in ('gate2_variants', 'pool_bundles', 'n_policies'))
    if any(type(x) is not int or x < 1 for x in (g2, bundles, pols)):
        raise EligibilityError('EXIT_BUDGET_NEEDS_GATE2_POOL_AND_POLICY_COUNTS')
    per = {s: stage_budget(v['K_long'], v['K_monthly'], v['K_quarterly'], gate2_variants=g2, pool_bundles=bundles,
                           n_policies=pols) for s, v in stages.items()}
    return {'stages': per, 'total': sum(x['total'] for x in per.values()), 'includes_S5': False,
            'copied_from_entry': False, 'status': 'EXIT_OWN_REGISTRY_NOMINAL_NOT_RUN'}
