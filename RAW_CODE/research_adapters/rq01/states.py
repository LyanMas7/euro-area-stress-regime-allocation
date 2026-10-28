"""RQ01-06 / B03: UP / DOWN / SIDEWAY(+UP_DRIFT, DOWN_DRIFT, FLAT) / OTHER, or NOT_EVALUATED.

SIDEWAY is an equilibrium-bounded path state, not a zero endpoint delta. With n
levels at month offsets t_i, OLS slope b and fitted path f, SIDEWAY holds iff
  n >= n_min (>=3), |b| <= slope_max, range <= range_max, rmse(x-f) <= rmse_max,
  |mean - eq| <= eq_band, #{|x_i - eq| <= eq_band} >= ceil(band_fraction * n),
  |x_last - eq| <= terminal_band.
Subtype: UP_DRIFT if b > flat_slope, DOWN_DRIFT if b < -flat_slope, else FLAT (disjoint).
Otherwise UP iff net > up_net and b > slope_max; DOWN iff net < -down_net and b < -slope_max;
else OTHER. The three states are exclusive by order.
GLOBAL NO-U: the path state is a typed Direction ('UP', 'DOWN', 'SIDEWAY') or 'OTHER'; the legacy
motif letter ('U' = UP, 'D' = DOWN, 'S' = SIDEWAY) is reported separately as `motif_letter` and is a
direction code, never an unknown.  Missing levels or < 2 levels -> record status NOT_EVALUATED with no
state at all (eligibility, not a vote and not a direction).
Per-comparison predicates for Joint Count use the same owner (motif letters):
  U(=UP): delta >= up_step;  D(=DOWN): -delta >= down_step;  S(=SIDEWAY): |delta| <= side_step.
A non-finite comparison is a non-evaluable component (NonVote), never FAIL.
"""
from __future__ import annotations
import math
from dataclasses import dataclass

import numpy as np

from ..vote_contract import Direction, legacy_letter, not_evaluated
from .common import reject_retired_owner, result

# Legacy motif direction letters (U = UP, D = DOWN, S = SIDEWAY): direction codes, not vote states.
MOTIF_LETTERS = ('U', 'D', 'S')
STATES = MOTIF_LETTERS          # retained import name; identical object, documented as direction letters
PATH_STATES = (Direction.UP.value, Direction.DOWN.value, Direction.SIDEWAY.value, 'OTHER')
SUBTYPES = ('UP_DRIFT', 'DOWN_DRIFT', 'FLAT')


@dataclass(frozen=True)
class StateOwner:
    source: str
    owner_hash: str
    equilibrium: float
    n_min: int
    slope_max: float
    flat_slope: float
    range_max: float
    rmse_max: float
    equilibrium_band: float
    terminal_band: float
    band_fraction: float
    up_net: float
    down_net: float
    up_step: float
    down_step: float
    side_step: float

    def __post_init__(self):
        reject_retired_owner(self.owner_hash)
        if not self.owner_hash:
            raise ValueError('STATE_OWNER_HASH_REQUIRED')
        if type(self.n_min) is not int or self.n_min < 3:
            raise ValueError('SIDEWAY_NATIVE_MIN_GE_3')
        vals = [self.slope_max, self.flat_slope, self.range_max, self.rmse_max, self.equilibrium_band,
                self.terminal_band, self.band_fraction, self.up_net, self.down_net, self.up_step,
                self.down_step, self.side_step]
        if any(not math.isfinite(v) or v < 0 for v in vals) or not math.isfinite(self.equilibrium):
            raise ValueError('NONNEGATIVE_FINITE_OWNER_DOMAIN')
        if self.flat_slope > self.slope_max or self.band_fraction > 1:
            raise ValueError('OWNER_DOMAIN_ORDER')
        if self.up_step <= 0 or self.down_step <= 0:
            raise ValueError('DIRECTIONAL_STEP_MUST_BE_POSITIVE')
        if self.side_step >= min(self.up_step, self.down_step):
            raise ValueError('SIDE_STEP_MUST_BE_BELOW_DIRECTIONAL_STEPS_FOR_EXCLUSIVE_STATES')


def classify_path(levels: list, owner: StateOwner, times: list | None = None) -> dict:
    x = list(levels)
    if len(x) < 2 or any(v is None or not math.isfinite(v) for v in x):
        return result('NOT_EVALUATED', 'INSUFFICIENT_OR_MISSING_NATIVE_LEVELS', source_ids=[owner.source],
                      n=len(x))
    t = np.asarray(times if times is not None else range(len(x)), float)
    xa = np.asarray(x, float)
    tc, xc = t - t.mean(), xa - xa.mean()
    slope = float(tc @ xc / (tc @ tc))
    fit = xa.mean() + slope * tc
    rmse = float(np.sqrt(np.mean((xa - fit) ** 2)))
    rng = float(xa.max() - xa.min())
    net = x[-1] - x[0]
    band = int((np.abs(xa - owner.equilibrium) <= owner.equilibrium_band).sum())
    terminal_ok = abs(x[-1] - owner.equilibrium) <= owner.terminal_band
    side = (len(x) >= owner.n_min and abs(slope) <= owner.slope_max and rng <= owner.range_max
            and rmse <= owner.rmse_max and abs(xa.mean() - owner.equilibrium) <= owner.equilibrium_band
            and band >= math.ceil(owner.band_fraction * len(x)) and terminal_ok)
    subtype = None
    if side:
        state = Direction.SIDEWAY.value
        subtype = 'UP_DRIFT' if slope > owner.flat_slope else 'DOWN_DRIFT' if slope < -owner.flat_slope else 'FLAT'
    elif net > owner.up_net and slope > owner.slope_max:
        state = Direction.UP.value
    elif net < -owner.down_net and slope < -owner.slope_max:
        state = Direction.DOWN.value
    else:
        state = 'OTHER'
    letter = legacy_letter(state) if state != 'OTHER' else None
    return result('PASS', source_ids=[owner.source], state=state, motif_letter=letter, subtype=subtype,
                  slope=slope, rmse=rmse, range=rng, net=net, band_count=band, terminal_equilibrium=terminal_ok,
                  n=len(x))


def step_predicate(delta, state: str, owner: StateOwner):
    """Binary predicate of one native comparison for a motif direction letter (U=UP, D=DOWN, S=SIDEWAY).

    A missing / non-finite comparison is a non-evaluable component (NonVote), never FAIL."""
    if state not in MOTIF_LETTERS:
        raise ValueError('MOTIF_STATE_U_D_S')
    if delta is None or not math.isfinite(delta):
        return not_evaluated('NATIVE_COMPARISON_NOT_COMPUTABLE')
    if state == 'U':
        ok = delta >= owner.up_step
    elif state == 'D':
        ok = -delta >= owner.down_step
    else:
        ok = abs(delta) <= owner.side_step
    return 'PASS' if ok else 'FAIL'
