"""W05 RQ03 scope, vocabulary, validators and the structured result object.

Six assets (E1,E2,E3 observable equity indices; B1,B2 CONSTRUCTED synthetic bond
indices; C1 cash accrual proxy), five risky assets, four issued regimes, three
tail confidences. Owner-free numbers (risky shares, caps, costs, cash rate,
utility, gamma) are never defaulted: a missing owner is BLOCKED, not a value.
Falsifiers: weights off the simplex, asymmetric/non-PSD covariance, a return
dated at or before its issue, an OOS date, an owner record without hash.
"""
from __future__ import annotations
import hashlib
import json
from datetime import date, datetime, timezone
from typing import Any

import numpy as np

from . import CALC_VERSION

ASSETS = ('E1', 'E2', 'E3', 'B1', 'B2', 'C1')
RISKY = ASSETS[:5]
EQUITY = ASSETS[:3]
BONDS = ('B1', 'B2')
REGIMES = ('N', 'D', 'G', 'BOTH')
STRESS = ('D', 'G', 'BOTH')
ALPHAS = (0.95, 0.975, 0.99)
METHODS = ('EW', 'IVOL', 'MVP', 'ERC', 'MaxDiv')
STATUSES = ('PASS', 'FAIL', 'U', 'BLOCKED')
DEV_START, DEV_END = date(2005, 1, 1), date(2019, 12, 31)
RL_STATUS = 'DORMANT'
CONFLICTS = {
    'C04_EW': 'EW_RISKY_SLEEVE (1/5 of risky sleeve) and EW_6_ASSET (1/6 incl. cash) are different benchmarks',
    'C05_CAP_UNITS': 'Normal 50%-75% / Stress 50%: capital share, risk budget or ES tolerance is owner-undefined',
    'SYNTHETIC_BONDS': 'B1/B2 are constructed spot-yield proxies, not tradable NAVs',
    'CASH_OWNER': 'C1 Euribor-3M accrual proxy lacks an approved funding/cash owner',
}


# Hard research envelope (spec RQ03 risk tolerance, section 1).  The 50-75% / <=50% bands are RISKY-CAPITAL
# SHARES a(w) = sum of the five risky weights, NOT VaR/ES numbers; the 60% cap binds each risky asset only
# (C1 is the residual 1 - a and is never capped).
#   K_N = {w >= 0, sum w = 1, 0.50 <= a <= 0.75, w_i <= 0.60}      K_S = {..., 0 <= a <= 0.50, ...}
HARD_ENVELOPE = {'asset_cap': 0.60, 'normal_band': (0.50, 0.75), 'stress_max': 0.50}
RISK_UNIT = 'DECIMAL_LOSS_FRACTION_OF_NAV'      # losses are on OPENING NAV, after the fee
RISK_BASIS = 'NET_AFTER_COST'
CAPITAL_SHARE_UNITS = ('RISKY_CAPITAL_FRACTION', 'RISK_BUDGET_FRACTION', 'ES_TOLERANCE')
ASSUMED_RESEARCH_ONLY = 'ASSUMED_RESEARCH_ONLY'   # label for 2% ES95 / 10 bps / lambda .94 / 50% research numbers
MAIN_IDS = ('MAIN', 'POLICY')                     # POLICY is the report label of MAIN; neither is a benchmark


class Blocked(ValueError):
    """Legal owner/readiness veto (maps to status BLOCKED)."""


def canonical_sha256(x: Any) -> str:
    return hashlib.sha256(json.dumps(x, sort_keys=True, default=str, separators=(',', ':'),
                                     ensure_ascii=True).encode()).hexdigest()


def result(status: str, reason: str | None = None, *, source_ids=(), issue_times=(), **payload) -> dict:
    if status not in STATUSES:
        raise ValueError('UNREGISTERED_STATUS:' + str(status))
    if status != 'PASS' and not reason:
        raise ValueError('NON_PASS_REQUIRES_REASON')
    return {'status': status, 'reason': reason, 'source_ids': list(source_ids),
            'issue_times': [str(t) for t in issue_times], 'calc_version': CALC_VERSION, **payload}


def owner_ok(record: Any) -> bool:
    """An owner record counts only when explicitly approved AND hash-identified."""
    return isinstance(record, dict) and record.get('approved') is True and bool(record.get('owner_hash'))


def simplex(w, n: int | None = None, *, tol: float = 1e-9) -> np.ndarray:
    x = np.asarray(w, dtype=float)
    if x.ndim != 1 or (n is not None and len(x) != n) or not np.isfinite(x).all():
        raise ValueError('INVALID_WEIGHT_VECTOR')
    if x.min() < -tol or abs(x.sum() - 1) > 1e-8:
        raise ValueError('WEIGHTS_OFF_SIMPLEX')
    return np.clip(x, 0, None) / np.clip(x, 0, None).sum()


def covariance(s, n: int | None = None, *, positive: bool = False) -> np.ndarray:
    m = np.asarray(s, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1] or (n is not None and m.shape != (n, n)) or not np.isfinite(m).all():
        raise ValueError('INVALID_COVARIANCE')
    scale = max(float(np.abs(m).max()), 1e-300)
    if np.abs(m - m.T).max() > 1e-10 * scale:
        raise ValueError('ASYMMETRIC_COVARIANCE')
    lam = np.linalg.eigvalsh((m + m.T) / 2)
    if lam.min() < -1e-10 * scale or (positive and lam.min() <= 1e-14 * scale):
        raise ValueError('NON_PSD_OR_DEGENERATE_COVARIANCE')
    return (m + m.T) / 2


def stamp(value) -> datetime:
    """Timezone-aware UTC timestamp; date-only means end of that day (UTC)."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError('TIMEZONE_AWARE_TIMESTAMP_REQUIRED')
        return value.astimezone(timezone.utc)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, 23, 59, 59, 999999, tzinfo=timezone.utc)
    text = str(value)
    if len(text) == 10:
        return stamp(date.fromisoformat(text))
    return stamp(datetime.fromisoformat(text.replace('Z', '+00:00')))


def require_dev(d) -> date:
    x = d if isinstance(d, date) and not isinstance(d, datetime) else stamp(d).date()
    if x > DEV_END:
        raise Blocked('OOS_SEALED:' + x.isoformat())
    return x
