"""RQ02 end-to-end run contract: calendar, label maturity, OOS and expanding gates (DEC-017).

* OOS access goes through the unchanged PC-002 gate `w01_contract.enforce_execution` with the DEC-015 token.
* Supervised expanding refit is authorized for RQ02 research ONLY (user directive 2026-10-25, DEC-017).  The
  frozen PC-002 gate keeps blocking expanding for every other caller; this module adds a separate RQ02 gate
  whose default is fail-closed and whose only key is the exact DEC-017 token.
* A supervised row (issue i, horizon h) is mature at u iff i + h + LABEL_RELEASE_LAG <= u and every target
  month is ESRB-scorable (DEV, or 2020-01..2021-04).  After 2021-04 no new label exists; nothing is ever
  pseudo-labelled from a model's own forecast.
"""
from __future__ import annotations

from .config import EA7
from .labels_next import (ESRB_PUBLICATION_LAG_MONTHS, ESRB_SCORABLE_END,
                          ESRB_SCORABLE_START, label_status, STATUS_SCORABLE)
from .w01_contract import (ContractBlockedError, OOS_AUTHORIZATION_TOKEN,
                           assert_ea7_only, enforce_execution)

RQ02_E2E_VERSION = "RQ02_E2E_DEC017_V1"
EXPANDING_AUTHORIZATION_TOKEN = "USER_AUTHORIZED_RQ02_EXPANDING_20261025_DEC017"
DEV_FIRST, DEV_LAST = "2005-01", "2019-12"
DEV_FIT_FIRST = "2005-07"
OOS_FIRST, OOS_LAST = "2020-01", "2024-12"
LABEL_RELEASE_LAG = ESRB_PUBLICATION_LAG_MONTHS
ENTRY_HORIZONS = ("H15", "H12", "H10", "H8", "H6")
ENTRY_MONTHS = {"H15": 45, "H12": 36, "H10": 30, "H8": 24, "H6": 18}
EXIT_HORIZONS = (18, 12, 6, 3)
POST_EXPOSURE = "POST_OOS_EXPOSURE_RESEARCH"


def mi(month: str) -> int:
    return int(month[:4]) * 12 + int(month[5:7]) - 1


def ml(idx: int) -> str:
    return f"{idx // 12:04d}-{idx % 12 + 1:02d}"


def add_m(month: str, k: int) -> str:
    return ml(mi(month) + k)


def period_of(month: str) -> str:
    if DEV_FIRST <= month <= DEV_LAST:
        return "DEV"
    if OOS_FIRST <= month <= OOS_LAST:
        return "OOS"
    return "OUTSIDE"


def oos_view(month: str) -> str:
    """OOS reporting slice of one chronology (not a research period)."""
    if not OOS_FIRST <= month <= OOS_LAST:
        return "NOT_OOS"
    return "OOS_ESRB_SCORABLE" if month <= ESRB_SCORABLE_END else "OOS_MODEL_ONLY_EXTRA"


def window_scorable(first: str, last: str) -> bool:
    """Every month of [first, last] carries verified-scope ESRB gold."""
    m = first
    while m <= last:
        if label_status(m) != STATUS_SCORABLE:
            return False
        m = add_m(m, 1)
    return True


def mature_at(issue: str, horizon_months: int, asof: str) -> bool:
    """Supervised (x, y) of (issue, horizon) may enter a fit made at `asof`."""
    target = add_m(issue, horizon_months)
    return (mi(target) + LABEL_RELEASE_LAG <= mi(asof)
            and window_scorable(add_m(issue, 1), target))


def require_oos(oos_authorization) -> str:
    return enforce_execution(oos_requested=True, oos_authorization=oos_authorization)


def require_expanding(expanding_authorization, oos_authorization) -> str:
    """RQ02-only supervised expanding gate.  Needs BOTH the OOS token and the DEC-017 token."""
    require_oos(oos_authorization)
    if expanding_authorization != EXPANDING_AUTHORIZATION_TOKEN:
        raise ContractBlockedError("BLOCKED_BY_CONTRACT: RQ02 supervised expanding not authorized (DEC-017 token)")
    return "PASS_RQ02_EXPANDING_AUTHORIZED_USER_DEC017"


def countries() -> tuple:
    return assert_ea7_only(EA7)


__all__ = ["RQ02_E2E_VERSION", "EXPANDING_AUTHORIZATION_TOKEN", "OOS_AUTHORIZATION_TOKEN",
           "DEV_FIRST", "DEV_LAST", "DEV_FIT_FIRST", "OOS_FIRST", "OOS_LAST", "ENTRY_HORIZONS",
           "ENTRY_MONTHS", "EXIT_HORIZONS", "ESRB_SCORABLE_START", "ESRB_SCORABLE_END", "POST_EXPOSURE",
           "mi", "ml", "add_m", "period_of", "oos_view", "window_scorable", "mature_at", "require_oos",
           "require_expanding", "countries"]
