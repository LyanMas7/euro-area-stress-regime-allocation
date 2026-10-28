"""Corrected labels, coverage status and point-in-time clocks (P0-01, P0-04, P0-06).

Defects repaired here
---------------------
P0-01  The baseline ``labels.label_month`` returns ``"N"`` for every month that
       is not inside an active episode.  Applied to 2020-2024 that silently
       manufactures 308 ''normal'' OOS labels whose ESRB gold does not exist.
       ``label_status`` now distinguishes SCORABLE / NOT_SCORABLE /
       PENDING_MATURITY / UNAVAILABLE and the model layer never scores the
       NOT_SCORABLE rows.

P0-04  ``first-N`` was defined as ''the month after the NormalDate month'',
       one month later than the episode contract ``[StartDate, NormalDate)``
       which already makes the NormalDate month normal.  ``first_normal_month``
       is the single canonical endpoint (inclusive) used by every caller.

P0-06  Month-to-week joins must be strict as-of.  ``release_week`` maps a
       monthly value to the first week whose close is at least the release
       date, so a decision made at week t cannot see a month published later.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from rev363.config import EA7, CLASSES
from rev363.data import add_months, month_end
from rev363.labels import load_episodes

# ---------------------------------------------------------------------------
# coverage policy
# ---------------------------------------------------------------------------
# User presentation boundary: ESRB benchark rows are only scored on
# 2020-01..2021-04.  The true ESRB publication end date is UNVERIFIED; this is
# an explicit POLICY_OF_PRESENTATION, not evidence.
ESRB_SCORABLE_START = "2020-01"
ESRB_SCORABLE_END = "2021-04"
DEV_START = "2005-01"
DEV_END = "2019-12"

# Research assumption: an ex-post ESRB chronology month becomes part of the
# supervised training set only this many months after the month itself.  The
# value is a documented research lag, never an official issuer stamp.
ESRB_PUBLICATION_LAG_MONTHS = 3

STATUS_SCORABLE = "SCORABLE"
STATUS_NOT_SCORABLE = "NOT_SCORABLE"
STATUS_PENDING = "PENDING_MATURITY"
STATUS_UNAVAILABLE = "UNAVAILABLE"


def _mi(m: str) -> int:
    return int(m[:4]) * 12 + int(m[5:7]) - 1


def label_status(month: str) -> str:
    """Coverage status of an ESRB gold row for ``month`` (YYYY-MM)."""
    if DEV_START <= month <= DEV_END:
        # DEV research convention: ``N`` outside active qualified episodes is
        # the accepted DEV contract (CONTRACT_AND_SCOPE row LABEL).
        return STATUS_SCORABLE
    if ESRB_SCORABLE_START <= month <= ESRB_SCORABLE_END:
        return STATUS_SCORABLE
    if month > ESRB_SCORABLE_END:
        return STATUS_NOT_SCORABLE
    return STATUS_UNAVAILABLE


def gold_label(episodes, country: str, month: str) -> str:
    """Ex-post ESRB chronology label; the NormalDate month is already N.

    Half-open interval ``[StartDate, NormalDate)`` -> origin; else ``N``.
    """
    d = month_end(int(month[:4]), int(month[5:7]))
    for e in episodes:
        if e["country"] != country or e["systemic"] != 1:
            continue
        if e["start"] <= d < e["normal"]:
            return e["origin"] if e["origin"] in CLASSES else None
    return "N"


def eligible_gold(episodes, country: str, month: str):
    """Return actual ex-post gold or None, never manufacture N outside scope.

    Gold is for retrospective scoring only; neither returned value nor the
    research-only maturity lag authenticates an original ESRB issuer stamp.
    """
    if label_status(month) != STATUS_SCORABLE:
        return None
    return gold_label(episodes, country, month)


def scoring_mask(df: pd.DataFrame) -> pd.Series:
    """Use both label coverage and genuine observed gold as hard mask."""
    if not {"label_status", "label"}.issubset(df.columns):
        raise ValueError("REQUIRES_EXPLICIT_GOLD_STATUS_AND_LABEL")
    return (df["label_status"].eq(STATUS_SCORABLE)
            & df["label"].isin(CLASSES)).fillna(False)


def issue_gold_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Separate issued prediction metadata from retrospective scoring gold."""
    f = df.copy()
    good = scoring_mask(f)
    f.loc[~good, "label"] = None
    f["scorable"] = good.astype(bool)
    f["gold_provenance"] = "ESRB_EX_POST_NOT_ISSUED_FEATURE"
    f["gold_release_verified"] = False
    return f


def label_release_month(month: str) -> str:
    """First month at which the ex-post gold for ``month`` is treatable as known."""
    return add_months(month, ESRB_PUBLICATION_LAG_MONTHS)


def first_normal_month(episode: dict) -> str:
    """Canonical first-N month (P0-04): the month containing NormalDate.

    ``[StartDate, NormalDate)`` already excludes the NormalDate month from the
    episode, therefore the NormalDate month *is* the first normal month.  The
    baseline added one extra month; that off-by-one is repaired here.
    """
    d = episode["normal"]
    return f"{d.year:04d}-{d.month:02d}"


def all_normal_months(episodes, country: str, *, allow_oos: bool) -> list[str]:
    out = []
    for e in episodes:
        if e["country"] != country or e["systemic"] != 1:
            continue
        if not allow_oos and e["start"].year > 2019:
            continue
        if e["dev_eligibility"] == "PRE_DEV_ONLY":
            continue
        out.append(first_normal_month(e))
    return sorted(set(out))


def all_onset_months(episodes, country: str, *, allow_oos: bool) -> list[str]:
    out = []
    for e in episodes:
        if e["country"] != country or e["systemic"] != 1:
            continue
        if not allow_oos and e["start"].year > 2019:
            continue
        if e["dev_eligibility"] == "PRE_DEV_ONLY":
            continue
        if e["start"].year >= 2005:
            out.append(f"{e['start'].year:04d}-{e['start'].month:02d}")
    return sorted(set(out))


# ---------------------------------------------------------------------------
# month -> week strict as-of join (P0-06)
# ---------------------------------------------------------------------------
def release_week(month: str, week_ends: pd.Series, lag_days: int = 0):
    """First week_end whose close is >= release date of ``month``."""
    rel = month_end(int(month[:4]), int(month[5:7])) + timedelta(days=lag_days)
    later = week_ends[week_ends >= pd.Timestamp(rel)]
    return None if later.empty else later.iloc[0]


def asof_week_join(weekly: pd.DataFrame, monthly: pd.DataFrame, *,
                   week_col: str = "week_end", month_col: str = "month",
                   lag_days: int = 0) -> pd.DataFrame:
    """Attach the *latest already published* monthly row to each week.

    A weekly decision at ``week_end`` may only use the monthly row whose
    publication date (month end + lag) is <= week_end.  This is the backward
    as-of direction required by P0-06.
    """
    m = monthly.copy()
    m["_pub"] = [pd.Timestamp(month_end(int(x[:4]), int(x[5:7])) + timedelta(days=lag_days))
                 for x in m[month_col]]
    m = m.sort_values("_pub")
    w = weekly.copy().sort_values(week_col)
    out = pd.merge_asof(w, m, left_on=week_col, right_on="_pub",
                        direction="backward", allow_exact_matches=True)
    return out


def coverage_table(episodes) -> pd.DataFrame:
    """RQ01_DATA_COVERAGE rows: country x month-eligibility x episodes."""
    rows = []
    for c in EA7:
        eps = [e for e in episodes if e["country"] == c and e["systemic"] == 1
               and e["dev_eligibility"] != "PRE_DEV_ONLY"]
        rows.append({
            "country": c,
            "n_episodes_eligible": len(eps),
            "onset_months": ";".join(all_onset_months(episodes, c, allow_oos=True)),
            "first_normal_months": ";".join(all_normal_months(episodes, c, allow_oos=True)),
            "dev_scorable": True,
            "oos_esrb_scorable_window": f"{ESRB_SCORABLE_START}..{ESRB_SCORABLE_END}",
            "oos_no_esrb_window": "2021-05..2024-12",
            "esrb_coverage_end": "UNVERIFIED",
            "label_status_policy": "POLICY_OF_PRESENTATION",
        })
    return pd.DataFrame(rows)
