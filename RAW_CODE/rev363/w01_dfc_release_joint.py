"""REV371 isolated research: DFC quarterly + counterpart event first-available-month join.

No carried-quarter counts, no future release, no unequal-source value summation.
This does NOT establish publisher-vintage PIT or restore an official RQ01 owner.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DFC_PAIRS = (("CISS", "DFC"), ("CLIFS", "DFC"), ("DFC", "GFC"), ("DFC", "VIX"))
COLUMNS = ["country", "issue", "native_fresh", "terminal_abs", "native_observed_at", "available_at"]


def _normalise(native: pd.DataFrame, source: str) -> pd.DataFrame:
    missing = set(COLUMNS).difference(native.columns)
    if missing:
        raise ValueError(f"MISSING_NATIVE_COLUMNS:{source}:{sorted(missing)}")
    x = native.loc[:, COLUMNS].copy()
    if x[["country", "issue"]].duplicated().any():
        raise ValueError("DUPLICATE_COUNTRY_ISSUE:" + source)
    if not x["issue"].astype(str).str.fullmatch(r"\d{4}-\d{2}").all():
        raise ValueError("INVALID_ISSUE_KEY:" + source)
    x["observed_date"] = pd.to_datetime(x["native_observed_at"], errors="coerce")
    x["available_date"] = pd.to_datetime(x["available_at"], errors="coerce")
    x["month_end"] = pd.to_datetime(x["issue"].astype(str) + "-01") + pd.offsets.MonthEnd(0)
    fresh = x["native_fresh"].fillna(False).astype(bool)
    if (fresh & (x["observed_date"].isna() | x["available_date"].isna())).any():
        raise ValueError("FRESH_WITHOUT_SOURCE_CLOCK:" + source)
    if (fresh & (x["observed_date"] > x["available_date"])).any():
        raise ValueError("OBSERVATION_AFTER_RELEASE:" + source)
    if (fresh & (x["available_date"] > x["month_end"])).any():
        raise ValueError("LOOKAHEAD_RELEASE_AFTER_ISSUE:" + source)
    if (fresh & x["observed_date"].notna() & x.duplicated(["country", "observed_date"], keep=False)).any():
        # One native period cannot be declared fresh more than once per country.
        observed_fresh = x.loc[fresh, ["country", "observed_date"]]
        if observed_fresh.duplicated().any():
            raise ValueError("DUPLICATE_FRESH_NATIVE_PERIOD:" + source)
    x["arrived_in_issue"] = fresh & x["available_date"].dt.strftime("%Y-%m").eq(x["issue"])
    x["signed_delta"] = pd.to_numeric(x["terminal_abs"], errors="coerce")
    return x


def dfc_pair_issue_vote(dfc: pd.DataFrame, partner: pd.DataFrame, *,
                        partner_name: str, dfc_min: float, partner_min: float,
                        partner_sign: int = 1) -> pd.DataFrame:
    """A same-issue *arrival* hypothesis, not a same-observation-date claim.

    Gate 0: both native periods genuinely new in the same issue month, available
    by issue-end. Gate 1/2: both oriented terminal magnitudes strictly positive
    and >= own source threshold; missing terminals => U; low magnitude => FAIL.
    Exactly one paired opportunity per country x DFC-native quarter; a carried
    quarter in subsequent months cannot create a pair.
    """
    if partner_name not in {"CISS", "CLIFS", "GFC", "VIX"}:
        raise ValueError("INVALID_PARTNER")
    if partner_sign not in (-1, 1):
        raise ValueError("INVALID_POLARITY")
    if not (np.isfinite(dfc_min) and dfc_min > 0 and np.isfinite(partner_min) and partner_min > 0):
        raise ValueError("INVALID_THRESHOLD")
    a = _normalise(dfc, "DFC")
    b = _normalise(partner, partner_name)
    a = a.rename(columns={c: c+"_DFC" for c in a if c not in ("country", "issue")})
    b = b.rename(columns={c: c+"_PARTNER" for c in b if c not in ("country", "issue")})
    z = a.merge(b, on=["country", "issue"], how="outer", validate="one_to_one", indicator=True)
    z = z.sort_values(["country", "issue"], kind="stable")
    eligible = (z["_merge"].eq("both") &
                z["arrived_in_issue_DFC"].fillna(False).astype(bool) &
                z["arrived_in_issue_PARTNER"].fillna(False).astype(bool) &
                z["signed_delta_DFC"].notna() & z["signed_delta_PARTNER"].notna())
    # Positive signed DFC increment is an explicit hypothesis, not a rescaling
    # of the signed data. The GFC counterparty uses the registered negative sign.
    gate_dfc = z["signed_delta_DFC"].ge(float(dfc_min))
    gate_other = (z["signed_delta_PARTNER"] * int(partner_sign)).ge(float(partner_min))
    passed = eligible & gate_dfc & gate_other
    # GLOBAL NO-U: no new matched native pair -> NOT_EVALUATED record (empty vote), never a third vote
    state = np.where(~eligible, None, np.where(passed, "PASS", "FAIL"))
    result = z[["country", "issue"]].copy()
    result["pair"] = ("DFC|" + partner_name if partner_name in ("GFC", "VIX")
                      else partner_name + "|DFC")
    result["vote"] = state
    result["eligible"] = eligible.to_numpy(bool)
    result["threshold_DFC"] = float(dfc_min)
    result["threshold_partner"] = float(partner_min)
    result["partner_orientation"] = int(partner_sign)
    oriented_other = z["signed_delta_PARTNER"] * int(partner_sign)
    sign_txt = lambda v: "?" if pd.isna(v) else "+" if v > 0 else "-" if v < 0 else "0"
    result["mode"] = [sign_txt(d) + sign_txt(o) if ok else None for d,o,ok in zip(z["signed_delta_DFC"],oriented_other,eligible)]
    result["decision_reason"] = np.where(~eligible,"UNAVAILABLE_NO_NEW_MATCHED_NATIVE_PAIR",
                                  np.where(passed,"BOTH_SOURCE_MAGNITUDES_PASS","GATE1_OR_GATE2_MAGNITUDE_FAIL"))
    result["gate_dfc"] = pd.Series(np.where(eligible, gate_dfc, None), index=z.index).tolist()
    result["gate_partner"] = pd.Series(np.where(eligible, gate_other, None), index=z.index).tolist()
    result["period_DFC"] = z["observed_date_DFC"].dt.strftime("%Y-%m-%d")
    result["period_partner"] = z["observed_date_PARTNER"].dt.strftime("%Y-%m-%d")
    result["available_DFC"] = z["available_date_DFC"].dt.strftime("%Y-%m-%d")
    result["available_partner"] = z["available_date_PARTNER"].dt.strftime("%Y-%m-%d")
    result["native_observation_gap_days"] = (z["observed_date_DFC"] - z["observed_date_PARTNER"]).abs().dt.days
    qualified = result.loc[result.eligible, ["country", "period_DFC"]]
    if qualified.duplicated().any():
        raise ValueError("CARRIED_DFC_QUARTER_REUSED")
    result["vote_status"] = np.where(result.vote.isin(["PASS", "FAIL"]), "EXECUTED", "NOT_EVALUATED")
    if not (result.vote.isin(["PASS", "FAIL"]) | (result.vote.isna() & ~result.eligible)).all():
        raise AssertionError("BAD_BINARY_VOTE_OR_NON_VOTE_RECORD")
    return result.reset_index(drop=True)
