"""REV363 labels: ex-post ESRB benchmark only, never a feature at issue time.

The canonical workbook supplies 21 episodes (13 with DEV chronology).  A month
belongs to an episode on the half-open interval ``[StartDate, NormalDate)``;
the NormalDate month is N.  This is the same 1,218-key contract used by REV362.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import EA7, CLASSES
from .data import month_end, month_range, _exact


def load_episodes(root) -> list[dict]:
    p = _exact(Path(root), "ESRB_RQ01_CANONICAL_BENCHMARK.xlsx")
    sheets = pd.ExcelFile(p).sheet_names
    sheet = "EA7_Canonical_Episodes" if "EA7_Canonical_Episodes" in sheets else "Canonical_Episodes"
    df = pd.read_excel(p, sheet_name=sheet)
    df["StartDate"] = pd.to_datetime(df["StartDate"]).dt.date
    df["NormalDate"] = pd.to_datetime(df["NormalDate"]).dt.date
    out = []
    for _, r in df.iterrows():
        out.append({
            "country": str(r["Country"]),
            "start": r["StartDate"],
            "normal": r["NormalDate"],
            "origin": str(r["Origin"]),
            "systemic": int(r["SystemicFlag"]),
            "dev_eligibility": str(r["DEV_Eligibility"]),
            "ea7_scope": str(r["EA7_ModelScope"]),
            "role": str(r["BenchmarkRole"]),
        })
    return out


def label_month(episodes: list[dict], country: str, month: str, *, allow_oos: bool) -> str:
    d = month_end(int(month[:4]), int(month[5:7]))
    for e in episodes:
        if e["country"] != country or e["systemic"] != 1:
            continue
        if not allow_oos and e["start"].year > 2019:
            continue
        if e["start"] <= d < e["normal"]:
            return e["origin"] if e["origin"] in CLASSES else "N"
    return "N"


def label_frame(panel: pd.DataFrame, episodes: list[dict], *, allow_oos: bool) -> pd.DataFrame:
    df = panel.copy()
    df["label"] = [label_month(episodes, c, m, allow_oos=allow_oos)
                   for c, m in zip(df["country"], df["month"])]
    df["stress"] = (df["label"] != "N").astype(int)
    return df


def episode_rows(episodes: list[dict], *, allow_oos: bool) -> list[dict]:
    out = []
    for e in episodes:
        if not allow_oos and e["start"].year > 2019:
            continue
        if e["country"] not in EA7 or e["systemic"] != 1:
            continue
        if e["dev_eligibility"] == "PRE_DEV_ONLY":
            continue
        out.append(e)
    return out


def onset_months(episodes: list[dict], country: str, *, allow_oos: bool) -> list[str]:
    starts = []
    for e in episode_rows(episodes, allow_oos=allow_oos):
        if e["country"] == country and e["start"].year >= 2005:
            starts.append(f"{e['start'].year:04d}-{e['start'].month:02d}")
    return sorted(set(starts))


def exit_months(episodes: list[dict], country: str, *, allow_oos: bool) -> list[str]:
    ends = []
    for e in episode_rows(episodes, allow_oos=allow_oos):
        if e["country"] == country and e["normal"].year >= 2005:
            # first N month is the month after NormalDate's month-end
            y, m = e["normal"].year, e["normal"].month
            m += 1
            if m == 13:
                y, m = y + 1, 1
            ends.append(f"{y:04d}-{m:02d}")
    return sorted(set(ends))


def onset_labels(panel: pd.DataFrame, episodes: list[dict], horizon_months: int,
                 *, allow_oos: bool) -> pd.Series:
    """1 if a first onset occurs in (issue, issue+horizon]; requires N at issue."""
    from .data import add_months
    starts = {c: onset_months(episodes, c, allow_oos=allow_oos) for c in EA7}

    def mi(m):
        return int(m[:4]) * 12 + int(m[5:7]) - 1

    out = []
    for c, m, lab in zip(panel["country"], panel["month"], panel["label"]):
        if lab != "N":
            out.append(0)
            continue
        u = mi(m)
        out.append(int(any(u < mi(t) <= u + horizon_months for t in starts.get(c, []))))
    return pd.Series(out, index=panel.index, name="onset")


def future_state_labels(panel: pd.DataFrame, episodes: list[dict], horizon_months: int,
                        *, allow_oos: bool) -> pd.Series:
    from .data import add_months
    from research_adapters.vote_contract import LABEL_CENSORED
    lookup = {(c, m): l for c, m, l in zip(panel["country"], panel["month"], panel["label"])}
    out = []
    for c, m in zip(panel["country"], panel["month"]):
        t = add_months(m, horizon_months)
        # an unobserved future month is a CENSORED label (scoring status), never a state class and never N
        out.append(lookup.get((c, t), LABEL_CENSORED))
    return pd.Series(out, index=panel.index, name="future_state")


def time_to_exit_labels(panel: pd.DataFrame, episodes: list[dict], *, allow_oos: bool):
    """For every stress month, months until first N (event) or censoring."""
    from .data import add_months
    exits = {c: exit_months(episodes, c, allow_oos=allow_oos) for c in EA7}

    def mi(m):
        return int(m[:4]) * 12 + int(m[5:7]) - 1

    out = []
    for c, m, lab in zip(panel["country"], panel["month"], panel["label"]):
        if lab == "N":
            out.append(None)
            continue
        u = mi(m)
        later = [mi(t) - u for t in exits.get(c, []) if mi(t) > u]
        out.append(min(later) if later else None)
    return pd.Series(out, index=panel.index, name="months_to_exit")
