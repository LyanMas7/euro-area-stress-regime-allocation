"""Episode-level conditional source confirmation diagnostic (RQ01-07).

Research diagnostic only: it changes the event from a monthly UP vote to a
source vote observed anywhere in a canonical phase episode.  It does not
change any official phase or threshold.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import pandas as pd

from .metrics import wilson_ci


def _episodes(tape: pd.DataFrame):
    out = []
    for country, sub in tape.sort_values(["country", "issue_at"]).groupby("country"):
        cur, rows = None, []
        for _, r in sub.iterrows():
            st = r["state_after"]
            if st != cur:
                if cur is not None and rows:
                    out.append({"country": country, "phase": cur, "issues": rows})
                cur, rows = st, []
            rows.append(r)
        if cur is not None and rows:
            out.append({"country": country, "phase": cur, "issues": rows})
    return out


def run(canonical_csv, native_individual_csv, out_dir) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    phase = pd.read_csv(canonical_csv)
    ind = pd.read_csv(native_individual_csv)
    phase = phase[phase["state_after"].isin(["ENTRY", "PEAK", "DE_ESCALATION", "RECOVERY"])]
    episodes = _episodes(phase)
    # Source is confirmed in an episode if it has a PASS vote at any issue in the phase.
    rows = []
    for ep in episodes:
        issues = [r["issue_at"] for r in ep["issues"]]
        sub = ind[(ind.country == ep["country"]) & (ind.issue.isin(issues))]
        present = {}
        ev = sub[sub["status"] == "EVALUATED"]
        for s, g in ev.groupby("source"):
            present[s] = int((g["vote"] == 1).any())
        rows.append({"country": ep["country"], "phase": ep["phase"],
                     "n_issues": len(issues), "n_sources_evaluated": len(present),
                     "evidence_status": "EVALUATED" if present else "NOT_EVALUATED",
                     **{f"present_{s}": present.get(s) for s in
                        ("CISS", "CLIFS", "DFC", "GFC", "VIX")}})
    ep = pd.DataFrame(rows)
    ep.to_csv(out / "RQ01_07_EPISODE_SOURCE_CONFIRMATION.csv", index=False)
    # Exact subset conditionals on the episode denominator; missing source is not 0.
    cond = []
    sources = ("CISS", "CLIFS", "DFC", "GFC", "VIX")
    for phase_name, sub in ep[ep.evidence_status == "EVALUATED"].groupby("phase"):
        n = len(sub)
        for r in (1, 2, 3, 4):
            for S in combinations(sources, r):
                colsS = [f"present_{s}" for s in S]
                mask = sub[colsS].notna().all(axis=1)
                nS = int(mask.sum())
                for j in [s for s in sources if s not in S]:
                    colj = f"present_{j}"
                    valid = mask & sub[colj].notna()
                    nv = int(valid.sum())
                    cnt = int((valid & (sub[colj] == 1)).sum())
                    cond.append({"phase": phase_name, "subset": "&".join(S), "source": j,
                                 "n_episodes": n, "n_subset": nS, "n_valid": nv,
                                 "n_event": cnt,
                                 "probability": (cnt / nv if nv else None),
                                 "ci95": wilson_ci(cnt, nv) if nv else [None, None],
                                 "status": ("IDENTIFIABLE" if nv >= 5 else
                                            "LOW_SUPPORT" if nv else "NO_SUBSET_SUPPORT")})
    pd.DataFrame(cond).to_csv(out / "RQ01_07_EPISODE_CONDITIONAL.csv", index=False)
    return {"n_episodes": len(ep), "n_evaluated_episodes": int((ep.evidence_status == "EVALUATED").sum()),
            "n_conditional_rows": len(cond),
            "phases": ep.loc[ep.evidence_status == "EVALUATED", "phase"].value_counts().to_dict() if len(ep) else {}}
