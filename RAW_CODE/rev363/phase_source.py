"""Phase-source adapter: canonical RQ01 six-phase tape -> downstream consumers.

The canonical tape uses `DE_ESCALATION`; the existing Bottom F1/F2 consumers
expect `DEESC`. This adapter maps only the phase labels, preserves a per-row
source flag, and falls back to the legacy phase where the canonical tape has no
row (currently FI). It does not alter RQ01 state records.
"""
from __future__ import annotations

import pandas as pd

PHASE_MAP = {"PEAK": "PEAK", "DE_ESCALATION": "DEESC", "RECOVERY": "RECOVERY",
             "EXIT": "EXIT", "ENTRY": "ENTRY", "PRE_ONSET": "PRE_ONSET", "NONE": "NONE"}
SOURCES = ("CANONICAL", "LEGACY")


def load_canonical_phase(path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"country", "issue_at", "state_after"}
    if not required.issubset(frame.columns):
        raise ValueError("CANONICAL_PHASE_SCHEMA_REQUIRED:" + ",".join(sorted(required - set(frame.columns))))
    frame = frame[["country", "issue_at", "state_after"]].copy()
    frame["month"] = frame["issue_at"].astype(str).str[:7]
    frame["canonical_phase"] = frame["state_after"].map(PHASE_MAP)
    if frame["canonical_phase"].isna().any():
        bad = sorted(set(frame.loc[frame["canonical_phase"].isna(), "state_after"]))
        raise ValueError("UNREGISTERED_CANONICAL_PHASE:" + ",".join(map(str, bad)))
    if frame.duplicated(["country", "month"]).any():
        raise ValueError("DUPLICATE_CANONICAL_PHASE_KEY")
    return frame[["country", "month", "canonical_phase"]]


def apply_canonical_phase(panel: pd.DataFrame, canonical: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return a copy with canonical phases where available and legacy per-row fallback."""
    out = panel.copy()
    if "phase" not in out.columns:
        raise ValueError("PANEL_PHASE_COLUMN_REQUIRED")
    out["phase_legacy"] = out["phase"]
    out["phase_source"] = "LEGACY"
    lookup = {(r["country"], r["month"]): r["canonical_phase"] for _, r in canonical.iterrows()}
    for i, row in out.iterrows():
        key = (row["country"], str(row["month"])[:7])
        mapped = lookup.get(key)
        if mapped is not None:
            out.at[i, "phase"] = mapped
            out.at[i, "phase_source"] = "CANONICAL"
    counts = out["phase_source"].value_counts().to_dict()
    return out, {"n_rows": int(len(out)), "n_canonical": int(counts.get("CANONICAL", 0)),
                 "n_legacy_fallback": int(counts.get("LEGACY", 0)),
                 "n_fallback_by_country": out.loc[out["phase_source"] == "LEGACY", "country"]
                 .value_counts().to_dict(),
                 "phase_source": "CANONICAL_WITH_LEGACY_FALLBACK"}
