"""Causal, label-free RQ02 Long/Short feature construction.

Both RQ02 runners previously calibrated magnitude quantiles from full-frame
``label == "N"`` rows. That made an issued feature depend on labels observed
later (and, for OOS, on OOS labels). This module replaces the calibration with
a strictly prior-only rule:

* at issue month ``t`` the quantile of ``|long_s|`` / ``|short_s|`` uses only
  observations with month ``< t`` in the same country;
* no label, no future magnitude, and no OOS observation enters the threshold;
* insufficient history yields ``NaN`` (unknown), never a zero vote;
* OOS callers must pass the DEV-frozen calibration table.

The emitted columns keep the REV363 names so existing model/fit consumers
remain compatible. Threshold provenance is kept in ``_thr_long_*`` and
``_thr_short_*`` and in ``df.attrs["long_short_calibration"]``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import EA7, PAIRS21, SOURCES7

CALIBRATION_QUANTILE = 0.75
CALIBRATION_MIN_PRIOR = 12
CALIBRATION_MODE = "EXPANDING_PRIOR_ABS_Q75_LABEL_FREE"
FROZEN_MODE = "FROZEN_DEV_CALIBRATION"


def add_long_short_changes(df: pd.DataFrame) -> pd.DataFrame:
    """27-month Long and 6-month Short signed changes on the PIT z scale."""
    out = df.sort_values(["country", "month"]).reset_index(drop=True).copy()
    for s in SOURCES7:
        out[f"long_{s}"] = out.groupby("country")[f"z_{s}"].transform(lambda x: x - x.shift(27))
        out[f"short_{s}"] = out.groupby("country")[f"z_{s}"].transform(lambda x: x - x.shift(6))
    return out


def _prior_abs_quantile(series: pd.Series, q: float, min_periods: int) -> pd.Series:
    """Quantile over strictly prior magnitudes within one country."""
    prior = series.astype(float).abs().shift(1)
    return prior.expanding(min_periods=min_periods).quantile(q)


def _fixed_threshold(df: pd.DataFrame, values: dict, field: str, source: str) -> pd.Series:
    value = values.get(field)
    if value is None or not np.isfinite(float(value)) or float(value) <= 0:
        raise ValueError(f"CALIBRATION_{source}_{field.upper()}_INVALID")
    return pd.Series(float(value), index=df.index, dtype=float)


def apply_long_short_votes(
    df: pd.DataFrame,
    *,
    calibration: dict | None = None,
    long_sources: tuple[str, ...] = SOURCES7,
    q: float = CALIBRATION_QUANTILE,
    min_periods: int = CALIBRATION_MIN_PRIOR,
) -> pd.DataFrame:
    """Add L_/S_, source counts, Long pair and 21 Short pair columns.

    ``calibration=None`` uses expanding prior-only thresholds (DEV). A supplied
    table is required for OOS and is applied unchanged to every row.
    """
    out = df.copy()
    if calibration is not None:
        missing = [s for s in SOURCES7 if s not in calibration]
        if missing:
            raise ValueError("CALIBRATION_SOURCES_MISSING:" + ",".join(missing))

    for s in SOURCES7:
        col_l = f"long_{s}"
        col_s = f"short_{s}"
        if calibration is None:
            for c in EA7:
                idx = out.index[out["country"] == c]
                if len(idx) == 0:
                    continue
                out.loc[idx, f"_thr_long_{s}"] = _prior_abs_quantile(
                    out.loc[idx, col_l], q, min_periods).values
                out.loc[idx, f"_thr_short_{s}"] = _prior_abs_quantile(
                    out.loc[idx, col_s], q, min_periods).values
            mode = CALIBRATION_MODE
        else:
            out[f"_thr_long_{s}"] = _fixed_threshold(out, calibration[s], "long_q75", s)
            out[f"_thr_short_{s}"] = _fixed_threshold(out, calibration[s], "short_q75", s)
            mode = FROZEN_MODE

        lv, sv, zz = out[col_l], out[col_s], out[f"z_{s}"]
        tl, ts = out[f"_thr_long_{s}"], out[f"_thr_short_{s}"]
        known_l = lv.notna() & zz.notna() & tl.notna()
        known_s = sv.notna() & zz.notna() & ts.notna()
        out[f"L_{s}"] = np.where(known_l, ((lv >= tl) & (zz > 0)).astype(float), np.nan)
        out[f"S_{s}"] = np.where(known_s, ((sv >= ts) & (zz > 0)).astype(float), np.nan)

    long_cols = [f"L_{s}" for s in long_sources]
    short_cols = [f"S_{s}" for s in SOURCES7]
    out["n_long_sources"] = out[long_cols].sum(axis=1, min_count=len(long_cols))
    out["n_short_sources"] = out[short_cols].sum(axis=1, min_count=len(short_cols))
    out["long_pair_pass"] = np.where(
        out[["L_CG", "L_DSR-P"]].notna().all(axis=1),
        ((out["L_CG"] == 1) & (out["L_DSR-P"] == 1)).astype(float), np.nan)
    for a, b in PAIRS21:
        out[f"short_pair_{a}_{b}"] = np.where(
            out[[f"S_{a}", f"S_{b}"]].notna().all(axis=1),
            ((out[f"S_{a}"] == 1) & (out[f"S_{b}"] == 1)).astype(float), np.nan)
    out["n_short_pairs"] = out[[f"short_pair_{a}_{b}" for a, b in PAIRS21]].sum(
        axis=1, min_count=len(PAIRS21))
    out["n_long_pairs"] = out["long_pair_pass"]
    out.attrs["long_short_calibration"] = final_calibration(out)
    out.attrs["long_short_calibration_mode"] = mode
    return out


def final_calibration(df: pd.DataFrame) -> dict:
    """Last available per-country prior threshold, pooled by source median."""
    last = df.sort_values(["country", "month"]).groupby("country", as_index=False).tail(1)
    out: dict[str, dict[str, float]] = {}
    for s in SOURCES7:
        vals_l = pd.to_numeric(last.get(f"_thr_long_{s}"), errors="coerce").dropna()
        vals_s = pd.to_numeric(last.get(f"_thr_short_{s}"), errors="coerce").dropna()
        out[s] = {
            "long_q75": float(vals_l.median()) if len(vals_l) else 1.0,
            "short_q75": float(vals_s.median()) if len(vals_s) else 1.0,
        }
    return out


def is_frozen_calibration(calibration: dict) -> bool:
    return all(
        isinstance(calibration.get(s), dict)
        and np.isfinite(float(calibration[s].get("long_q75", np.nan)))
        and np.isfinite(float(calibration[s].get("short_q75", np.nan)))
        for s in SOURCES7
    )
