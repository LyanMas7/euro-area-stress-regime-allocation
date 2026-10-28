"""Corrected panel + clean 4M factorial design (P1-10, P0-03).

Block contract (V8 REV363 §3):
    M00 = BASE
    M01 = BASE + IND
    M10 = BASE + JOINT_selected
    M11 = BASE + IND + JOINT_selected
All four share the *same* baseline, including the country terms, so a
difference between M models is attributable only to the IND / JOINT blocks.
The baseline REV363 put the country-varying block only in M11 (defect P1-10);
here it is part of the common BASE for every M.

Feature construction is prior-only (expanding/rolling z-scores, differences and
drawdowns computed strictly from earlier months), so an OOS label can never
change a feature available at issue time (defect P0-03).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from rev363 import features as F
from rev363 import labels as L
from rev363.config import CLASSES, MODEL_BLOCKS
from rev363.data import build_monthly_raw, find_dataset_root
from rev363.labels_next import STATUS_SCORABLE, gold_label, label_status


def build_panel(dataset_root, *, row_start="2004-01", row_end="2024-12",
                data_end="2024-12", spec=None, label_mode="auto"):
    """Assemble the corrected monthly panel with features, votes and labels."""
    root = find_dataset_root(dataset_root)
    episodes = L.load_episodes(root)
    raw = build_monthly_raw(root, row_start="2004-01", row_end=data_end, data_end=data_end)

    df = raw.copy()
    df["label"] = [gold_label(episodes, c, m) for c, m in zip(df["country"], df["month"])]
    df["label_status"] = [label_status(m) for m in df["month"]]
    df["stress"] = (df["label"] != "N").astype(int)

    feats = F.build_raw_features(df)
    rows = feats[(feats["month"] >= row_start) & (feats["month"] <= row_end)].reset_index(drop=True)

    if spec is None:
        dev_rows = rows[(rows["month"] >= "2005-07") & (rows["month"] <= "2019-12")]
        spec = F.calibrate_spec(dev_rows, dev_rows["label"])
    votes = F.compute_votes(rows, spec)
    votes = F.compute_phases(votes, spec)
    design = F.build_design(votes, spec, fit=(spec.base_columns == []))
    return {"root": root, "episodes": episodes, "raw": raw, "panel": design["df"],
            "spec": design["spec"], "design": design}


def corrected_blocks(spec) -> dict:
    """Corrected 4M block definition: country terms live in the common BASE."""
    base = list(dict.fromkeys(spec.base_columns + spec.country_columns))
    ind = list(spec.ind_columns)
    joint = list(spec.joint_columns)
    return {
        "M00": base,
        "M01": base + ind,
        "M10": base + joint,
        "M11": base + ind + joint,
        "_blocks": {"BASE": base, "IND": ind, "JOINT": joint},
    }


def matrix_for(df: pd.DataFrame, columns: list) -> np.ndarray:
    return np.column_stack([pd.to_numeric(df[c], errors="coerce").values.astype(float)
                            for c in columns])


def y4_from(df: pd.DataFrame) -> np.ndarray:
    return np.asarray([CLASSES.index(x) for x in df["label"]], int)
