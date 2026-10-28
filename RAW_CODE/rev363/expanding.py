"""Point-in-time-safe expanding-window refit engine (contract §C, P0-08).

For an issue month ``u`` the training set is

    D_u = { (x_s, y_s) : feature_available_at(s) <= u
                        AND label_release_at(s) <= u
                        AND label_status(s) == SCORABLE }

Features are already prior-only, and the label release clock is the documented
ESRB research lag, so no future information can enter ``D_u``.  After the last
scorable ESRB month (2021-04 by user presentation policy) no new labelled row is
added and the model stops updating -- it is never pseudo-labelled from its own
predictions.

Every refit is logged in ``ASOF_EXPANDING_FIT_LEDGER`` and an invariance probe
re-runs the refit with all post-``u`` labels perturbed, asserting the issuance
prediction at ``u`` is byte-identical.

Status: OOS 2020-2024 has already been inspected by earlier revisions, so every
expanding result is labelled ``POST_OOS_EXPOSURE_EXPERIMENT`` and is not
independent validation.  Under the frozen File-01 contract (two indivisible
blocks, no OOS refit) this axis is ``BLOCKED_BY_ACTIVE_CONTRACT`` pending a
signed amendment; the experiment is executed only to produce the requested
evidence and the interface.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from rev363 import estimators as EST
from rev363.labels_next import STATUS_SCORABLE, label_release_month
from rev363.util import sha256_json

UPDATE_POLICY = "OOS_EXPANDING"
LABEL = "POST_OOS_EXPOSURE_EXPERIMENT"
CONTRACT_STATUS = "BLOCKED_BY_ACTIVE_CONTRACT"


def _train_index(df: pd.DataFrame, columns: list, u: str, dev_mask: np.ndarray) -> np.ndarray:
    rel = np.array([label_release_month(m) for m in df["month"]])
    status = df["label_status"].values
    oos_ok = (np.array(df["month"].values) < u) & (rel <= u) & (status == STATUS_SCORABLE)
    return np.where(dev_mask | oos_ok)[0]


def expanding_refit(df: pd.DataFrame, columns: list, y4: np.ndarray, *,
                    dev_mask: np.ndarray, family: str, name: str,
                    oos_months: list[str]) -> dict:
    # Hard scientific contract: do not enter any OOS supervised refit path.
    # The function interface is retained for research design and testing only.
    from rev363.w01_contract import enforce_execution
    enforce_execution(supervised_refit=True)
    preds, ledger = [], []
    for u in oos_months:
        tr = _train_index(df, columns, u, dev_mask)
        Xtr = EST.standardise_apply  # placeholder to keep import explicit
        X = np.column_stack([pd.to_numeric(df[c], errors="coerce").values.astype(float)
                             for c in columns])
        model = EST.fit_two_family(X[tr], columns, y4[tr], family=family, name=name)
        tgt = np.where(df["month"].values == u)[0]
        for i in tgt:
            preds.append({"country": df["country"].values[i], "issue": u,
                          "p4": model.p4(X[[i]])[0].tolist()})
        last_lab = df["month"].values[tr].max()
        ledger.append({
            "issue_at": u, "model_family": family, "model_name": name,
            "update_policy": UPDATE_POLICY, "fit_cutoff_at": last_lab,
            "last_usable_label_at": last_lab, "n_training": int(len(tr)),
            "n_training_stress": int((y4[tr] != 0).sum()),
            "data_hash": sha256_json([[df["country"].values[i], df["month"].values[i]]
                                      for i in tr]),
            "model_signature": model.signature, "model_digest": model.digest,
            "prediction_frozen_at": u,
            "label_status": (STATUS_SCORABLE if u <= "2021-04" else "NOT_SCORABLE"),
            "contract_status": CONTRACT_STATUS, "label": LABEL,
        })
    return {"predictions": preds, "ledger": ledger,
            "status": LABEL, "contract_status": CONTRACT_STATUS}


def invariance_probe(df: pd.DataFrame, columns: list, y4: np.ndarray, *,
                     dev_mask: np.ndarray, family: str, name: str, u: str) -> dict:
    """Perturb all post-u labels and prove the issuance prediction at u is unchanged."""
    tr = _train_index(df, columns, u, dev_mask)
    X = np.column_stack([pd.to_numeric(df[c], errors="coerce").values.astype(float)
                         for c in columns])
    m1 = EST.fit_two_family(X[tr], columns, y4[tr], family=family, name=name)
    y4b = np.asarray(y4, int).copy()
    future = np.where(np.array(df["month"].values) > u)[0]
    rng = np.random.default_rng(363)
    y4b[future] = rng.integers(0, 4, size=len(future))
    m2 = EST.fit_two_family(X[tr], columns, y4b[tr], family=family, name=name)
    tgt = np.where(df["month"].values == u)[0]
    p1 = m1.p4(X[tgt])
    p2 = m2.p4(X[tgt])
    same = bool(np.allclose(p1, p2, atol=1e-12))
    return {"issue_at": u, "family": family, "name": name,
            "n_perturbed_future_rows": int(len(future)),
            "predictions_identical": same,
            "max_abs_diff": float(np.max(np.abs(p1 - p2))) if len(tgt) else None,
            "status": "PASS_NO_LOOKAHEAD" if same else "FAIL_LOOKAHEAD"}
