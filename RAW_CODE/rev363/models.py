"""REV363 four-M two-head estimators with frozen serialization.

Each M is a two-head model:

    stress head   : P(stress) = sigma(x' beta)
    subtype head  : q = softmax(P(D), P(G), P(BOTH) | stress)
    p4            : [1 - P(stress), P(stress)*q_D, P(stress)*q_G, P(stress)*q_BOTH]

This keeps the four-class simplex and a separately interpretable Stress/N
decision.  The M hierarchy is the REV363 block hierarchy in ``config``.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

from .config import CLASSES, MODEL_BLOCKS, MODEL_VERSION, sha256_json

warnings.filterwarnings("ignore")


@dataclass
class TwoHeadModel:
    name: str
    blocks: tuple
    columns: dict
    stress_coef: np.ndarray
    stress_intercept: float
    subtype_coef: np.ndarray
    subtype_intercept: np.ndarray
    n_fit: int
    n_stress: int
    fit_logloss: float
    fit_brier: float
    version: str = MODEL_VERSION

    def to_dict(self):
        return {
            "name": self.name, "blocks": list(self.blocks), "columns": self.columns,
            "stress_coef": self.stress_coef.tolist(),
            "stress_intercept": float(self.stress_intercept),
            "subtype_coef": self.subtype_coef.tolist(),
            "subtype_intercept": self.subtype_intercept.tolist(),
            "n_fit": int(self.n_fit), "n_stress": int(self.n_stress),
            "fit_logloss": float(self.fit_logloss), "fit_brier": float(self.fit_brier),
            "version": self.version, "classes": list(CLASSES),
        }

    @classmethod
    def from_dict(cls, d):
        return cls(name=d["name"], blocks=tuple(d["blocks"]), columns=d["columns"],
                   stress_coef=np.asarray(d["stress_coef"], float),
                   stress_intercept=float(d["stress_intercept"]),
                   subtype_coef=np.asarray(d["subtype_coef"], float),
                   subtype_intercept=np.asarray(d["subtype_intercept"], float),
                   n_fit=d["n_fit"], n_stress=d["n_stress"],
                   fit_logloss=d["fit_logloss"], fit_brier=d["fit_brier"],
                   version=d.get("version", MODEL_VERSION))

    @property
    def digest(self):
        return sha256_json(self.to_dict())


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _fit_binary(X, y, *, C=1.0, class_weight=None):
    m = LogisticRegression(C=C, solver="lbfgs", max_iter=8000,
                           class_weight=class_weight, random_state=363)
    m.fit(X, y)
    coef = np.asarray(m.coef_, float).ravel()
    intercept = float(np.asarray(m.intercept_, float).ravel()[0])
    return coef, intercept, m


def _fit_multinomial(X, y, *, C=1.0, class_weight=None):
    m = LogisticRegression(C=C, solver="lbfgs", max_iter=8000,
                           class_weight=class_weight, random_state=363)
    m.fit(X, y)
    coef = np.asarray(m.coef_, float)
    intercept = np.asarray(m.intercept_, float).ravel()
    return coef, intercept, m


def build_matrix(design, blocks):
    cols = []
    for b in blocks:
        cols.append(design["X"][b])
    return np.column_stack(cols), sum((design["columns"][b] for b in blocks), [])


def fit_model(design, y4, name, *, C=1.0, class_weight=None):
    blocks = MODEL_BLOCKS[name]
    X, cols = build_matrix(design, blocks)
    y4 = np.asarray(y4, int)
    yb = (y4 != 0).astype(int)
    sc, si, sm = _fit_binary(X, yb, C=C, class_weight=class_weight)
    # subtype conditional on stress
    stress_idx = np.where(yb == 1)[0]
    if len(stress_idx) == 0 or len(np.unique(y4[stress_idx])) < 2:
        raise ValueError("SUBTYPE_HEAD_REQUIRES_AT_LEAST_TWO_STRESS_CLASSES")
    sub_y = y4[stress_idx] - 1  # D=0,G=1,BOTH=2
    uc, ui, um = _fit_multinomial(X[stress_idx], sub_y, C=C, class_weight=class_weight)
    p = predict_probabilities_raw(X, sc, si, uc, ui)
    p_true = p[np.arange(len(y4)), y4]
    ll = float(-np.mean(np.log(np.clip(p_true, 1e-15, 1.0))))
    brier = float(np.mean(np.sum((p - np.eye(4)[y4]) ** 2, axis=1)))
    return TwoHeadModel(name=name, blocks=blocks,
                        columns={"all": cols,
                                 "stress": cols, "subtype": cols},
                        stress_coef=sc, stress_intercept=si,
                        subtype_coef=uc, subtype_intercept=ui,
                        n_fit=len(y4), n_stress=int(yb.sum()),
                        fit_logloss=ll, fit_brier=brier)


def predict_probabilities_raw(X, stress_coef, stress_intercept, subtype_coef, subtype_intercept):
    ps = _sigmoid(X @ stress_coef + stress_intercept)
    z = X @ subtype_coef.T + subtype_intercept[None, :]
    q = _softmax(z)
    p = np.column_stack([1.0 - ps, ps * q[:, 0], ps * q[:, 1], ps * q[:, 2]])
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    return p


def predict_model(model: TwoHeadModel, design, blocks=None):
    X, _ = build_matrix(design, model.blocks if blocks is None else blocks)
    return predict_probabilities_raw(X, model.stress_coef, model.stress_intercept,
                                     model.subtype_coef, model.subtype_intercept)


def fit_all(design, y4, *, C=1.0, class_weight=None):
    out = {}
    for name in MODEL_BLOCKS:
        out[name] = fit_model(design, y4, name, C=C, class_weight=class_weight)
    return out


def serialize_models(models: dict) -> dict:
    payload = {k: v.to_dict() for k, v in models.items()}
    payload["bundle_version"] = MODEL_VERSION
    payload["bundle_sha256"] = sha256_json({k: v.to_dict() for k, v in models.items()})
    return payload


def deserialize_models(payload: dict) -> dict:
    out = {k: TwoHeadModel.from_dict(v) for k, v in payload.items()
           if k in MODEL_BLOCKS}
    if payload.get("bundle_sha256"):
        check = sha256_json({k: v.to_dict() for k, v in out.items()})
        if check != payload["bundle_sha256"]:
            raise ValueError("MODEL_BUNDLE_SHA_MISMATCH")
    return out
