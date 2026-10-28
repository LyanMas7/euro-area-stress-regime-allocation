"""Estimator families: LOGISTIC (incumbent) and XGBOOST (challenger).

Both families consume the *same* standardised 4-class target and the *same*
feature matrix for a given M-block set, so the comparison isolates the
estimator axis.  A calibration nuisance (Platt scaling fitted once on the whole DEV block,
no folds) is reported but never used to select between models or to choose thresholds.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

from rev363.util import sha256_json

try:  # pragma: no cover - availability is reported, never assumed
    import xgboost as xgb
    XGB_AVAILABLE = True
    XGB_VERSION = xgb.__version__
except Exception:  # pragma: no cover
    xgb = None
    XGB_AVAILABLE = False
    XGB_VERSION = None

# Pre-registered, identical booster geometry across M00..M11.
XGB_PARAMS = dict(
    n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.80,
    colsample_bytree=0.80, min_child_weight=2.0, reg_alpha=0.0, reg_lambda=1.0,
    seed=363, tree_method="hist", objective="multi:softprob", num_class=4,
    n_jobs=2, verbosity=0,
)
XGB_BINARY_PARAMS = dict(
    n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.80,
    colsample_bytree=0.80, min_child_weight=2.0, reg_alpha=0.0, reg_lambda=1.0,
    seed=363, tree_method="hist", objective="binary:logistic",
    n_jobs=2, verbosity=0,
)
LOGISTIC_PARAMS = dict(C=1.0, solver="lbfgs", max_iter=8000, random_state=363)


def standardise_fit(X: np.ndarray):
    med = np.nanmedian(X, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    filled = np.where(np.isfinite(X), X, med)
    sd = np.nanstd(filled, axis=0, ddof=0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-12), sd, 1.0)
    return med, sd


def standardise_apply(X: np.ndarray, med, sd) -> np.ndarray:
    Xf = np.where(np.isfinite(X), X, med)
    return (Xf - med) / sd


@dataclass
class FittedModel:
    family: str
    name: str
    columns: list
    median: np.ndarray
    scale: np.ndarray
    n_fit: int
    n_stress: int
    fit_logloss: float
    fit_brier: float
    n_class: int = 4
    coef: np.ndarray | None = None
    intercept: np.ndarray | None = None
    booster_b64: str | None = None
    xgb_params: dict | None = None
    fit_timestamp: str | None = None

    def matrix(self, X: np.ndarray) -> np.ndarray:
        return standardise_apply(np.asarray(X, float), self.median, self.scale)

    def p4(self, X: np.ndarray) -> np.ndarray:
        Xs = self.matrix(X)
        if self.family == "LOGISTIC":
            z = Xs @ self.coef.T + self.intercept[None, :]
            if self.n_class == 2:
                p1 = 1.0 / (1.0 + np.exp(-np.clip(z[:, 0], -40, 40)))
                return np.column_stack([1.0 - p1, p1])
            z = z - z.max(axis=1, keepdims=True)
            e = np.exp(z)
            return e / e.sum(axis=1, keepdims=True)
        bo = xgb.Booster()
        bo.load_model(bytearray(base64.b64decode(self.booster_b64)))
        p = np.asarray(bo.predict(xgb.DMatrix(Xs)), float)
        if self.n_class == 2:
            p = p.ravel()
            return np.column_stack([1.0 - p, p])
        return p

    def score(self, X: np.ndarray) -> np.ndarray:
        p = self.p4(X)
        return p[:, 1] if self.n_class == 2 else 1.0 - p[:, 0]

    @property
    def signature(self) -> dict:
        return {"family": self.family, "name": self.name, "columns": list(self.columns),
                "n_fit": self.n_fit, "n_stress": self.n_stress,
                "xgb_params": self.xgb_params, "n_coef": None if self.coef is None else int(self.coef.size),
                "booster_sha": (None if self.booster_b64 is None
                                else sha256_json(self.booster_b64))}

    @property
    def digest(self) -> str:
        return sha256_json(self.signature)


def _metrics(y4, p4) -> tuple[float, float]:
    y4 = np.asarray(y4, int)
    p = np.clip(np.asarray(p4, float), 1e-15, 1.0)
    ll = float(-np.mean(np.log(p[np.arange(len(y4)), y4])))
    oh = np.eye(p.shape[1])[y4]
    br = float(np.mean(np.sum((p - oh) ** 2, axis=1)))
    return ll, br


def fit_logistic(X, y4) -> tuple[np.ndarray, np.ndarray]:
    m = LogisticRegression(**LOGISTIC_PARAMS)
    m.fit(X, y4)
    return np.asarray(m.coef_, float), np.asarray(m.intercept_, float).ravel()


def fit_xgb(X, y4, *, n_class: int) -> str:
    params = dict(XGB_PARAMS if n_class > 2 else XGB_BINARY_PARAMS)
    dtrain = xgb.DMatrix(X, label=np.asarray(y4, int))
    bo = xgb.train(params, dtrain, num_boost_round=params["n_estimators"])
    raw = bo.save_raw(raw_format="json")
    return base64.b64encode(bytes(raw)).decode("ascii")


def fit_two_family(design_X: np.ndarray, columns: list, y4, *, family: str, name: str,
                   timestamp: str | None = None) -> FittedModel:
    """Fit one (family, M-name) model.  Handles binary or 4-class targets."""
    X = np.asarray(design_X, float)
    y4 = np.asarray(y4, int)
    n_class = 2 if len(np.unique(y4)) <= 2 else 4
    med, sd = standardise_fit(X)
    Xs = standardise_apply(X, med, sd)
    if family == "LOGISTIC":
        coef, intercept = fit_logistic(Xs, y4)
        model = FittedModel(family=family, name=name, columns=list(columns),
                            median=med, scale=sd, n_fit=len(y4),
                            n_stress=int((y4 != 0).sum()), fit_logloss=float("nan"),
                            fit_brier=float("nan"), n_class=n_class, coef=coef,
                            intercept=intercept, xgb_params=None, fit_timestamp=timestamp)
    elif family == "XGBOOST":
        if not XGB_AVAILABLE:
            raise RuntimeError("XGBOOST_NOT_AVAILABLE")
        b64 = fit_xgb(Xs, y4, n_class=n_class)
        model = FittedModel(family=family, name=name, columns=list(columns),
                            median=med, scale=sd, n_fit=len(y4),
                            n_stress=int((y4 != 0).sum()), fit_logloss=float("nan"),
                            fit_brier=float("nan"), n_class=n_class, booster_b64=b64,
                            xgb_params=dict(XGB_PARAMS if n_class > 2 else XGB_BINARY_PARAMS),
                            fit_timestamp=timestamp)
    else:
        raise ValueError(f"UNKNOWN_FAMILY:{family}")
    ll, br = _metrics(y4, model.p4(X))
    model.fit_logloss = ll
    model.fit_brier = br
    return model


# ---------------------------------------------------------------------------
# calibration nuisance (diagnostic only; never a selection device)
# ---------------------------------------------------------------------------
def _platt(scores, y):
    """Platt scaling fitted ONCE on the whole DEV block: p = sigmoid(a*logit(score) + b); returns (a, b).

    DEV is one indivisible block (no folds, no holdout, no refit), so there is no cross-validation here:
    deterministic Newton iterations on the logistic log-loss with a tiny ridge for stability.
    """
    s = np.clip(np.asarray(scores, float), 1e-9, 1 - 1e-9)
    z = np.log(s / (1 - s))
    y = np.asarray(y, float)
    theta = np.array([1.0, 0.0])
    design = np.column_stack([z, np.ones_like(z)])
    for _ in range(100):
        p = 1 / (1 + np.exp(-(design @ theta)))
        grad = design.T @ (p - y) / len(z) + 1e-9 * theta
        hess = (design.T * (p * (1 - p))) @ design / len(z) + 1e-9 * np.eye(2)
        step = np.linalg.solve(hess, grad)
        theta = theta - step
        if float(np.max(np.abs(step))) < 1e-12:
            break
    return float(theta[0]), float(theta[1])


def calibration_report(p4_dev, y4_dev, p4_oos, y4_oos) -> dict:
    """Reliability diagnostics and an explicit whole-DEV-block Platt nuisance calibrator (no folds)."""
    s_dev = 1.0 - np.asarray(p4_dev, float)[:, 0]
    s_oos = 1.0 - np.asarray(p4_oos, float)[:, 0]
    y_dev = (np.asarray(y4_dev, int) != 0).astype(int)
    y_oos = (np.asarray(y4_oos, int) != 0).astype(int)
    out = {"brier_dev_raw": float(np.mean((s_dev - y_dev) ** 2)),
           "brier_oos_raw": float(np.mean((s_oos - y_oos) ** 2)) if len(y_oos) else None,
           "status": "DIAGNOSTIC_ONLY_NOT_MODEL_SELECTION"}
    if len(np.unique(y_dev)) > 1:
        a, b = _platt(s_dev, y_dev)
        cal_dev = 1 / (1 + np.exp(-(a * np.log(np.clip(s_dev, 1e-9, 1 - 1e-9) /
                                                (1 - np.clip(s_dev, 1e-9, 1 - 1e-9))) + b)))
        out["platt_a"], out["platt_b"] = a, b
        out["brier_dev_platt"] = float(np.mean((cal_dev - y_dev) ** 2))
        if len(y_oos):
            cal_oos = 1 / (1 + np.exp(-(a * np.log(np.clip(s_oos, 1e-9, 1 - 1e-9) /
                                                    (1 - np.clip(s_oos, 1e-9, 1 - 1e-9))) + b)))
            out["brier_oos_platt"] = float(np.mean((cal_oos - y_oos) ** 2))
    return out
