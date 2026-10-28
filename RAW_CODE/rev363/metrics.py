"""REV363 evaluation metrics and exact threshold-family search."""
from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import roc_auc_score


def _safe_div(a, b):
    return float(a) / float(b) if b else None


def confusion_binary(y, pred):
    y = np.asarray(y, int)
    pred = np.asarray(pred, int)
    tp = int(((y == 1) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    return {"TP": tp, "FN": fn, "FP": fp, "TN": tn,
            "TPR": _safe_div(tp, tp + fn), "FNR": _safe_div(fn, tp + fn),
            "FPR": _safe_div(fp, fp + tn), "TNR": _safe_div(tn, fp + tn),
            "precision": _safe_div(tp, tp + fp), "recall": _safe_div(tp, tp + fn),
            "accuracy": _safe_div(tp + tn, tp + tn + fp + fn)}


def confusion4(y, pred, labels=("N", "D", "G", "BOTH")):
    y = np.asarray(y, int)
    pred = np.asarray(pred, int)
    mat = np.zeros((4, 4), int)
    for a, b in zip(y, pred):
        mat[a, b] += 1
    per = {}
    for i, name in enumerate(labels):
        tp = int(mat[i, i])
        fn = int(mat[i, :].sum() - tp)
        fp = int(mat[:, i].sum() - tp)
        per[name] = {"TP": tp, "FN": fn, "FP": fp, "precision": _safe_div(tp, tp + fp),
                     "recall": _safe_div(tp, tp + fn)}
    return {"matrix": mat.tolist(), "labels": list(labels), "per_class": per,
            "stress_vs_N": confusion_binary((y != 0).astype(int), (pred != 0).astype(int))}


def multiclass_logloss(y, p):
    y = np.asarray(y, int)
    p = np.clip(np.asarray(p, float), 1e-15, 1.0)
    return float(-np.mean(np.log(p[np.arange(len(y)), y])))


def multiclass_brier(y, p):
    y = np.asarray(y, int)
    oh = np.eye(p.shape[1])[y]
    return float(np.mean(np.sum((p - oh) ** 2, axis=1)))


def auc_score(y, score):
    y = np.asarray(y, int)
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, score))


def wilson_ci(k, n, z=1.959963984540054):
    if n == 0:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, centre - half), min(1.0, centre + half)]


def block_bootstrap_metrics(y, pred, groups, n_boot=1000, seed=363):
    """Resample whole country blocks with replacement (country dependence)."""
    y = np.asarray(y, int)
    pred = np.asarray(pred, int)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    fnrs, fprs = [], []
    for _ in range(n_boot):
        take = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in take])
        c = confusion_binary(y[idx], pred[idx])
        if c["FNR"] is not None:
            fnrs.append(c["FNR"])
        if c["FPR"] is not None:
            fprs.append(c["FPR"])
    def q(a):
        return [float(np.quantile(a, .025)), float(np.quantile(a, .975))] if a else [None, None]
    return {"FNR_ci95": q(fnrs), "FPR_ci95": q(fprs),
            "n_boot": n_boot, "unit": "COUNTRY_BLOCK"}


def f1_f5(y4, p, score, model, baseline_p=None, *, n_masks=1, edf=None):
    y4 = np.asarray(y4, int)
    ll = multiclass_logloss(y4, p)
    br = multiclass_brier(y4, p)
    auc = auc_score((y4 != 0).astype(int), score)
    n = len(y4)
    k = edf if edf is not None else p.shape[1] * 3
    # Selection-aware penalty: BIC/2 complexity plus log(number of masks)/n.
    penalty = (k / (2.0 * max(n, 1))) * math.log(max(n, 2)) + math.log(max(n_masks, 1)) / max(n, 1)
    f4 = ll + penalty
    f5 = None
    if baseline_p is not None:
        f5 = {"delta_logloss_vs_baseline": ll - multiclass_logloss(y4, baseline_p),
              "delta_brier_vs_baseline": br - multiclass_brier(y4, baseline_p)}
    return {"F1_logloss": ll, "F2_brier": br, "F3_auc_stress": auc,
            "F4_selection_aware": f4, "F4_penalty": penalty,
            "F5_incremental": f5, "n": n, "edf_assumed": k, "n_masks": n_masks}


def interval_score(y_true, lower, upper, alpha=0.05):
    """Winkler/interval score; lower/upper are the issued zone bounds."""
    if lower is None or upper is None or y_true is None:
        return None
    w = upper - lower
    s = w
    if y_true < lower:
        s += (2.0 / alpha) * (lower - y_true)
    elif y_true > upper:
        s += (2.0 / alpha) * (y_true - upper)
    return float(s)
