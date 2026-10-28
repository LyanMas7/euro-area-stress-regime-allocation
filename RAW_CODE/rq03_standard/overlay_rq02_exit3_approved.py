"""RQ02 EXIT-3m applied to the RQ03 Pn/Ps interface.

Correction to the previous overlays:
  * ENTRY (normal weeks) uses H12 onset with the 12m -> weekly hazard conversion.
  * EXIT (stress weeks) uses the approved EXIT-3m tape, not H12 and not a stale
    carry-forward.  The approved threshold 0.027174 is applied as the exit gate.
  * A record is fresh for 13 weeks only; after that the baseline exit head is used.
  * 3m cumulative exit probability -> weekly P_N: 1-(1-p)^(1/13).

The approved EXIT-3m tape covers 2020-03..2021-03 (91 country-months); there is
no issued EXIT tape for 2022-2024, so those weeks keep the baseline head.
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("RQ_HANDOFF_ROOT") or (Path(__file__).resolve().parents[3] / "AI_HANDOFF"))
RQ03 = ROOT / "AI_HANDOFF" / "RQ03_AGGRESSIVE_20261026"
sys.path.insert(0, str(RQ03))
import run_wave16 as W16  # noqa: E402
OUT = Path(os.environ.get("RQ_OUT") or (Path(__file__).resolve().parents[3] / "results/rq03_standard"))

EXIT_TAPE = ROOT / "AI_HANDOFF/OOS_FIRST_REDEVELOPMENT_20261027/EXIT_PNPS_FINAL_OOS_TAPE.csv"
H12_TAPE = ROOT / "AI_HANDOFF/DEV_RUN_20261025_FNR_FPR_BAND_CALIBRATION/RQ02_FULL_V3_F3B/RQ02_03/RQ02_03_ISSUED_P_TAPE.csv"
H12_COL = "p_on__H12__M10__LOGISTIC_L2_STRONG"
EXIT_THR = 0.027174


def exit_monthly():
    d = pd.read_csv(EXIT_TAPE)
    d = d[d.horizon == 3].copy()
    d["issue"] = pd.to_datetime(d["issue"]).dt.to_period("M").dt.to_timestamp("M")
    d["avail"] = d["issue"] + pd.offsets.MonthEnd(0)
    return d


def h12_monthly():
    d = pd.read_csv(H12_TAPE, usecols=["country", "month", H12_COL])
    d["month"] = pd.to_datetime(d["month"]).dt.to_period("M").dt.to_timestamp("M")
    g = d.groupby("month")[H12_COL].mean().reset_index()
    g["avail"] = g["month"] + pd.offsets.MonthEnd(0)
    return g.sort_values("avail")


def weekly_series(m, ex, h12):
    n = len(m)
    pe = np.full(n, np.nan)
    p12 = np.full(n, np.nan)
    signal = np.zeros(n, bool)
    for i, ws in enumerate(pd.to_datetime(m["week_start"])):
        sub = ex[ex.avail < ws]
        if len(sub):
            latest = sub.sort_values("avail").groupby("country").tail(1)
            latest = latest[(ws - latest.avail).dt.days <= 91]
            if len(latest):
                pe[i] = float(latest.p_exit_CV.mean())
                signal[i] = pe[i] >= EXIT_THR
        s12 = h12[h12.avail < ws]
        if len(s12):
            p12[i] = float(s12[H12_COL].iloc[-1])
    return pe, p12, signal


def rebuild(pk, st, pe, p12, mode="both"):
    pk = pk.copy().reset_index(drop=True)
    ps_old = pk["P_S"].to_numpy(float)
    comp = pk[["p_D", "p_G", "p_BOTH"]].to_numpy(float)
    s_old = comp.sum(axis=1)
    ps = ps_old.copy()
    stress = st > 0
    normal = st == 0
    if mode in ("both", "exit"):
        haz = 1.0 - (1.0 - np.clip(pe, 0, 1)) ** (1.0 / 13.0)
        use = stress & np.isfinite(haz) & np.isfinite(ps_old)
        ps[use] = 1.0 - np.clip(haz[use], 0.0, 1.0)
    if mode in ("both", "entry"):
        h12haz = 1.0 - (1.0 - np.clip(p12, 0, 1)) ** (1.0 / 52.0)
        use = normal & np.isfinite(h12haz) & np.isfinite(ps_old)
        ps[use] = np.clip(h12haz[use], 0.0, 1.0)
    scale = np.where(s_old > 1e-12, ps / np.where(s_old > 1e-12, s_old, 1.0), 0.0)
    new = comp * scale[:, None]
    zero = (s_old <= 1e-12) & (ps > 1e-12)
    if zero.any():
        new[zero] = np.outer(ps[zero], np.array([0.0, 1 / 3, 2 / 3]))
    pk["P_S"] = ps; pk["P_N"] = 1.0 - ps; pk["p_N"] = 1.0 - ps
    pk["p_D"] = new[:, 0]; pk["p_G"] = new[:, 1]; pk["p_BOTH"] = new[:, 2]
    return pk


def run(ctx, R_full, offset, theta, tag, period):
    p = W16.case_set()["W16_S4BOT_Z2"]
    return W16.run_one(ctx, R_full, offset, theta, tag, p, period)


def main():
    ex = exit_monthly(); h12 = h12_monthly()
    rows = []; ctxs = {}
    for period in ("DEV", "OOS"):
        base, R_full, offset, theta, _ = W16.RB.context(period)
        pe, p12, sig = weekly_series(base["m"], ex, h12)
        print(period, "exit records fresh weeks", int(np.isfinite(pe).sum()),
              "signals", int(sig.sum()), "p_exit mean", round(float(np.nanmean(pe)), 4) if np.isfinite(pe).any() else None,
              flush=True)
        for mode in ("exit", "both", "entry"):
            ctx = dict(base); ctx["pk"] = rebuild(base["pk"], base["st"], pe, p12, mode)
            met, path = run(ctx, R_full, offset, theta, f"EXIT3_{mode}", period)
            rows.append(dict(period=period, mode=mode, nav=met["terminal_nav"],
                             mdd=met["mdd_full"], es95=met["es95"],
                             avg_equity=float(path.equity.mean()),
                             ps_mean=float(np.nanmean(ctx["pk"]["P_S"]))))
            print(period, mode, round(met["terminal_nav"], 4),
                  "eq", round(float(path.equity.mean()), 4),
                  "P_S mean", round(float(np.nanmean(ctx["pk"]["P_S"])), 4), flush=True)
            if mode == "both":
                ctxs[period] = (ctx, pe, p12)
        ctxs.setdefault(period, None)
    # FULL for the combined mode
    (ctxD, peD, p12D) = ctxs["DEV"]; (ctxO, peO, p12O) = ctxs["OOS"]
    ctx = {
        "m": pd.concat([ctxD["m"], ctxO["m"]], ignore_index=True),
        "W": np.vstack([ctxD["W"], ctxO["W"]]), "R": np.vstack([ctxD["R"], ctxO["R"]]),
        "PRE0": ctxD["PRE0"], "pk": pd.concat([ctxD["pk"], ctxO["pk"]], ignore_index=True),
        "st": np.r_[ctxD["st"], ctxO["st"]], "phase": np.r_[ctxD["phase"], ctxO["phase"]],
        "ev": {k: (None if v is None else np.concatenate([ctxD["ev"][k], ctxO["ev"][k]], axis=0))
               for k, v in ctxD["ev"].items()},
        "mu": np.concatenate([ctxD["mu"], ctxO["mu"]], axis=0),
        "Sg": np.concatenate([ctxD["Sg"], ctxO["Sg"]], axis=0),
        "trend13": np.r_[ctxD["trend13"], ctxO["trend13"]],
        "gate3f": np.r_[ctxD["gate3f"], ctxO["gate3f"]],
        "feats": pd.concat([ctxD["feats"], ctxO["feats"]], ignore_index=True),
    }
    dev_len = len(ctxD["m"])
    met, path = run(ctx, ctx["R"], 0, theta, "EXIT3_both", "FULL")
    met["period"] = "FULL"; met["oos_segment_return"] = float(path.nav.iloc[-1] / path.nav.iloc[dev_len - 1])
    rows.append(dict(period="FULL", mode="both", nav=met["terminal_nav"], mdd=met["mdd_full"],
                     es95=met["es95"], avg_equity=float(path.equity.mean()),
                     ps_mean=float(np.nanmean(ctx["pk"]["P_S"])),
                     oos_segment=met["oos_segment_return"]))
    path.to_csv(OUT / "RQ02_EXIT3_APPROVED_LEDGER_FULL.csv", index=False)
    print("FULL both", round(met["terminal_nav"], 4), "seg", round(met["oos_segment_return"], 4), flush=True)
    pd.DataFrame(rows).to_csv(OUT / "RQ02_EXIT3_APPROVED_NAV.csv", index=False)


if __name__ == "__main__":
    main()
