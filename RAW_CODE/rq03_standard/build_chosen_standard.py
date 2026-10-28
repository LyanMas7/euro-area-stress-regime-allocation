"""Build the owner-selected NAV standard.

Standard: W16_S4BOT_Z2 + H12 entry (outside RQ03 short/bottom phases)
          + EXIT-3m (approved tape, threshold 0.027174) on the RQ03 phase clock.
Outputs DEV/OOS/FULL ledgers, Pn/Ps change tables, metrics, recurrence check.
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("RQ_HANDOFF_ROOT") or (Path(__file__).resolve().parents[2].parent / "AI_HANDOFF"))
RQ03 = Path(__file__).resolve().parent / "w16"
sys.path.insert(0, str(RQ03))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_wave16 as W16  # noqa: E402
import overlay_rq02_exit3_approved as O  # noqa: E402

OUT = Path(os.environ.get("RQ_OUT") or Path(__file__).resolve().parent)
PHASES = {"ACTIVE_PRE_SHORT", "FROZEN_SHORT_TO_BOTTOM", "BOTTOM_CONFIRMED_ACTIVE", "EXIT_ACTIVE"}
TAG = "RQ03_NAV_H12_ENTRY_EXIT3_PHASE"


def build_ctx(base, pe, p12):
    pm = np.isin(np.asarray(base["phase"], dtype=object), list(PHASES))
    pk = O.rebuild(base["pk"], np.where(pm, 1, 0), pe, p12, "both")
    ctx = dict(base); ctx["pk"] = pk
    return ctx, pm


def pnps_table(base, ctx, pe, p12, period):
    d = base["m"][["week_end", "week_start", "regime"]].copy()
    d["phase"] = base["phase"]; d["st"] = base["st"]
    d["p_exit_3m_ea7"] = pe; d["h12_p12m_ea7"] = p12
    d["P_S_baseline"] = base["pk"]["P_S"].to_numpy(float)
    d["P_S_new"] = ctx["pk"]["P_S"].to_numpy(float)
    d["P_N_baseline"] = base["pk"]["P_N"].to_numpy(float)
    d["P_N_new"] = ctx["pk"]["P_N"].to_numpy(float)
    d["delta_P_S"] = d.P_S_new - d.P_S_baseline
    d["period"] = period
    return d


def nav_check(ctx, ledger):
    R = ctx["R"]; k = W16.W1.B.E.fee_vector(10.0)
    W = ledger[["w_E1", "w_E2", "w_E3", "w_B1", "w_B2", "w_C1"]].to_numpy(float)
    pre = ctx["PRE0"].copy(); nav_prev = 1.0; mx = 0.0
    for t in range(len(ledger)):
        target = W[t]
        c, _ = W16.W1.B.E.trade_cost(pre, target, k)
        calc = nav_prev * (1.0 - c) * (1.0 + float(target @ R[t]))
        mx = max(mx, abs(calc - float(ledger.nav.iloc[t])))
        pre = target * (1.0 + R[t]); pre = pre / pre.sum()
        nav_prev = float(ledger.nav.iloc[t])
    return mx


def main():
    ex = O.exit_monthly(); h12 = O.h12_monthly()
    rows = []; ctxs = {}; changes = []
    for period in ("DEV", "OOS"):
        base, R_full, offset, theta, _ = W16.RB.context(period)
        pe, p12, _ = O.weekly_series(base["m"], ex, h12)
        ctx, pm = build_ctx(base, pe, p12)
        met, path = O.run(ctx, R_full, offset, theta, TAG, period)
        met["period"] = period
        path.to_csv(OUT / f"CHOSEN_STANDARD_LEDGER_{period}.csv", index=False)
        tab = pnps_table(base, ctx, pe, p12, period)
        changes.append(tab)
        met["pnps_changed_weeks"] = int((tab.delta_P_S.abs() > 1e-9).sum())
        met["phase_weeks"] = int(pm.sum())
        met["nav_recurrence_max_diff"] = nav_check(ctx, path)
        rows.append(met)
        ctxs[period] = (base, ctx, pe, p12, pm)
        print(period, "NAV", round(met["terminal_nav"], 4), "MDD", round(met["mdd_full"], 4),
              "ES95", round(met["es95"], 4), "changed", met["pnps_changed_weeks"],
              "recurrence", met["nav_recurrence_max_diff"], flush=True)
    # FULL
    (bD, cD, peD, p12D, pmD) = ctxs["DEV"]; (bO, cO, peO, p12O, pmO) = ctxs["OOS"]
    ctx = {
        "m": pd.concat([bD["m"], bO["m"]], ignore_index=True),
        "W": np.vstack([bD["W"], bO["W"]]), "R": np.vstack([bD["R"], bO["R"]]),
        "PRE0": bD["PRE0"], "pk": pd.concat([cD["pk"], cO["pk"]], ignore_index=True),
        "st": np.r_[bD["st"], bO["st"]], "phase": np.r_[bD["phase"], bO["phase"]],
        "ev": {k: (None if v is None else np.concatenate([bD["ev"][k], bO["ev"][k]], axis=0))
               for k, v in bD["ev"].items()},
        "mu": np.concatenate([bD["mu"], bO["mu"]], axis=0),
        "Sg": np.concatenate([bD["Sg"], bO["Sg"]], axis=0),
        "trend13": np.r_[bD["trend13"], bO["trend13"]],
        "gate3f": np.r_[bD["gate3f"], bO["gate3f"]],
        "feats": pd.concat([bD["feats"], bO["feats"]], ignore_index=True),
    }
    dev_len = len(bD["m"])
    met, path = O.run(ctx, ctx["R"], 0, theta, TAG, "FULL")
    met["period"] = "FULL"; met["oos_segment_return"] = float(path.nav.iloc[-1] / path.nav.iloc[dev_len - 1])
    met["pnps_changed_weeks"] = None
    met["nav_recurrence_max_diff"] = nav_check(ctx, path)
    rows.append(met)
    path.to_csv(OUT / "CHOSEN_STANDARD_LEDGER_FULL.csv", index=False)
    pd.concat(changes, ignore_index=True).to_csv(OUT / "CHOSEN_STANDARD_PNPS.csv", index=False)
    df = pd.DataFrame(rows)
    cols = ["period", "terminal_nav", "mdd_full", "es95", "avg_equity", "pnps_changed_weeks",
            "phase_weeks", "nav_recurrence_max_diff", "oos_segment_return"]
    df[ [c for c in cols if c in df.columns] ].to_csv(OUT / "CHOSEN_STANDARD_METRICS.csv", index=False)
    print("FULL", "NAV", round(met["terminal_nav"], 4), "seg", round(met["oos_segment_return"], 4),
          "recurrence", met["nav_recurrence_max_diff"], flush=True)


if __name__ == "__main__":
    main()
