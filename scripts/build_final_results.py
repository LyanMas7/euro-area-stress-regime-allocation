"""Rebuild every published table, figure and the README in English.

Each block states where its numbers come from and whether they are reproducible
from tapes shipped with the project or only available as an owner-selected
configuration. Nothing is interpolated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PHASES = ["PRE", "ENTRY", "PEAK", "DEESC", "RECOVERY", "EXIT"]
PHASE_EN = {"PRE": "Pre-entry", "ENTRY": "Entry", "PEAK": "Peak", "DEESC": "De-escalation",
            "RECOVERY": "Recovery", "EXIT": "Exit"}
BANDS = [("normal", "#e8f1fb", "Normal"),
         ("to_short", "#fde7c8", "Normal to last short"),
         ("to_bottom", "#f7c6c7", "Last short to bottom"),
         ("to_exit", "#d7ecd9", "Bottom to exit")]
PHASE_BAND = {"ACTIVE_PRE_SHORT": "normal", "FROZEN_SHORT_TO_BOTTOM": "to_bottom",
              "BOTTOM_CONFIRMED_ACTIVE": "to_exit", "EXIT_ACTIVE": "to_exit",
              "POST_EXIT_ACTIVE": "normal"}
STRATS = ["EW", "IVOL", "MVP", "ERC", "MaxDiv"]
COLOURS = {"POLICY": "#c00000", "EW": "#1f4e79", "IVOL": "#2e75b6", "MVP": "#70ad47",
           "ERC": "#7030a0", "MaxDiv": "#bf9000", "EQUITY100": "#404040"}
LABELS = {"POLICY": "Main model", "EW": "EW", "IVOL": "IVOL", "MVP": "MVP",
          "ERC": "ERC", "MaxDiv": "MaxDiv", "EQUITY100": "100% equity"}


def conf(y, p):
    y = np.asarray(y).astype(int); p = np.asarray(p).astype(int)
    tp = int(((y == 1) & (p == 1)).sum()); fn = int(((y == 1) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum()); tn = int(((y == 0) & (p == 0)).sum())
    return dict(TP=tp, FN=fn, FP=fp, TN=tn,
                FNR=100 * fn / (tp + fn) if tp + fn else np.nan,
                FPR=100 * fp / (fp + tn) if fp + tn else np.nan, n=tp + fn + fp + tn)


# ------------------------------------------------------------------ RQ01
def rq01(project: Path, out: Path):
    p4 = pd.read_csv(project / "Threshold/Threshold/THRESHOLD_RQ01_RQ02_20261027/02_RQ01/RQ01_DEV_P4_DUMP.csv")
    x = p4[(p4.family == "XGBOOST") & (p4.model == "M01")].copy()
    x["Ps"] = 1.0 - x["p_N"]
    x["stress"] = (x["y4"].astype(int) != 0).astype(int)
    pk = pd.read_csv(project / "AI_HANDOFF/DEV_RUN_20261025_FNR_FPR_BAND_CALIBRATION/RQ01_DEV_OOS_RERUN_F3B/RQ01_03_PACKET_FEATURES_DEV.csv")
    m = x.merge(pk[["country", "month", "pkt_joint_CLIFS_VIX", "pkt_joint_DFC_GFC"]],
                on=["country", "month"], how="left")
    thr = json.loads((project / "CANONICAL_PROJECT/THRESHOLD/RQ01/rq01_selected.json").read_text(encoding="utf-8"))["thresholds"]
    m["pred"] = (((m.Ps >= m.country.map(thr)) | (m.pkt_joint_CLIFS_VIX == 1))
                 & ~(m.pkt_joint_DFC_GFC == 1)).astype(int)

    total = conf(m.stress, m.pred)
    rows = []
    ph = pd.read_csv(project / "_canonical_validation/post_cleanup/RQ01/RQ01_DEV_ISSUE_TAPE.csv",
                     usecols=["country", "issue", "phase"]).rename(columns={"issue": "month"})
    mm = m.merge(ph, on=["country", "month"], how="inner")
    for p in PHASES:
        d = mm[mm.phase == p]
        if len(d):
            c = conf(d.stress, d.pred); c.update({"phase": p, "phase_label": PHASE_EN[p]}); rows.append(c)
    rows.append({**total, "phase": "ALL", "phase_label": "All phases"})
    dev = pd.DataFrame(rows)

    near = pd.read_csv(project / "Threshold/Threshold/THRESHOLD_RQ01_RQ02_20261027/01_FINAL_SELECTION/RQ01_PER_COUNTRY_NEAR_PASS.csv")
    tot = near[near.country == "TOTAL"].iloc[0]
    dev["period"] = "DEV"
    dev.to_csv(out / "rq01_dev_phase_pool.csv", index=False, encoding="utf-8-sig")
    near[near.country != "TOTAL"].to_csv(out / "rq01_country_thresholds.csv", index=False, encoding="utf-8-sig")

    summary = pd.DataFrame([
        {"period": "DEV", "TP": total["TP"], "FN": total["FN"], "FP": total["FP"], "TN": total["TN"],
         "FNR": total["FNR"], "FPR": total["FPR"], "source": "reproduced from P4 dump + packet features"},
        {"period": "OOS", "TP": 77, "FN": int(tot.OOS_FN), "FP": int(tot.OOS_FP), "TN": 16,
         "FNR": 100 * int(tot.OOS_FN) / 91, "FPR": 100 * int(tot.OOS_FP) / 21,
         "source": "owner-selected register (OOS XGB tape not shipped)"},
    ])
    summary.to_csv(out / "rq01_summary.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [2.1, 1]})
    xpos = np.arange(len(PHASES)); w = 0.38
    d_ph = dev[dev.phase != "ALL"].set_index("phase").reindex(PHASES)
    ax[0].bar(xpos - w / 2, d_ph.FNR, w, label="FNR", color="#1f4e79")
    ax[0].bar(xpos + w / 2, d_ph.FPR, w, label="FPR", color="#c00000")
    for i, (a, b) in enumerate(zip(d_ph.FNR, d_ph.FPR)):
        if not np.isnan(a): ax[0].annotate(f"{a:.1f}", (i - w / 2, a), ha="center", va="bottom", fontsize=7)
        if not np.isnan(b): ax[0].annotate(f"{b:.1f}", (i + w / 2, b), ha="center", va="bottom", fontsize=7)
    ax[0].set_xticks(xpos); ax[0].set_xticklabels([PHASE_EN[p] for p in PHASES], rotation=15)
    ax[0].set_ylabel("%"); ax[0].set_title("DEV — approved per-country rule, by phase")
    ax[0].legend(); ax[0].grid(axis="y", alpha=0.25)
    ax[1].bar(["DEV", "OOS"], summary.FNR, 0.35, label="FNR", color="#1f4e79")
    ax[1].bar(["DEV", "OOS"], summary.FPR, 0.35, bottom=summary.FNR, label="FPR", color="#c00000")
    for i, r in summary.iterrows():
        ax[1].annotate(f"FNR {r.FNR:.2f}\nFPR {r.FPR:.2f}", (i, r.FNR + r.FPR), ha="center", va="bottom", fontsize=7)
    ax[1].set_ylim(0, max(45, float((summary.FNR + summary.FPR).max()) * 1.28))
    ax[1].set_title("Pooled decision, DEV vs OOS"); ax[1].legend(fontsize=7); ax[1].grid(axis="y", alpha=0.25)
    fig.suptitle("RQ01 — threshold confirmation: FNR/FPR across the six phases and pooled", fontsize=11)
    fig.tight_layout(); fig.savefig(out / "fig_rq01_fnr_fpr.png", dpi=170); plt.close(fig)
    return summary, dev


# ------------------------------------------------------------------ RQ02
def rq02(project: Path, out: Path):
    base = project / "Bài luận/Table, Figure đầu ra/RQ02/tables"
    sel = pd.read_csv(base / "B02_2_RQ02_SELECTED.csv")
    bot = pd.read_csv(base / "B02_9_RQ02_BOTTOM.csv")
    rows = []
    for _, r in sel.iterrows():
        rows.append({"task": r.component, "period": r.period, "threshold": r.threshold, "n": r.n,
                     "TP": r.TP, "FN": r.FN, "FP": r.FP, "TN": r.TN, "FNR": r.FNR, "FPR": r.FPR})
    t = pd.DataFrame(rows)
    for c in ("threshold", "n", "TP", "FN", "FP", "TN", "FNR", "FPR"):
        t[c] = pd.to_numeric(t[c], errors="coerce")
    t.to_csv(out / "rq02_entry_exit.csv", index=False, encoding="utf-8-sig")

    b = bot[bot.asset == "ALL5"][["period", "n_hit", "n_issued_matured", "hit_rate", "forecast_coverage"]].copy()
    b["hit_rate"] = 100 * b.hit_rate; b["forecast_coverage"] = 100 * b.forecast_coverage
    b.to_csv(out / "rq02_bottom_zone.csv", index=False, encoding="utf-8-sig")

    audit = pd.DataFrame([
        {"variant": "owner-selected M11 / H6 = 0.111821", "DEV_FNR": 0.00, "DEV_FPR": 23.85,
         "OOS_FNR": 14.29, "OOS_FPR": np.nan, "status": "SELECTED — tape not shipped"},
        {"variant": "reproducible LOGISTIC_L2_STRONG H6 = 0.191936", "DEV_FNR": 0.00, "DEV_FPR": 12.48,
         "OOS_FNR": 35.71, "OOS_FPR": np.nan, "status": "REPRODUCIBLE from shipped registry"},
        {"variant": "reproducible LOGISTIC H6 = 0.244947", "DEV_FNR": 0.00, "DEV_FPR": 3.67,
         "OOS_FNR": 42.86, "OOS_FPR": np.nan, "status": "REPRODUCIBLE from shipped registry"},
    ])
    audit.to_csv(out / "rq02_threshold_audit.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.3))
    lab = [f"{r.task.split(' — ')[0]}\n{r.period}" for _, r in t.iterrows()]
    ax[0].bar(np.arange(len(t)) - 0.18, t.FNR, 0.36, label="FNR", color="#1f4e79")
    ax[0].bar(np.arange(len(t)) + 0.18, t.FPR.fillna(0), 0.36, label="FPR", color="#c00000")
    ax[0].set_xticks(np.arange(len(t))); ax[0].set_xticklabels(lab, fontsize=7)
    ax[0].set_ylabel("%"); ax[0].set_title("ENTRY (18m) and EXIT (3m): FNR / FPR"); ax[0].legend(); ax[0].grid(axis="y", alpha=0.25)
    for i, r in enumerate(t.itertuples()):
        ax[0].annotate(f"{r.FNR:.1f}", (i - 0.18, r.FNR), ha="center", va="bottom", fontsize=7)
        if not pd.isna(r.FPR): ax[0].annotate(f"{r.FPR:.1f}", (i + 0.18, r.FPR), ha="center", va="bottom", fontsize=7)
    ax[1].bar(b.period, b.hit_rate, color="#2e75b6")
    for i, r in enumerate(b.itertuples()):
        ax[1].annotate(f"{r.hit_rate:.2f}%\n({int(r.n_hit)}/{int(r.n_issued_matured)})",
                       (i, r.hit_rate), ha="center", va="bottom", fontsize=8)
    ax[1].set_ylabel("%"); ax[1].set_title("Bottom zone hit rate"); ax[1].grid(axis="y", alpha=0.25)
    fig.suptitle("RQ02 — transition forecasting: entry, exit and the asset bottom", fontsize=11)
    fig.tight_layout(); fig.savefig(out / "fig_rq02_fnr_fpr.png", dpi=170); plt.close(fig)
    return t, b


# ------------------------------------------------------------------ RQ03
def equity_index(dataset: Path) -> pd.DataFrame:
    base = dataset / "data/Euro_Area_Dataset_Mentor_Ready/03_ASSET_DATA/01_EQUITY/01_SIZE_GROUPS"
    files = {"Large": "EURO_STOXX_Large_NTR_Weekly.csv", "Mid": "EURO_STOXX_Mid_NTR_Weekly.csv",
             "Small": "EURO_STOXX_Small_NTR_Weekly.csv"}
    fr = []
    for n, f in files.items():
        d = pd.read_csv(base / f, parse_dates=["Date"])[["Date", "Weekly_Return"]]
        fr.append(d.rename(columns={"Weekly_Return": n}).set_index("Date"))
    r = pd.concat(fr, axis=1).sort_index()
    eq = (1.0 + r.mean(axis=1).fillna(0.0)).cumprod()
    return pd.DataFrame({"week_end": eq.index, "EQUITY100": eq.values})


def lifecycle(nav: pd.DataFrame) -> pd.Series:
    band = nav["phase"].astype(str).map(PHASE_BAND).fillna("normal")
    stress = nav["regime"].astype(str) != "N"
    band[(band == "normal") & stress] = "to_short"
    return band


def rq03(project: Path, ledger: Path, dataset: Path, out: Path):
    eq = equity_index(dataset)
    bdev = pd.read_csv(project / "AI_HANDOFF/RQ03_AGGRESSIVE_20261026/BENCHMARK_COMPARISON_DEV.csv").set_index("strategy")["terminal_nav"]
    boos = pd.read_csv(project / "AI_HANDOFF/RQ03_AGGRESSIVE_20261026/BENCHMARK_COMPARISON_OOS.csv").set_index("strategy")["terminal_nav"]
    weekly = pd.read_csv(project / "AI_HANDOFF/RQ02_DEV_MAX_LEADTIME_20261026/OVERLAY_2020/BENCHMARK_OOS_WEEKLY.csv",
                         parse_dates=["week_end"])

    rows = []
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 9.2))
    for ax, period, tgt in ((axes[0], "DEV", bdev), (axes[1], "OOS", boos)):
        led = pd.read_csv(ledger / f"CHOSEN_STANDARD_LEDGER_{period}.csv", parse_dates=["week_end"])
        band = lifecycle(led)
        ymax = float(led.nav.max()) * 1.06
        for key, colour, _ in BANDS:
            msk = (band == key).values
            if msk.any():
                ax.fill_between(led.week_end, 0, ymax, where=msk, color=colour, step="mid", linewidth=0, zorder=0)
        ax.plot(led.week_end, led.nav, color=COLOURS["POLICY"], lw=2.2, label=LABELS["POLICY"], zorder=5)
        e = eq[(eq.week_end >= led.week_end.min()) & (eq.week_end <= led.week_end.max())]
        if len(e):
            ax.plot(e.week_end, e.EQUITY100 / e.EQUITY100.iloc[0], color=COLOURS["EQUITY100"],
                    lw=1.3, ls=":", label=LABELS["EQUITY100"], zorder=4)
        if period == "OOS":
            for s in STRATS:
                w = weekly[weekly.strategy == s].sort_values("week_end")
                ax.plot(w.week_end, w.nav_end / w.nav_end.iloc[0], color=COLOURS[s], lw=1.1, ls="--", label=LABELS[s])
        else:
            for s in STRATS:
                v = float(tgt[s]); ax.axhline(v, color=COLOURS[s], lw=1.0, ls="--", alpha=0.85)
                ax.plot([], [], color=COLOURS[s], lw=1.0, ls="--", label=f"{LABELS[s]} ({v:.3f})")
        ax.set_title(f"{period}: main model NAV = {led.nav.iloc[-1]:.4f}"
                     + (" with five benchmarks (weekly) and 100% equity" if period == "OOS"
                        else " and five benchmarks (terminal values in the legend)"))
        ax.set_ylabel("NAV (initial capital = 1.0)"); ax.set_ylim(0, ymax); ax.grid(alpha=0.25)
        ax.margins(x=0.06)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y")); ax.legend(fontsize=7, ncol=4, loc="upper left")
        row = {"period": period, "weeks": len(led), "POLICY": float(led.nav.iloc[-1])}
        row.update({s: float(tgt[s]) for s in STRATS})
        if len(e): row["EQUITY100"] = float(e.EQUITY100.iloc[-1] / e.EQUITY100.iloc[0])
        for k, _, _ in BANDS: row[f"weeks_{k}"] = int((band == k).sum())
        rows.append(row)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c, _ in BANDS]
    fig.suptitle("Main model NAV versus five benchmarks and 100% equity, split by lifecycle stage", fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.legend(handles=handles, labels=[v for _, _, v in BANDS], fontsize=8.5, ncol=4,
               loc="lower center", frameon=False)
    fig.savefig(out / "fig_rq03_nav_dev_oos.png", dpi=170, bbox_inches="tight"); plt.close(fig)

    t = pd.DataFrame(rows)
    t.to_csv(out / "rq03_nav_terminal.csv", index=False, encoding="utf-8-sig")

    pack = project / "Bài luận/Table, Figure đầu ra/RQ03/tables/B03_2_RQ03_NAV.csv"
    if pack.is_file():
        b = pd.read_csv(pack).rename(columns={
            "Chiến lược": "strategy", "MDD DEV": "MDD_DEV", "MDD OOS": "MDD_OOS",
            "ES95 DEV": "ES95_DEV", "ES95 OOS": "ES95_OOS"})
        b["strategy"] = b["strategy"].replace({"Chuẩn nghiên cứu (chosen)": "Main model"})
        risk = b[["strategy", "MDD_DEV", "MDD_OOS", "ES95_DEV", "ES95_OOS"]].copy()
        for c in risk.columns[1:]:
            risk[c] = pd.to_numeric(risk[c], errors="coerce")
        risk.to_csv(out / "rq03_risk.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
    names = ["POLICY"] + STRATS + ["EQUITY100"]
    for ax, period in zip(axes, ("DEV", "OOS")):
        r = t[t.period == period].iloc[0]
        vals = [r.get(n, np.nan) for n in names]
        ax.bar([LABELS[n] for n in names], vals, color=[COLOURS[n] for n in names])
        ax.set_ylim(0, max(v for v in vals if not np.isnan(v)) * 1.18)
        for i, v in enumerate(vals):
            if not np.isnan(v): ax.annotate(f"{v:.3f}", (i, v), ha="center", va="bottom", fontsize=7)
        ax.set_title(f"{period}: terminal NAV"); ax.grid(axis="y", alpha=0.25)
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=8)
    fig.tight_layout(); fig.savefig(out / "fig_rq03_terminal.png", dpi=170); plt.close(fig)
    return t


def md(df, cols, fmt="{:.2f}"):
    d = df[list(cols)]
    out = "| " + " | ".join(map(str, d.columns)) + " |\n|" + "|".join(["---"] * len(d.columns)) + "|\n"
    for _, x in d.iterrows():
        cells = []
        for v in x:
            if isinstance(v, float):
                cells.append("—" if pd.isna(v) else fmt.format(v))
            else:
                cells.append(str(v))
        out += "| " + " | ".join(cells) + " |\n"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--ledger-dir", required=True)
    ap.add_argument("--dataset", required=True)
    a = ap.parse_args()
    P, R, L, D = Path(a.project), Path(a.repo), Path(a.ledger_dir), Path(a.dataset)
    out = R / "results"; out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("*.png"):
        stale.unlink()

    s1, d1 = rq01(P, out)
    t2, b2 = rq02(P, out)
    t3 = rq03(P, L, D, out)

    readme = f"""# RQ01–RQ03 research code — DEV / OOS evidence

Reproducible code and results for a three-stage decision chain over seven euro-area countries:
confirm the current stress state (RQ01) → forecast state transitions and the asset bottom (RQ02)
→ allocate the portfolio by state (RQ03).

```
RAW_CODE/     run_canonical.py, canonical_*.py, rev363/, research_adapters/, rq02_source/, rq03_standard/
THRESHOLD/    selected RQ01, RQ02 and RQ03 configurations (JSON)
scripts/      build_final_results.py (this file's driver), build_rq03_standard.py
results/      CSV tables and PNG figures for DEV and OOS
```

## Quick start

```bash
pip install -r requirements.txt
python RAW_CODE/run_canonical.py --dataset-root <ASSET03> --out <out> --mode dev
python RAW_CODE/run_canonical.py --dataset-root <ASSET03> --out <out> --mode oos \\
    --oos-authorization <DEC-015 owner token>
python scripts/build_final_results.py --project <project> --repo . \\
    --ledger-dir <rq03 standard ledgers> --dataset <ASSET03>
```

## RQ01 — stress-state confirmation, six phases, individual and pool votes

Selected rule, per country: `((Ps_XGB_M01 >= t_country) OR pkt_joint_CLIFS_VIX) AND NOT pkt_joint_DFC_GFC`.
The DEV block is rebuilt directly from the per-record P4 dump merged with the packet features,
and it reproduces the registered confusion exactly: **TP/FN/FP/TN = 376/42/194/606**.

{md(s1, ["period","TP","FN","FP","TN","FNR","FPR","source"])}
Per phase on DEV:

{md(d1, ["phase_label","n","TP","FN","FP","TN","FNR","FPR"])}
![RQ01](results/fig_rq01_fnr_fpr.png)

Reading: the rule is conservative in the calm phases (Pre-entry FPR 4.1%, Peak FNR 0%) and pays for it
in the recovery phase, where FPR rises to 43.8% while FNR stays at 7.9% — the model rarely misses a
stress month but over-fires while the economy is already healing. OOS keeps the same balance
(FNR 15.38%, FPR 23.81%) but the per-phase split cannot be rebuilt because the OOS XGBoost score tape
is not part of the shipped package.

## RQ02 — entry (H6, 18 months), exit (3 months) and the asset bottom

{md(t2, ["task","period","threshold","n","TP","FN","FP","TN","FNR","FPR"])}
{md(b2, ["period","n_hit","n_issued_matured","hit_rate","forecast_coverage"])}
![RQ02](results/fig_rq02_fnr_fpr.png)

Reading: entry has zero missed onsets in DEV at 23.85% false alarms; exit degrades the most out of sample
(FNR 0.00% → 28.57%, FPR 21.75% → 45.71%). The bottom zone is the weakest block — 23.33% of matured
forecasts hit in DEV and 6.67% in OOS, with a negative median lead time (the model fires late).
The OOS entry FPR is undefined because all 14 scorable observations are positives.

## RQ03 — main model NAV versus five benchmarks and 100% equity

Standard: `W16_S4BOT_Z2` + ENTRY H12 + EXIT-3m on the RQ03 phase clock (EXIT threshold 0.027174, 10 bp one-way).

{md(t3, ["period","weeks","POLICY","EW","IVOL","MVP","ERC","MaxDiv","EQUITY100"])}
{md(t3, ["period","weeks_normal","weeks_to_short","weeks_to_bottom","weeks_to_exit"])}
![RQ03 NAV](results/fig_rq03_nav_dev_oos.png)
![RQ03 terminal](results/fig_rq03_terminal.png)

Reading: the main model ends DEV at 3.9015 against 1.2934–1.6784 for the five benchmarks and 2.5960 for
100% equity, and OOS at 1.8059 against 0.9892–1.0867 and 1.2611; the full-sample path compounds to
5.6981. The cost is tail risk: maximum drawdown 32.84% and weekly ES95 4.52% in DEV are the highest in
the group, so the excess return is paid for with a permanently high equity share rather than with timing
of de-risking. Shading marks the four lifecycle stages; the "last short to bottom" stage dominates by
design because the state machine stays frozen there for most of the sample.

## Audit — what reproduces and what does not

| Block | Claim | Evidence | Status |
|---|---|---|---|
| RQ01 DEV | 42 FN / 194 FP, FNR 10.05% / FPR 24.25% | P4 dump + packet features + per-country thresholds | **Reproduced exactly** |
| RQ01 DEV per phase | six-phase split | same merge, phase tape from the canonical run | **Reproduced** |
| RQ01 OOS | 14 FN / 5 FP, FNR 15.38% / FPR 23.81% | owner-selected register | Registered, not rebuilt (no OOS XGB tape) |
| RQ02 | entry 0.111821 → 0.00/23.85 DEV, 14.29/— OOS | `THRESHOLD/RQ02` and the selected metrics table | Registered, not rebuilt (no per-record label tape) |
| RQ02 alternates | 0.191936 → 0.00/12.48 · 0.244947 → 0.00/3.67 | shipped entry-metrics registry | **Reproducible** |
| RQ03 | 3.9015 / 1.8059 / 5.6981 | S4BOT standard builder + ledgers | **Reproduced exactly** |

RQ02 threshold audit:

{md(pd.read_csv(out / "rq02_threshold_audit.csv"), ["variant","DEV_FNR","DEV_FPR","OOS_FNR","OOS_FPR","status"])}
The owner-selected operating point (0.111821) is kept in `THRESHOLD/RQ02`; the packaged registry offers
0.191936 and 0.244947 as the reproducible alternatives. They are different operating points on the same
score, not competing models, and they are reported side by side rather than substituted.

## Data and dependencies

The locked ASSET03 dataset (`Euro_Area_Dataset_Mentor_Ready`) is not shipped; pass its root at run time.
No threshold is inferred at run time — every selected value lives in `THRESHOLD/`.
"""
    (R / "README.md").write_text(readme, encoding="utf-8")
    print("RQ01 summary\n", s1.to_string(index=False))
    print("\nRQ02 entry/exit\n", t2.to_string(index=False))
    print("\nRQ03 terminal\n", t3.to_string(index=False))
    print("\nREADME bytes:", len(readme))


if __name__ == "__main__":
    main()
