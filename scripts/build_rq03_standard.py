"""RQ03 standard figure: owner-selected S4BOT NAV versus five benchmarks and 100% equity.

Inputs (all external to this repo, passed on the command line):
  --ledger-dir    CHOSEN_STANDARD_LEDGER_{DEV,OOS}.csv produced by the standard builder
  --handoff       AI_HANDOFF tree holding the benchmark comparison tables and OOS weekly tape
  --dataset       locked ASSET03 dataset root (equity indices live under 03_ASSET_DATA/01_EQUITY)
  --out           results directory
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BANDS = [("normal", "#e8f1fb", "Bình thường"),
         ("to_short", "#fde7c8", "Bình thường → Last short"),
         ("to_bottom", "#f7c6c7", "Last short → Vùng đáy"),
         ("to_exit", "#d7ecd9", "Vùng đáy → Thoát")]

# ledger phase -> lifecycle band
PHASE_BAND = {
    "ACTIVE_PRE_SHORT": "normal",
    "FROZEN_SHORT_TO_BOTTOM": "to_bottom",
    "BOTTOM_CONFIRMED_ACTIVE": "to_exit",
    "EXIT_ACTIVE": "to_exit",
    "POST_EXIT_ACTIVE": "normal",
}
PHASE_VI = {"ACTIVE_PRE_SHORT": "Bình thường (trước short)", "FROZEN_SHORT_TO_BOTTOM": "Last short → vùng đáy",
            "BOTTOM_CONFIRMED_ACTIVE": "Vùng đáy → phục hồi", "EXIT_ACTIVE": "Thoát",
            "POST_EXIT_ACTIVE": "Bình thường sau exit"}

STRATS = ["EW", "IVOL", "MVP", "ERC", "MaxDiv"]
COLOURS = {"POLICY": "#c00000", "EW": "#1f4e79", "IVOL": "#2e75b6", "MVP": "#70ad47",
           "ERC": "#7030a0", "MaxDiv": "#bf9000", "EQUITY100": "#404040"}
LABELS = {"POLICY": "Mô hình chính (chuẩn S4BOT)", "EW": "EW", "IVOL": "IVOL", "MVP": "MVP",
          "ERC": "ERC", "MaxDiv": "MaxDiv", "EQUITY100": "100% cổ phiếu"}


def equity_index(dataset: Path) -> pd.DataFrame:
    base = dataset / "data/Euro_Area_Dataset_Mentor_Ready/03_ASSET_DATA/01_EQUITY/01_SIZE_GROUPS"
    files = {"Large": "EURO_STOXX_Large_NTR_Weekly.csv",
             "Mid": "EURO_STOXX_Mid_NTR_Weekly.csv",
             "Small": "EURO_STOXX_Small_NTR_Weekly.csv"}
    frames = []
    for name, fn in files.items():
        d = pd.read_csv(base / fn, parse_dates=["Date"])[["Date", "Weekly_Return"]]
        d = d.rename(columns={"Weekly_Return": name}).set_index("Date")
        frames.append(d)
    ret = pd.concat(frames, axis=1).sort_index()
    eq = (1.0 + ret.mean(axis=1).fillna(0.0)).cumprod()
    return pd.DataFrame({"week_end": eq.index, "EQUITY100": eq.values})


def lifecycle_bands(nav: pd.DataFrame) -> pd.Series:
    ph = nav["phase"].astype(str).map(PHASE_BAND).fillna("normal")
    band = ph.copy()
    # the run-up to the last short: stress weeks still logged as pre-short
    stress = nav["regime"].astype(str) != "N"
    band[(band == "normal") & stress] = "to_short"
    return band


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger-dir", required=True)
    ap.add_argument("--handoff", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    L, H, D, out = Path(a.ledger_dir), Path(a.handoff), Path(a.dataset), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    eq = equity_index(D)
    bench_dev = pd.read_csv(H / "RQ03_AGGRESSIVE_20261026/BENCHMARK_COMPARISON_DEV.csv")
    bench_oos = pd.read_csv(H / "RQ03_AGGRESSIVE_20261026/BENCHMARK_COMPARISON_OOS.csv")
    weekly = pd.read_csv(H / "RQ02_DEV_MAX_LEADTIME_20261026/OVERLAY_2020/BENCHMARK_OOS_WEEKLY.csv",
                         parse_dates=["week_end"])

    rows, fig_ax = [], {}
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 9.4))
    for ax, period in zip(axes, ("DEV", "OOS")):
        led = pd.read_csv(L / f"CHOSEN_STANDARD_LEDGER_{period}.csv", parse_dates=["week_end"])
        band = lifecycle_bands(led)
        ymax = float(led["nav"].max()) * 1.06
        for key, colour, _ in BANDS:
            m = (band == key).values
            if m.any():
                ax.fill_between(led["week_end"], 0, ymax, where=m, color=colour,
                                step="mid", linewidth=0, zorder=0)
        ax.plot(led["week_end"], led["nav"], color=COLOURS["POLICY"], lw=2.2,
                label=LABELS["POLICY"], zorder=5)

        e = eq[(eq.week_end >= led.week_end.min()) & (eq.week_end <= led.week_end.max())]
        if len(e):
            base = float(e["EQUITY100"].iloc[0])
            ax.plot(e.week_end, e["EQUITY100"] / base, color=COLOURS["EQUITY100"], lw=1.3,
                    ls=":", label=LABELS["EQUITY100"], zorder=4)

        if period == "OOS":
            for s in STRATS:
                w = weekly[(weekly.strategy == s)].sort_values("week_end")
                if len(w):
                    ax.plot(w.week_end, w.nav_end / float(w.nav_end.iloc[0]),
                            color=COLOURS[s], lw=1.1, ls="--", label=LABELS[s], zorder=3)
            tgt = bench_oos.set_index("strategy")["terminal_nav"]
        else:
            for s in STRATS:
                v = float(bench_dev[bench_dev.strategy == s]["terminal_nav"].iloc[0])
                ax.axhline(v, color=COLOURS[s], lw=1.0, ls="--", alpha=0.85, zorder=3)
                ax.annotate(f"{s} {v:.3f}", xy=(led.week_end.iloc[-1], v),
                            xytext=(4, 0), textcoords="offset points", fontsize=7,
                            color=COLOURS[s], va="center")
            tgt = bench_dev.set_index("strategy")["terminal_nav"]

        ax.set_title(f"{period}: NAV chuẩn S4BOT ({period} = {led['nav'].iloc[-1]:.4f}) so với năm đối chứng"
                     + (" (đường theo tuần)" if period == "OOS" else " (mức cuối kỳ, nét ngang)"))
        ax.set_ylabel("NAV (vốn ban đầu = 1,0)")
        ax.set_ylim(0, ymax)
        ax.grid(alpha=0.25)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.legend(fontsize=7, ncol=4, loc="upper left")

        row = {"period": period, "weeks": len(led), "POLICY": float(led["nav"].iloc[-1])}
        row.update({s: float(tgt.get(s, np.nan)) for s in STRATS})
        if len(e):
            row["EQUITY100"] = float(e["EQUITY100"].iloc[-1] / e["EQUITY100"].iloc[0])
        for key, _, _ in BANDS:
            row[f"weeks_{key}"] = int((band == key).sum())
        rows.append(row)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c, _ in BANDS]
    axes[0].legend(handles=handles, labels=[v for _, _, v in BANDS], fontsize=7,
                   loc="lower right", title="Pha vòng đời")
    fig.suptitle("RQ03 — NAV của mô hình chính (chuẩn S4BOT) so với năm đối chứng và 100% cổ phiếu",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "fig_rq03_nav_dev_oos.png", dpi=170)
    plt.close(fig)

    tab = pd.DataFrame(rows)
    tab.to_csv(out / "rq03_nav_terminal.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    names = ["POLICY"] + STRATS + ["EQUITY100"]
    for ax, period in zip(axes, ("DEV", "OOS")):
        r = tab[tab.period == period]
        if not len(r):
            continue
        r = r.iloc[0]
        vals = [r.get(n, np.nan) for n in names]
        ax.bar([LABELS[n] for n in names], vals, color=[COLOURS[n] for n in names])
        for i, v in enumerate(vals):
            if not np.isnan(v):
                ax.annotate(f"{v:.3f}", (i, v), ha="center", va="bottom", fontsize=7)
        ax.set_title(f"{period}: NAV cuối kỳ")
        ax.grid(axis="y", alpha=0.25)
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_rq03_terminal.png", dpi=170)
    plt.close(fig)

    print(tab.to_string(index=False))


if __name__ == "__main__":
    main()
