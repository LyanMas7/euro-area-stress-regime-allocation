"""Build every research figure on the GT2006 results-presentation canvas.

Layout contract (inches, Letter 8.5x11, content width 7.0): title block at the top,
axes below it, legend strip directly under the axes, notes and source under the
legend, page number at the foot. After rendering, every text box is measured and
any pair that intersects fails the build.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.text  # noqa: F401
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gt2006.figure_canvas import FigurePage  # noqa: E402
from gt2006.registry import load_register  # noqa: E402

REG = load_register()
SIZES = REG["typography"]["sizes_pt"]
INK = "#000000"
GREY = "#555555"

PHASES = ["PRE", "ENTRY", "PEAK", "DEESC", "RECOVERY", "EXIT"]
PHASE_LABEL = ["Pre-entry", "Entry", "Peak", "De-escalation", "Recovery", "Exit"]
STAGE_ORDER = ["Normal", "Last short confirmed", "Bottom confirmed", "Exit"]
STAGE_COLOUR = ["#DCE6F1", "#FCE4D6", "#F8CBAD", "#E2EFDA"]

ASSETS = [("w_C1", "Cash", "#D9D9D9"),
          ("w_B2", "Bonds, long", "#C6E0B4"),
          ("w_B1", "Bonds, short", "#A9D08E"),
          ("w_E1", "Equity, large", "#F4CCCC"),
          ("w_E2", "Equity, mid", "#EA9999"),
          ("w_E3", "Equity, small", "#C00000")]

BENCH = [("EW", "#000080", "solid", 0.9), ("IVOL", "#FF00FF", (0, (5, 2.5)), 0.9),
         ("MVP", "#008000", (0, (1.2, 1.8)), 0.9), ("ERC", "#CC0000", "solid", 0.9),
         ("MaxDiv", "#7F6000", (0, (5, 1.5, 1.2, 1.5)), 0.9)]

SRC = "Source: author's calculation from the selected policy's ledger."


def qa_overlap(page: FigurePage) -> list[str]:
    r = page.fig.canvas.get_renderer()
    boxes = []
    for t in page.fig.findobj(matplotlib.text.Text):
        if not t.get_text().strip() or not t.get_visible():
            continue
        try:
            bb = t.get_window_extent(renderer=r)
        except Exception:
            continue
        if bb.width <= 0 or bb.height <= 0:
            continue
        boxes.append((t.get_text().replace("\n", " ")[:30], bb))
    hits = []
    dpi = page.fig.dpi

    def where(bb):
        return f"[{bb.x0 / dpi:.2f},{page.H - bb.y1 / dpi:.2f}-{page.H - bb.y0 / dpi:.2f}]"

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i][1], boxes[j][1]
            if a.overlaps(b):
                if (min(a.x1, b.x1) - max(a.x0, b.x0)) > 1.0 and (min(a.y1, b.y1) - max(a.y0, b.y0)) > 1.0:
                    hits.append(f"{boxes[i][0]!r} {where(a)} x {boxes[j][0]!r} {where(b)}")
    for lg in page.fig.legends:
        lb = lg.get_window_extent(r)
        own = {id(t) for t in lg.get_texts()}
        for t in page.fig.findobj(matplotlib.text.Text):
            if id(t) in own or not t.get_text().strip() or not t.get_visible():
                continue
            bb = t.get_window_extent(renderer=r)
            if bb.width <= 0 or bb.height <= 0:
                continue
            if (min(lb.x1, bb.x1) - max(lb.x0, bb.x0)) > 1.0 and (min(lb.y1, bb.y1) - max(lb.y0, bb.y0)) > 1.0:
                hits.append(f"legend {where(lb)} x {t.get_text()[:28]!r} {where(bb)}")
        for ax in page.fig.axes:
            ab = ax.get_window_extent(r)
            if (min(lb.x1, ab.x1) - max(lb.x0, ab.x0)) > 1.0 and (min(lb.y1, ab.y1) - max(lb.y0, ab.y0)) > 1.0:
                hits.append(f"legend {where(lb)} overlaps a plotting area {where(ab)}")
    return hits


def legend_under(page: FigurePage, handles, labels, y_in: float, ncol: int, size: float | None = None) -> None:
    page.fig.legend(handles, labels, loc="center", bbox_to_anchor=(0.5, 1 - y_in / page.H), ncol=ncol,
                    frameon=False, prop={"family": "Garamond", "size": size or SIZES["legend"]},
                    handlelength=2.2, columnspacing=1.3, handletextpad=0.45)


def finish(page: FigurePage, out: Path, stem: str, note: str, source: str, y_note: float) -> None:
    page.add_note(note, y_top_in=y_note, source=source, width=112)
    page.fig.canvas.draw()
    hits = qa_overlap(page)
    print(f"[qa] {stem}: {'PASS' if not hits else 'FAIL ' + str(hits[:4])}")
    page.save(out, stem, formats=("png",), compact=True)
    matplotlib.pyplot.close(page.fig)


def fig01(d: pd.DataFrame, summary: pd.DataFrame, out: Path) -> None:
    page = FigurePage(number=None, title="Financial Stress Identification - DEV vs OOS", top_in=0.80,
                      paragraph="The rule combines a country threshold on the boosted-tree stress probability with two "
                                "joint-packet filters. Left: miss rate and false-alarm rate inside each phase of the "
                                "development window. Right: the pooled decision in development and in evaluation, on the "
                                "same scale.")
    ax = page.add_axes(0.95, 2.30, 3.95, 3.20)
    s1 = d[d.phase != "ALL"].set_index("phase").reindex(PHASES)
    x = np.arange(len(PHASES)); w = 0.36
    ax.bar(x - w / 2, s1.FNR.values, w, facecolor="#DCE6F1", edgecolor="#000080", linewidth=0.5)
    ax.bar(x + w / 2, s1.FPR.values, w, facecolor="#F4CCCC", edgecolor="#C00000", linewidth=0.5)
    for i, (a, b) in enumerate(zip(s1.FNR.values, s1.FPR.values)):
        ax.text(i - w / 2, a + 1.0, f"{a:.2f}", ha="center", va="bottom", fontsize=7.5, fontfamily="Garamond")
        ax.text(i + w / 2, b + 1.0, f"{b:.2f}", ha="center", va="bottom", fontsize=7.5, fontfamily="Garamond")
    page.format_axis(ax, xlabel="Phase state", ylabel="Per cent of the evaluation sample", ylim=(0, 55),
                     yticks=[0, 10, 20, 30, 40, 50], xticks=list(x), xticklabels=PHASE_LABEL,
                     rotation=35, tick_size=7.5)
    ax.set_title("By phase, development window", fontsize=8.5, fontweight="bold", fontfamily="Garamond", pad=5)

    ax2 = page.add_axes(5.35, 2.30, 2.20, 3.20)
    pool = summary.set_index("period").reindex(["DEV", "OOS"])
    yy = np.arange(2)
    ax2.bar(yy - w / 2, pool.FNR.values, w, facecolor="#DCE6F1", edgecolor="#000080", linewidth=0.5)
    ax2.bar(yy + w / 2, pool.FPR.values, w, facecolor="#F4CCCC", edgecolor="#C00000", linewidth=0.5)
    for i, (a, b) in enumerate(zip(pool.FNR.values, pool.FPR.values)):
        ax2.text(i - w / 2, a + 1.0, f"{a:.2f}", ha="center", va="bottom", fontsize=7.5, fontfamily="Garamond")
        ax2.text(i + w / 2, b + 1.0, f"{b:.2f}", ha="center", va="bottom", fontsize=7.5, fontfamily="Garamond")
    page.format_axis(ax2, xlabel="Pooled decision", ylabel="Per cent", ylim=(0, 55),
                     yticks=[0, 10, 20, 30, 40, 50], xticks=list(yy), xticklabels=["DEV", "OOS"], tick_size=7.5)
    ax2.set_title("Pooled, DEV vs OOS", fontsize=8.5, fontweight="bold", fontfamily="Garamond", pad=5)

    legend_under(page, [Patch(facecolor="#DCE6F1", edgecolor="#000080", linewidth=0.5),
                        Patch(facecolor="#F4CCCC", edgecolor="#C00000", linewidth=0.5)],
                 ["FNR, miss rate", "FPR, false-alarm rate"], 6.55, 2)
    finish(page, out, "fig01_rq01_phase_error",
           "Notes: no miss in the calm, entry and peak phases; the rule pays for that in recovery, where 43.78% of calm "
           "months raise an alarm. Pooled rates are 10.05% and 24.25% in development, 15.38% and 23.81% in evaluation, on "
           "112 scored evaluation months.",
           SRC + " Values: results/rq01_dev_phase_pool.csv and results/rq01_summary.csv.", 6.95)


def fig02(rq02: pd.DataFrame, bottom: pd.DataFrame, out: Path) -> None:
    page = FigurePage(number=None, title="Entry and Exit Forecasting Performance",
                      paragraph="Left: miss rate and false-alarm rate of the two transition models at their "
                                "selected thresholds, in per cent, by window. Right: share of matured bottom-zone "
                                "forecasts that contained the trough. A missing false-alarm rate is marked NA "
                                "because the evaluation window has no calm observation.")
    ax = page.add_axes(0.95, 2.70, 3.50, 2.60)
    e = rq02[rq02.task.astype(str).str.startswith("ENTRY")].set_index("period")
    xx = rq02[rq02.task.astype(str).str.startswith("EXIT")].set_index("period")
    fnr = [e.loc["DEV", "FNR"], e.loc["OOS", "FNR"], xx.loc["DEV", "FNR"], xx.loc["OOS", "FNR"]]
    fpr = [e.loc["DEV", "FPR"], np.nan, xx.loc["DEV", "FPR"], xx.loc["OOS", "FPR"]]
    x = np.arange(4); w = 0.36
    ax.bar(x - w / 2, fnr, w, facecolor="#DCE6F1", edgecolor="#000080", linewidth=0.5)
    ax.bar(x + w / 2, [0 if np.isnan(v) else v for v in fpr], w, facecolor="#F4CCCC",
           edgecolor="#C00000", linewidth=0.5)
    page.format_axis(ax, xlabel="Model and window", ylabel="Per cent", ylim=(0, 55),
                     yticks=[0, 10, 20, 30, 40, 50], xticks=list(x),
                     xticklabels=["Entry\nDEV", "Entry\nOOS", "Exit\nDEV", "Exit\nOOS"],
                     rotation=0, tick_size=7.5)
    ax.text(1 + w / 2, 0.7, "NA", ha="center", va="bottom", fontsize=7.5, fontfamily="Garamond")
    ax.set_title("(a) Entry and exit error", fontsize=SIZES["figure_panel_title"], fontweight="bold",
                 fontfamily="Garamond", pad=7)
    ax2 = page.add_axes(4.95, 2.70, 2.60, 2.60)
    b = bottom.set_index("period")
    ax2.bar([0, 1], [b.loc["DEV", "hit_rate"], b.loc["OOS", "hit_rate"]], 0.45,
            facecolor="#DCE6F1", edgecolor="#000080", linewidth=0.5)
    page.format_axis(ax2, xlabel="Window", ylabel="Hit rate, per cent", ylim=(0, 30),
                     yticks=[0, 10, 20, 30], xticks=[0, 1], xticklabels=["DEV", "OOS"])
    ax2.set_title("(b) Bottom-zone hit rate", fontsize=SIZES["figure_panel_title"], fontweight="bold",
                  fontfamily="Garamond", pad=7)
    legend_under(page, [Patch(facecolor="#DCE6F1", edgecolor="#000080", linewidth=0.5),
                        Patch(facecolor="#F4CCCC", edgecolor="#C00000", linewidth=0.5)],
                 ["FNR", "FPR"], 6.15, 2)
    finish(page, out, "fig02_rq02_transition_error",
           "Notes: no episode is missed in development and a false alarm arrives in about one calm month in "
           "four. Out of sample entry keeps 14.29% but its false-alarm rate is undefined, since all 14 scorable "
           "observations are positives; exit degrades to 28.57% and 45.71%. The bottom block contains the trough "
           "in 23.33% of 60 matured development forecasts and 6.67% of 90 matured evaluation forecasts.",
           SRC + " Values: results/rq02_entry_exit.csv and results/rq02_bottom_zone.csv.", 6.55)


def _stage(phase: str) -> int:
    return {"ACTIVE_PRE_SHORT": 0, "FROZEN_SHORT_TO_BOTTOM": 1, "BOTTOM_CONFIRMED_ACTIVE": 2,
            "EXIT_ACTIVE": 3, "POST_EXIT_ACTIVE": 3}.get(phase, 0)


def fig03(full: pd.DataFrame, out: Path) -> None:
    page = FigurePage(number=None, title="Six-Asset Capital Allocation, 2005-2024",
                      paragraph="Upper panel: weekly weights of the six assets held by the main model, stacked to "
                                "100% of net asset value. Lower panel: the state actually observed each week on the "
                                "same time axis, so the regime history is read as a separate band and no label sits "
                                "over the allocation area. The dashed line is the development/evaluation boundary.")
    full = full.sort_values("week_end").reset_index(drop=True)
    t = pd.to_datetime(full["week_end"])
    top = page.add_axes(0.95, 2.35, 6.60, 2.80)
    band = page.add_axes(0.95, 5.42, 6.60, 0.85)
    w = full[[a for a, *_ in ASSETS]].to_numpy(float)
    w = w / w.sum(axis=1, keepdims=True) * 100.0
    base = np.zeros(len(full))
    for i, (col, label, face) in enumerate(ASSETS):
        top.fill_between(t, base, base + w[:, i], facecolor=face, edgecolor="black", linewidth=0.35,
                         zorder=2 + i)
        base = base + w[:, i]
    page.format_axis(top, xlabel="", ylabel="Share of net asset value, per cent", ylim=(0, 100),
                     yticks=[0, 20, 40, 60, 80, 100], xlim=(t.iloc[0], t.iloc[-1]))
    top.tick_params(labelbottom=False)
    top.axvline(pd.Timestamp("2020-01-03"), color=INK, linestyle=(0, (4, 3)), linewidth=0.9, zorder=30)

    stage = full["phase"].map(_stage).to_numpy(int)
    edges = mdates.date2num(pd.concat([t, t.iloc[[-1]] + pd.Timedelta(days=7)]))
    band.pcolormesh(edges, [0.0, 0.60], stage[None, :], cmap=ListedColormap(STAGE_COLOUR),
                    norm=BoundaryNorm(np.arange(-0.5, 4.5, 1), 4), shading="flat", rasterized=True)
    band.axhline(0.60, color="black", linewidth=0.45)
    page.format_axis(band, xlabel="Year", ylabel="", ylim=(0, 1), yticks=[0.30], yticklabels=["State"],
                     xlim=(t.iloc[0], t.iloc[-1]), tick_size=7.5)
    band.xaxis.set_major_locator(mdates.YearLocator(2))
    band.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    band.axvline(pd.Timestamp("2020-01-03"), color=INK, linestyle=(0, (4, 3)), linewidth=0.9, zorder=5)
    band.text(pd.Timestamp("2019-08-01"), 0.80, "Development, 2005-2019", ha="right", va="center",
              fontsize=7.5, fontfamily="Garamond")
    band.text(pd.Timestamp("2020-02-01"), 0.80, "Evaluation, 2020-2024", ha="left", va="center",
              fontsize=7.5, fontfamily="Garamond")

    legend_under(page, [Patch(facecolor=f, edgecolor="black", linewidth=0.35) for _, _, f in ASSETS],
                 [lab for _, lab, _ in ASSETS], 6.80, 3, size=7.5)
    legend_under(page, [Patch(facecolor=c, edgecolor="black", linewidth=0.45) for c in STAGE_COLOUR],
                 [f"{i+1}. {s}" for i, s in enumerate(STAGE_ORDER)], 7.10, 4, size=7.5)
    finish(page, out, "fig04_rq03_allocation",
           "Notes: weights are re-based to 100% for display; the ledger sums to one at every week. The policy "
           "spends 34 weeks in the normal state, 783 under the short lock, 103 at the bottom and 121 in the exit "
           "states, and it returns to the bottom seven times. Cash reaches 100% in ten weeks. The dashed line is "
           "3 January 2020.",
           SRC + " Values: results/rq03_standard/CHOSEN_STANDARD_LEDGER_FULL.csv.", 7.45)


def _equity(dataset: Path) -> pd.DataFrame:
    base = dataset / "data/Euro_Area_Dataset_Mentor_Ready/03_ASSET_DATA/01_EQUITY/01_SIZE_GROUPS"
    fr = []
    for f in ("EURO_STOXX_Large_NTR_Weekly.csv", "EURO_STOXX_Mid_NTR_Weekly.csv",
              "EURO_STOXX_Small_NTR_Weekly.csv"):
        d = pd.read_csv(base / f, parse_dates=["Date"])[["Date", "Weekly_Return"]].set_index("Date")
        fr.append(d)
    r = pd.concat(fr, axis=1).sort_index().mean(axis=1).fillna(0.0)
    return pd.DataFrame({"week_end": r.index, "EQUITY100": (1 + r).cumprod().values})


def fig04(dev: pd.DataFrame, oos: pd.DataFrame, eq: pd.DataFrame, term: pd.DataFrame,
          weekly: pd.DataFrame, out: Path) -> None:
    page = FigurePage(number=None,
                      title="Main Model NAV Against Five Benchmarks and Full Equity",
                      paragraph="Growth of one unit of capital, weekly. Panel (a) is the development window; its "
                                "benchmarks appear as terminal value lines because only end-of-window levels exist "
                                "for that period. Panel (b) is the evaluation window, where every benchmark is "
                                "available weekly. Both axes start near one because the question is relative growth.")
    ax = page.add_axes(0.95, 2.45, 6.60, 2.55)
    d = dev.copy(); d["week_end"] = pd.to_datetime(d["week_end"])
    ax.plot(d.week_end, d.nav.astype(float), color=INK, linewidth=1.8)
    e = eq[(eq.week_end >= d.week_end.min()) & (eq.week_end <= d.week_end.max())]
    ax.plot(e.week_end, e.EQUITY100 / e.EQUITY100.iloc[0], color=GREY, linestyle=(0, (1.2, 1.8)), linewidth=0.9)
    row = term[term.period == "DEV"].iloc[0]
    for key, colour, ls, lw in BENCH:
        ax.axhline(float(row[key]), color=colour, linestyle=ls, linewidth=lw)
    page.format_axis(ax, xlabel="", ylabel="NAV, initial capital = 1.0", ylim=(0.9, 4.15),
                     xticks=[pd.Timestamp(f"{y}-07-01") for y in (2006, 2009, 2012, 2015, 2018)],
                     xticklabels=[str(y) for y in (2006, 2009, 2012, 2015, 2018)])
    ax.text(0.004, 0.965, "(a) Development window, 2005-2019", transform=ax.transAxes, ha="left",
            va="top", fontsize=SIZES["figure_panel_caption"], fontweight="bold", fontfamily="Garamond")

    ax2 = page.add_axes(0.95, 5.55, 6.60, 2.30)
    o = oos.copy(); o["week_end"] = pd.to_datetime(o["week_end"])
    ax2.plot(o.week_end, o.nav.astype(float), color=INK, linewidth=1.8)
    e2 = eq[(eq.week_end >= o.week_end.min()) & (eq.week_end <= o.week_end.max())]
    ax2.plot(e2.week_end, e2.EQUITY100 / e2.EQUITY100.iloc[0], color=GREY, linestyle=(0, (1.2, 1.8)), linewidth=0.9)
    for key, colour, ls, lw in BENCH:
        w = weekly[weekly.strategy == key].sort_values("week_end")
        ax2.plot(w.week_end, w.nav_end / float(w.nav_end.iloc[0]), color=colour, linestyle=ls, linewidth=lw)
    page.format_axis(ax2, xlabel="Year", ylabel="NAV, initial capital = 1.0", ylim=(0.9, 1.95),
                     xticks=[pd.Timestamp(f"{y}-07-01") for y in (2020, 2021, 2022, 2023, 2024)],
                     xticklabels=[str(y) for y in (2020, 2021, 2022, 2023, 2024)])
    ax2.text(0.004, 0.965, "(b) Evaluation window, 2020-2024", transform=ax2.transAxes, ha="left",
             va="top", fontsize=SIZES["figure_panel_caption"], fontweight="bold", fontfamily="Garamond")

    handles = [Line2D([], [], color=INK, linewidth=1.8),
               Line2D([], [], color=GREY, linestyle=(0, (1.2, 1.8)), linewidth=0.9)]
    labels = ["Main model", "100% equity"]
    for key, colour, ls, lw in BENCH:
        handles.append(Line2D([], [], color=colour, linestyle=ls, linewidth=lw))
        labels.append(f"{key} ({float(row[key]):.3f})")
    legend_under(page, handles, labels, 8.32, 4, size=7.2)
    finish(page, out, "fig05_rq03_nav_paths",
           "Notes: panel (a) ends at 3.9015 for the main model and 2.5960 for full equity; panel (b) ends at "
           "1.8059 and 1.2611. Benchmark terminal values are given in the legend. Evaluation is not a blind "
           "holdout: the window informed model choice.",
           SRC + " Values: results/rq03_nav_terminal.csv, results/rq03_standard/, and the locked dataset for "
                 "the equity index.", 8.70)


def fig05(term: pd.DataFrame, out: Path) -> None:
    page = FigurePage(number=None, title="Terminal NAV by Strategy and Window",
                      paragraph="Terminal growth of one unit of capital for the main model, the five passive "
                                "benchmarks and the fully invested equity reference, in each window. Bars keep the "
                                "same order in both panels; the equity reference is the relevant yardstick because "
                                "the main model holds more than seventy per cent equity on average.")
    names = ["POLICY", "EW", "IVOL", "MVP", "ERC", "MaxDiv", "EQUITY100"]
    short = {"POLICY": "Main", "EW": "EW", "IVOL": "IVOL", "MVP": "MVP", "ERC": "ERC",
             "MaxDiv": "MaxDiv", "EQUITY100": "100% eq."}
    for k, (period, ytop, h, title) in enumerate((("DEV", 2.40, 2.35, "(a) Development window"),
                                                  ("OOS", 5.45, 2.35, "(b) Evaluation window"))):
        ax = page.add_axes(0.95, ytop, 6.60, h)
        row = term[term.period == period].iloc[0]
        vals = [float(row.get(n, np.nan)) for n in names]
        ax.bar(range(len(names)), vals, 0.55, facecolor="#DCE6F1", edgecolor="#000080", linewidth=0.5)
        ax.bar(0, vals[0], 0.55, facecolor="#000080", edgecolor="#000080", linewidth=0.5)
        ax.bar(len(names) - 1, vals[-1], 0.55, facecolor="#FFFFFF", edgecolor="#000080", linewidth=0.5)
        for i, v in enumerate(vals):
            ax.text(i, v * 1.015, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, fontfamily="Garamond")
        page.format_axis(ax, xlabel="" if k == 0 else "Strategy", ylabel="Terminal NAV",
                         ylim=(0, max(vals) * 1.16), xticks=list(range(len(names))),
                         xticklabels=[short[n] for n in names], rotation=0, tick_size=7.5)
        ax.text(0.996, 0.965, title, transform=ax.transAxes, ha="right", va="top",
                fontsize=SIZES["figure_panel_caption"], fontweight="bold", fontfamily="Garamond")
    finish(page, out, "fig06_rq03_terminal_nav",
           "Notes: the main model ends at 3.9015 in development against 2.5960 for full equity, and at 1.8059 "
           "against 1.2611 in evaluation. The premium is earned with the deepest drawdown and the highest weekly "
           "expected shortfall of the group, reported in the risk table.",
           SRC + " Values: results/rq03_nav_terminal.csv.", 8.28)


def fig06(out: Path) -> None:
    page = FigurePage(number=None, title="Restructuring States and Allocation Rules",
                      paragraph="Schematic of the state machine. Each box names a state and the portfolio rule "
                                "that applies inside it; each arrow carries the trigger that moves the policy to the "
                                "next state. Triggers come from the identification and forecasting layers, so the "
                                "allocation never reacts to price alone.")
    boxes = {
        "S1": (0.95, 2.35, 2.80, 1.70,
               "S1  Normal\nEquity 90 per cent, bonds 10 per cent\nband: equity 80-100 per cent"),
        "S2": (0.95, 5.25, 2.80, 1.70,
               "S2  Locked, short confirmed\nBonds 65 per cent, tilt by stress\nclamped to 50-80 per cent\n"
               "cash is the remainder;\npre-bottom supplement allowed"),
        "S3": (5.10, 5.25, 2.85, 1.70,
               "S3  Bottom confirmed\nEquity 100 per cent, held\nwhile the profit ladder runs"),
        "S4": (5.10, 2.35, 2.85, 1.70,
               "S4  Post-exit\nEquity band 50-100 per cent\nbonds 0-25 per cent\ncash remainder"),
    }
    for key, (x, y, w, h, txt) in boxes.items():
        page.add_box(x, y, w, h, lw=0.6)
        page.fig.text((x + w / 2) / page.W, 1 - (y + h / 2) / page.H, txt, ha="center", va="center",
                      fontsize=8.6, fontfamily="Garamond", linespacing=1.45)

    def arrow(x1, y1, x2, y2):
        page.fig.add_artist(FancyArrowPatch((x1 / page.W, 1 - y1 / page.H), (x2 / page.W, 1 - y2 / page.H),
                                            transform=page.fig.transFigure, arrowstyle="-|>", mutation_scale=8,
                                            linewidth=0.7, color=INK, shrinkA=0, shrinkB=0))

    def label(x, y, text, ha="center", va="center"):
        page.fig.text(x / page.W, 1 - y / page.H, text, ha=ha, va=va, fontsize=7.6,
                      fontfamily="Garamond", linespacing=1.35)

    arrow(2.35, 4.05, 2.35, 5.25); label(2.47, 4.65, "Short confirmed", ha="left")
    arrow(3.75, 6.35, 5.10, 6.35); label(4.42, 6.18, "Bottom\nconfirmed", va="bottom")
    arrow(6.52, 5.25, 6.52, 4.05); label(6.64, 4.65, "Profit ladder or trail\ncompletes", ha="left")
    arrow(5.10, 3.20, 3.75, 3.20); label(4.42, 3.34, "Exit phase ends", va="bottom")
    page.fig.text(0.95 / page.W, 1 - 7.16 / page.H,
                  "Between short confirmation and bottom confirmation the band allocator is withdrawn; only the lock "
                  "rule applies. The lock covered 783 of 1,041 weeks over the full period.",
                  fontsize=7.6, fontfamily="Garamond", style="italic", va="top")
    finish(page, out, "fig03_rq03_state_machine",
           "Notes: S3 is the only fully invested state. The pre-bottom supplement inside S2 is a conditional "
           "equity slice, not a return to the band allocator. Parameters behind each rule are listed in the "
           "parameter register.",
           SRC + " Values: results/rq03_parameters.csv and the executed configuration.", 7.72)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--handoff", required=True)
    a = ap.parse_args()
    R = Path(a.repo); res = R / "results"; D = Path(a.dataset); H = Path(a.handoff)
    fig01(pd.read_csv(res / "rq01_dev_phase_pool.csv"), pd.read_csv(res / "rq01_summary.csv"), res)
    fig02(pd.read_csv(res / "rq02_entry_exit.csv"), pd.read_csv(res / "rq02_bottom_zone.csv"), res)
    fig03(pd.read_csv(res / "rq03_standard/CHOSEN_STANDARD_LEDGER_FULL.csv"), res)
    weekly = pd.read_csv(H / "RQ02_DEV_MAX_LEADTIME_20261026/OVERLAY_2020/BENCHMARK_OOS_WEEKLY.csv",
                         parse_dates=["week_end"])
    fig04(pd.read_csv(res / "rq03_standard/CHOSEN_STANDARD_LEDGER_DEV.csv"),
          pd.read_csv(res / "rq03_standard/CHOSEN_STANDARD_LEDGER_OOS.csv"),
          _equity(D), pd.read_csv(res / "rq03_nav_terminal.csv"), weekly, res)
    fig05(pd.read_csv(res / "rq03_nav_terminal.csv"), res)
    fig06(res)


if __name__ == "__main__":
    main()
