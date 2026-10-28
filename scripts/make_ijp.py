"""Representative Individual / Joint / Pool methodology diagrams (RQ01 and RQ02).

Compact, unnumbered GT2006-canvas exhibits. Connectors use orthogonal segments
and a single junction per convergence; arrows never cross text or boxes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gt2006.figure_canvas import FigurePage  # noqa: E402
from make_figures import finish  # noqa: E402

INK = "#000000"
NAVY = "#000080"
MAG = "#FF00FF"
GREEN = "#008000"
RED = "#CC0000"
GREY = "#555555"


def box(page: FigurePage, x: float, y: float, w: float, h: float, text: str,
        size: float = 7.8, fill: str | None = None) -> None:
    page.fig.add_artist(Rectangle((x / page.W, 1 - (y + h) / page.H), w / page.W, h / page.H,
                                  transform=page.fig.transFigure, facecolor=fill or "white",
                                  edgecolor=INK, linewidth=0.6, zorder=1))
    page.fig.text((x + w / 2) / page.W, 1 - (y + h / 2) / page.H, text, ha="center", va="center",
                  fontsize=size, fontfamily="Garamond", linespacing=1.28, zorder=2)


def seg(page: FigurePage, x1: float, y1: float, x2: float, y2: float, arrow_end: bool = True) -> None:
    """Straight horizontal or vertical connector; no diagonal routing."""
    page.fig.add_artist(FancyArrowPatch((x1 / page.W, 1 - y1 / page.H), (x2 / page.W, 1 - y2 / page.H),
                                        transform=page.fig.transFigure, arrowstyle="-|>" if arrow_end else "-",
                                        mutation_scale=9, linewidth=0.8, color=INK))


def junction(page: FigurePage, centers: list[float], y_from: float, y_bus: float,
             x_in: float, y_in: float) -> None:
    """Drop each input to a horizontal bus, then one vertical into the target node."""
    for cx in centers:
        seg(page, cx, y_from, cx, y_bus, arrow_end=False)
    if centers:
        seg(page, centers[0], y_bus, centers[-1], y_bus, arrow_end=False)
    seg(page, x_in, y_bus, x_in, y_in)


def timeline_ax(page: FigurePage, x0: float, y: float, w: float, h: float):
    ax = page.fig.add_axes([x0 / page.W, 1 - (y + h) / page.H, w / page.W, h / page.H])
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis("off")
    return ax


def fig_ijp01(out: Path) -> None:
    page = FigurePage(number=None, title="Individual Signal - Six-Month Observation Window", top_in=0.80,
                      paragraph="An Individual source votes at each issue month. Seven monthly observations form "
                                "21 pairwise comparisons and one endpoint difference; a comparison confirms only when "
                                "it is positive and reaches a magnitude cut. The vote needs one strong movement, or "
                                "a weak movement backed by a count of confirming comparisons.")
    ax = timeline_ax(page, 0.95, 2.25, 6.60, 2.30)
    xs = np.linspace(1, 11, 7)
    ys = [7.6, 6.4, 7.0, 5.2, 6.0, 4.6, 8.6]
    ax.plot(xs, ys, "o", color=NAVY, markersize=6, zorder=3)
    for i in range(6):
        col = GREEN if ys[i + 1] >= ys[i] else RED
        ax.annotate("", xy=(xs[i + 1], ys[i + 1]), xytext=(xs[i], ys[i]),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=0.9, mutation_scale=7))
    ax.plot([xs[0], xs[0]], [1.9, ys[0]], color=INK, lw=0.5, linestyle=":")
    ax.plot([xs[-1], xs[-1]], [1.9, ys[-1]], color=INK, lw=0.5, linestyle=":")
    ax.annotate("", xy=(xs[0], 1.9), xytext=(xs[-1], 1.9),
                arrowprops=dict(arrowstyle="<|-|>", color=INK, lw=0.9))
    ax.text(6.0, 1.15, "endpoint difference D = sign x (last - first)", fontsize=8,
            fontfamily="Garamond", ha="center")
    ax.axhline(7.9, xmin=0.02, xmax=0.85, color=MAG, lw=0.9, linestyle=(0, (5, 2.5)))
    ax.text(10.35, 7.9, "strong cut = Q75", fontsize=7.4, fontfamily="Garamond", va="center",
            color=MAG, ha="left", clip_on=False)
    ax.axhline(3.2, xmin=0.02, xmax=0.85, color=GREEN, lw=0.9, linestyle=(0, (5, 2.5)))
    ax.text(10.35, 3.2, "weak cut = Q35", fontsize=7.4, fontfamily="Garamond", va="center",
            color=GREEN, ha="left", clip_on=False)
    ax.text(6.0, 9.4, "up = confirming direction (green), down = opposing (red)", fontsize=7.6,
            fontfamily="Garamond", ha="center")

    box(page, 0.95, 4.95, 2.05, 0.72, "Strong movement:\nD >= Q75")
    box(page, 3.225, 4.95, 2.05, 0.72, "Weak movement:\nQ35 <= D < Q75")
    box(page, 5.50, 4.95, 2.05, 0.72, "Count confirmation:\nk >= ceil(M / 2) of M pairs")
    box(page, 2.675, 6.30, 3.15, 0.70, "Individual vote = Strong OR (Weak AND Count)", fill="#FCE4D6", size=8.2)
    junction(page, [1.975, 4.25, 6.525], 5.67, 5.92, 4.25, 6.30)

    finish(page, out, "fig_ijp01_rq01_individual",
           "Notes: seven observations create 21 comparisons; the count threshold is half of them. The vote is issued "
           "at the issue month only, with no persistence requirement.",
           "Source: author's illustration of the executed confirmation kernel. Full thresholds: Appendix A.1.", 7.45)


def fig_ijp02(out: Path) -> None:
    page = FigurePage(number=None, title="Joint Signal - Two Synchronized Indicators", top_in=0.80,
                      paragraph="A Joint pair combines two indicators observed in the same month. Each member keeps "
                                "its own magnitude cut; paired movements are counted on synchronized bases. Selection "
                                "gates decide which pairs may enter the rule; the activation gates run afterwards.")
    ax = timeline_ax(page, 0.95, 2.25, 6.60, 2.30)
    xa = np.linspace(1, 11, 7)
    ax.plot(xa, [8.0] * 7, "o", color=NAVY, markersize=5)
    ax.text(0.05, 9.15, "Indicator A (monthly) - its own Q35/Q75 cuts", fontsize=8.5, fontfamily="Garamond", va="bottom")
    ax.plot([1, 10.2], [8.6, 8.6], color=MAG, lw=0.8, linestyle=(0, (5, 2.5)))
    ax.text(10.35, 8.6, "Q75(A)", fontsize=7.4, fontfamily="Garamond", va="center", color=MAG,
            ha="left", clip_on=False)
    xb = np.linspace(1.5, 11.5, 7)
    ax.plot(xb, [3.4] * 7, "s", color=GREEN, markersize=5)
    ax.text(0.05, 4.55, "Indicator B (same calendar months) - its own cuts", fontsize=8.5, fontfamily="Garamond", va="bottom")
    ax.plot([1, 10.2], [4.0, 4.0], color=GREEN, lw=0.8, linestyle=(0, (5, 2.5)))
    ax.text(10.35, 4.0, "Q35(B)", fontsize=7.4, fontfamily="Garamond", va="center", color=GREEN,
            ha="left", clip_on=False)
    for a, b in zip(xa[:4], xb[:4]):
        ax.plot([a, b], [8.0, 3.4], color=INK, lw=0.5, linestyle=":")
    ax.text(8.1, 5.9, "synchronized bases\n(same issue, EXACT L = 0)", fontsize=7.6, fontfamily="Garamond",
            ha="center", va="center")

    box(page, 0.95, 4.95, 2.05, 0.72, "Selection gates:\n>= 2 matched pairs,\n15% false-alarm budget")
    box(page, 3.225, 4.95, 2.05, 0.72, "Gate 1 - magnitudes:\neach member must pass\nits own base test")
    box(page, 5.50, 4.95, 2.05, 0.72, "Gate 2 - counts:\nstrong pairs, or a majority\nof paired confirmations")
    box(page, 2.675, 6.30, 3.15, 0.70, "Joint activation = Gate 1 AND Gate 2", fill="#FCE4D6", size=8.2)
    junction(page, [1.975, 4.25, 6.525], 5.67, 5.92, 4.25, 6.30)

    finish(page, out, "fig_ijp02_rq01_joint",
           "Notes: the thresholds that select a pair are applied once; the activation thresholds are applied at every "
           "issue afterwards. Insufficient data yields NOT_EVALUATED, never a calm vote.",
           "Source: author's illustration of the executed pair machinery. Full thresholds: Appendix A.2-A.3.", 7.45)


def fig_ijp03(out: Path) -> None:
    page = FigurePage(number=None, title="Pool Integration - From Signals to a Stress Decision", top_in=0.80,
                      paragraph="The Pool is not an average and not a vote. A boosted-tree stress probability is "
                                "compared with the country threshold first; the joint packets then modify the final "
                                "classification: one can only add stress, one can only remove it.")
    box(page, 0.95, 2.30, 1.85, 0.85, "Individual layer\nfive indicators,\nstrong / weak / count")
    seg(page, 2.80, 2.725, 3.10, 2.725)
    box(page, 3.10, 2.30, 1.75, 0.85, "Boosted-tree\nprobability Ps")
    seg(page, 4.85, 2.725, 5.15, 2.725)
    box(page, 5.15, 2.30, 2.40, 0.85, "Country threshold\nPs >= t(country)")
    box(page, 2.60, 3.60, 3.30, 0.72, "Joint packets: CLIFS + VIX adds stress;\nDFC + GFC price removes stress")
    box(page, 2.075, 4.75, 3.90, 0.85, "Final stress decision\n( threshold OR additive )  AND NOT  removal", fill="#FCE4D6", size=8.4)
    seg(page, 6.35, 3.15, 6.35, 4.42, arrow_end=False)
    seg(page, 6.35, 4.42, 4.025, 4.42, arrow_end=False)
    seg(page, 4.025, 4.42, 4.025, 4.75)
    seg(page, 4.25, 4.32, 4.25, 4.75)
    box(page, 1.55, 6.10, 1.90, 0.65, "STRESS\n(rule fires)", fill="#F8CBAD")
    box(page, 5.10, 6.10, 1.90, 0.65, "CALM\n(rule silent)", fill="#DCE6F1")
    seg(page, 4.025, 5.60, 4.025, 5.88, arrow_end=False)
    seg(page, 2.50, 5.88, 6.05, 5.88, arrow_end=False)
    seg(page, 2.50, 5.88, 2.50, 6.10)
    seg(page, 6.05, 5.88, 6.05, 6.10)
    page.fig.text(0.5, 1 - 7.35 / page.H,
                  "Decision rule:  stress = ( Ps >= t(country) )  OR  pkt_joint_CLIFS_VIX ;   AND NOT  pkt_joint_DFC_GFC",
                  ha="center", va="top", fontsize=9.5, fontfamily="Garamond", fontweight="bold")
    finish(page, out, "fig_ijp03_rq01_pool",
           "Notes: the removal packet can cancel stress raised by either other layer. Country thresholds span two "
           "orders of magnitude because the score is not comparable across countries.",
           "Source: author's illustration of the registered rule. Country cuts: Appendix A.4.", 7.95)


def fig_ijp04(out: Path) -> None:
    page = FigurePage(number=None, title="RQ02 Short Signal - The H15 to H12 Observation Interval", top_in=0.80,
                      paragraph="The representative short observation interval runs from the 45-month milestone (H15) "
                                "to the 36-month milestone (H12): a nine-month window. Comparison counts depend on the "
                                "indicator's frequency; the numbers below are structural capacity, not confirmations.")
    ax = timeline_ax(page, 0.95, 2.30, 6.60, 2.90)
    xs = np.linspace(1, 11, 10)
    ax.plot(xs, [7.2] * 10, "o", color=NAVY, markersize=5)
    ax.text(0.05, 8.4, "ten eligible monthly observations inside H15 -> H12", fontsize=8.5, fontfamily="Garamond", va="bottom")
    ax.text(6.0, 5.9, "structural maximum: 45 pairwise comparisons and 9 endpoint differences",
            fontsize=8.5, fontfamily="Garamond", ha="center")
    xq = np.linspace(2, 10, 4)
    ax.plot(xq, [2.6] * 4, "s", color=GREEN, markersize=5)
    ax.text(0.05, 3.8, "four eligible quarterly observations", fontsize=8.5, fontfamily="Garamond", va="bottom")
    ax.text(6.0, 1.2, "structural maximum: 6 pairwise comparisons and 3 endpoint differences",
            fontsize=8.5, fontfamily="Garamond", ha="center")
    box(page, 0.95, 5.75, 2.05, 0.72, "Short magnitude:\nQ75 of prior absolute\nsigned changes")
    box(page, 3.225, 5.75, 2.05, 0.72, "Direction:\nchange must be positive\n(after sign adjustment)")
    box(page, 5.50, 5.75, 2.05, 0.72, "Activation:\nvalue >= cut AND\nz-score > 0")
    finish(page, out, "fig_ijp04_rq02_short",
           "Notes: the same interval contains fewer comparisons for lower-frequency indicators. The count is computed "
           "on eligible bases only; it is never assumed to equal the structural maximum.",
           "Source: author's illustration of the executed feature geometry. Full thresholds: Appendix B.1.", 6.95)


def fig_ijp05(out: Path) -> None:
    page = FigurePage(number=None, title="RQ02 Long Signal - Progressive Milestones H15 to H6", top_in=0.80,
                      paragraph="The long system accumulates confirmations at five milestones from 45 months down to "
                                "18 months. Each later milestone adds information to an already-issued forecast; it "
                                "never rewrites what an earlier issue would have seen.")
    ax = timeline_ax(page, 0.95, 2.30, 6.60, 2.90)
    ms = [("H15", "45m", 1), ("H12", "36m", 3), ("H10", "30m", 5), ("H8", "24m", 7), ("H6", "18m", 9)]
    for name, months, x in ms:
        ax.plot([x], [6.4], "o", color=NAVY, markersize=6)
        ax.text(x, 7.5, f"{name}\n{months}", fontsize=7.8, fontfamily="Garamond", ha="center", va="bottom")
    for i in range(4):
        ax.annotate("", xy=(ms[i + 1][2], 6.4), xytext=(ms[i][2], 6.4),
                    arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.0, mutation_scale=8))
    ax.text(5, 5.0, "information accumulates as milestones are confirmed", fontsize=8.2,
            fontfamily="Garamond", ha="center")
    ax.text(5, 3.6, "five distinct milestones: structural maximum of 10 pairwise comparisons and 4 endpoint differences",
            fontsize=8.5, fontfamily="Garamond", ha="center")
    box(page, 0.95, 5.75, 2.05, 0.72, "Long magnitude:\nQ75 of prior absolute\nsigned changes")
    box(page, 3.225, 5.75, 2.05, 0.72, "Minimum history:\nat least 12 prior values\nbefore a cut applies")
    box(page, 5.50, 5.75, 2.05, 0.72, "No look-ahead:\nforecasts use data\npublished by the issue date")
    finish(page, out, "fig_ijp05_rq02_long",
           "Notes: the entry horizon ladder runs H15/H12/H10/H8/H6; the published operating point uses the 18-month "
           "horizon. Later observations never reconstruct earlier issued forecasts.",
           "Source: author's illustration of the executed ladder. Full thresholds: Appendix B.1 and B.4.", 6.95)


def fig_ijp06(out: Path) -> None:
    page = FigurePage(number=None, title="RQ02 Joint Confirmation - Long and Short Pairs", top_in=0.80,
                      paragraph="RQ02 Joint confirmations pair two synchronized indicators, count the paired bases "
                                "that both pass their own magnitude cuts, and require the members to be published "
                                "before the issue date. Long pairs feed entry; the short pair set feeds exit and "
                                "bottom factors.")
    ax = timeline_ax(page, 0.95, 2.30, 6.60, 2.90)
    xa = np.linspace(1, 11, 7)
    ax.plot(xa, [7.6] * 7, "o", color=NAVY, markersize=5)
    ax.text(0.05, 8.8, "long indicator (27-month change window)", fontsize=8.5, fontfamily="Garamond", va="bottom")
    xb = np.linspace(1.5, 11.5, 7)
    ax.plot(xb, [3.0] * 7, "s", color=GREEN, markersize=5)
    ax.text(0.05, 4.2, "short indicator (6-month change window)", fontsize=8.5, fontfamily="Garamond", va="bottom")
    for a, b in zip(xa, xb):
        ax.plot([a, b], [7.6, 3.0], color=INK, lw=0.5, linestyle=":")
    ax.text(6.0, 1.45, "paired bases counted only when both members pass their cuts\nand both are published before the issue (EXACT L = 0)",
            fontsize=7.8, fontfamily="Garamond", ha="center", va="center")
    box(page, 0.95, 5.75, 2.05, 0.72, "Long pairs:\nL_CG and L_DSR-P\nfeed the entry pool")
    box(page, 3.225, 5.75, 2.05, 0.72, "Short pairs:\nall 21 pairs\nfeed exit and bottom")
    box(page, 5.50, 5.75, 2.05, 0.72, "Pool:\nindividual + joint\nthresholded per model")
    finish(page, out, "fig_ijp06_rq02_joint",
           "Notes: synchronization uses the monthly intersection with a single quarterly anchor, so a carried quarter "
           "is never counted twice.",
           "Source: author's illustration of the executed pair geometry. Full thresholds: Appendix B.2-B.3.", 6.95)


def main() -> None:
    out = Path("results")
    fig_ijp01(out)
    fig_ijp02(out)
    fig_ijp03(out)
    fig_ijp04(out)
    fig_ijp05(out)
    fig_ijp06(out)


if __name__ == "__main__":
    main()
