"""Restructuring decision flow: RQ01 state, RQ02 transitions, RQ03 constraints.

Orthogonal junction routing: the three layers feed one bus into the engine, and
the engine fans out through one bus to four balanced outcome nodes. Node text is
kept short; the complete condition expressions live in the table beneath the
figure in the presentation and in Appendix C.3.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.patches import FancyArrowPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gt2006.figure_canvas import FigurePage  # noqa: E402
from make_figures import finish  # noqa: E402

INK = "#000000"


def box(page: FigurePage, x: float, y: float, w: float, h: float, text: str,
        size: float = 7.4, fill: str | None = None) -> None:
    page.fig.add_artist(Rectangle((x / page.W, 1 - (y + h) / page.H), w / page.W, h / page.H,
                                  transform=page.fig.transFigure, facecolor=fill or "white",
                                  edgecolor=INK, linewidth=0.6, zorder=1))
    page.fig.text((x + w / 2) / page.W, 1 - (y + h / 2) / page.H, text, ha="center", va="center",
                  fontsize=size, fontfamily="Garamond", linespacing=1.28, zorder=2)


def seg(page: FigurePage, x1: float, y1: float, x2: float, y2: float, arrow_end: bool = True) -> None:
    page.fig.add_artist(FancyArrowPatch((x1 / page.W, 1 - y1 / page.H), (x2 / page.W, 1 - y2 / page.H),
                                        transform=page.fig.transFigure, arrowstyle="-|>" if arrow_end else "-",
                                        mutation_scale=9, linewidth=0.8, color=INK))


def main() -> None:
    page = FigurePage(number=None, top_in=0.80,
                      title="Restructuring Decision Flow - How RQ01, RQ02 and RQ03 Control Allocation",
                      paragraph="Three verified layers feed one decision engine. The engine reads the state, then the "
                                "transition and bottom information, then the risk and weight constraints; only then "
                                "does it choose to maintain, restructure under the common rules, restructure under "
                                "the bottom rules, or refuse to restructure at all.")

    box(page, 0.95, 2.30, 3.30, 1.42,
        "LAYER A - RQ01 FINANCIAL STATE\n"
        "P_S = stress probability\n"
        "P_N = 1 - P_S by construction\n"
        "normal: P_S = weekly entry hazard\n"
        "stress: P_N = weekly exit hazard",
        size=7.4, fill="#DCE6F1")
    box(page, 0.95, 3.97, 3.30, 1.42,
        "LAYER B - RQ02 TRANSITIONS\n"
        "entry H12, applies outside phases\n"
        "last-short flag: 5 votes, valid 18 months\n"
        "lock: short confirmed, bottom pending\n"
        "bottom gates; F1-F3; exit >= 0.027174",
        size=7.4, fill="#FCE4D6")
    box(page, 0.95, 5.64, 3.30, 1.42,
        "LAYER C - RQ03 RISK AND WEIGHTS\n"
        "weekly expected-shortfall cap 2%\n"
        "VaR/ES confidence 95 / 97.5 / 99%\n"
        "state bands; ladder trims; 10 bp cost\n"
        "long-only, weekly rebalancing",
        size=7.4, fill="#E2EFDA")

    box(page, 5.15, 3.98, 2.40, 1.90,
        "RESTRUCTURING\nDECISION ENGINE\n\n"
        "1 verify state\n2 check lock\n3 common vs bottom eligibility\n4 risk caps\n5 feasibility and clamps\n"
        "6 costs and turnover\n7 execute or hold",
        size=7.4)

    for y in (3.01, 4.68, 6.35):
        seg(page, 4.25, y, 4.85, y, arrow_end=False)
    seg(page, 4.85, 3.01, 4.85, 6.35, arrow_end=False)
    seg(page, 4.85, 4.93, 5.15, 4.93)

    outcomes = [("MAINTAIN\ncurrent allocation", 0.95, "#FFFFFF"),
                ("COMMON\nrestructuring", 2.70, "#DCE6F1"),
                ("BOTTOM-SPECIFIC\nrestructuring", 4.45, "#F8CBAD"),
                ("RESTRUCTURING\nPROHIBITED (lock)", 6.20, "#F4CCCC")]
    centers = [1.725, 3.475, 5.225, 6.975]
    for (text, x, fill), cx in zip(outcomes, centers):
        box(page, x, 7.55, 1.55, 0.72, text, size=7.0, fill=fill)
    seg(page, 6.35, 5.88, 6.35, 7.10, arrow_end=False)
    seg(page, 1.725, 7.10, 6.975, 7.10, arrow_end=False)
    for cx in centers:
        seg(page, cx, 7.10, cx, 7.55)

    finish(page, Path("results"), "fig03_rq03_restructuring_flow",
           "Notes: P_N and P_S are complementary by construction in the executed overlay, not independent "
           "probabilities. The complete condition expressions are in the restructuring table beneath this figure and "
           "in Appendix C.3.",
           "Source: author's construction from the executed overlay and the selected configuration. "
           "Thresholds: Appendix C.3.", 8.85)


if __name__ == "__main__":
    main()
