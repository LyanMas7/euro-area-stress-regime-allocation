"""Restructuring cycle: Normal -> Last Short -> Bottom -> Post-Bottom -> Normal.

RQ01 identifies the state; RQ02 supplies Pn/Ps from the H6 quarterly pre-onset
model and the 3-month exit model; RQ03 supplies the risk cap, volatility and
z-score constraints. Four phases only, each with its verified conditions.
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
        size: float = 7.2, fill: str | None = None) -> None:
    page.fig.add_artist(Rectangle((x / page.W, 1 - (y + h) / page.H), w / page.W, h / page.H,
                                  transform=page.fig.transFigure, facecolor=fill or "white",
                                  edgecolor=INK, linewidth=0.6, zorder=1))
    page.fig.text((x + w / 2) / page.W, 1 - (y + h / 2) / page.H, text, ha="center", va="center",
                  fontsize=size, fontfamily="Garamond", linespacing=1.25, zorder=2)


def seg(page: FigurePage, x1: float, y1: float, x2: float, y2: float, arrow_end: bool = True) -> None:
    page.fig.add_artist(FancyArrowPatch((x1 / page.W, 1 - y1 / page.H), (x2 / page.W, 1 - y2 / page.H),
                                        transform=page.fig.transFigure, arrowstyle="-|>" if arrow_end else "-",
                                        mutation_scale=9, linewidth=0.8, color=INK))


def main() -> None:
    page = FigurePage(number=None, top_in=0.80,
                      title="Restructuring Cycle - Normal, Last Short, Bottom, Post-Bottom",
                      paragraph="RQ01 identifies the current state, RQ02 supplies the probabilities Pn and Ps from "
                                "the H6 quarterly pre-onset model and the 3-month exit model, and RQ03 applies the "
                                "risk cap, volatility and z-score constraints. The cycle has four phases; the "
                                "portfolio is locked between last short and bottom confirmation.")

    box(page, 0.95, 2.20, 2.00, 0.95,
        "RQ01 - STATE\npooled rule:\ncalm or stress",
        size=7.2)
    box(page, 3.10, 2.20, 2.55, 0.95,
        "RQ02 - Pn / Ps\npre-onset: H6 quarterly\n(18-month model)\nexit: 3M, p >= 0.027174",
        size=7.2)
    box(page, 5.80, 2.20, 1.75, 0.95,
        "RQ03 - RISK\nES cap 2%\nvolatility, z\n10 bp cost",
        size=7.2)

    box(page, 0.95, 3.70, 1.80, 2.60,
        "NORMAL  (S1)\n\nRQ01: calm\nP_S = weekly hazard of\nthe H6 quarterly\npre-onset probability\n\nallocation:\nequity 80-100%\nbonds 0-20%",
        size=7.2, fill="#DCE6F1")
    box(page, 2.75, 3.70, 1.80, 2.60,
        "LAST SHORT  (S2)\n\nRQ02: five-vote\nshort system fires\nP_N = 1 - P_S\n\nallocation:\nbonds 50-80%\nno equity\nband withdrawn",
        size=7.2, fill="#FCE4D6")
    box(page, 4.55, 3.70, 1.80, 2.60,
        "BOTTOM  (S3)\n\ngates: P_N >= 0.995\nP_S <= 0.02\nz >= 2\ndrawdown <= 5%\nF1-F3: 2 of 3 agree\n\nallocation: equity 100%",
        size=7.2, fill="#F8CBAD")
    box(page, 6.35, 3.70, 1.20, 2.60,
        "POST-\nBOTTOM\n(S4)\n\nexit 3M:\np >= 0.027174\nor ladder\ncomplete\n\nequity 50-\n100%\nbonds 0-25%",
        size=6.8, fill="#E2EFDA")

    seg(page, 2.75, 5.00, 3.65, 5.00)
    seg(page, 4.55, 5.00, 5.45, 5.00)
    seg(page, 6.35, 5.00, 7.10, 5.00)
    seg(page, 6.95, 6.30, 6.95, 6.85, arrow_end=False)
    seg(page, 6.95, 6.85, 1.85, 6.85, arrow_end=False)
    seg(page, 1.85, 6.85, 1.85, 6.30)
    page.fig.text((1.85 + 4.4) / page.W, 1 - 6.98 / page.H,
                  "26 consecutive normal weeks re-arm the cycle", ha="center", va="top",
                  fontsize=7.4, fontfamily="Garamond", style="italic")

    seg(page, 1.95, 3.15, 1.95, 3.70)
    seg(page, 4.375, 3.15, 4.375, 3.42, arrow_end=False)
    seg(page, 4.375, 3.42, 5.45, 3.42, arrow_end=False)
    seg(page, 5.45, 3.42, 5.45, 3.70)
    seg(page, 6.675, 3.15, 6.675, 3.70)

    page.fig.text(0.5, 1 - 7.60 / page.H,
                  "weekly entry hazard 1-(1-p_H6)^(1/78)      weekly exit hazard 1-(1-p_3M)^(1/13)",
                  ha="center", va="top", fontsize=9.0, fontfamily="Garamond", fontweight="bold")
    finish(page, Path("results"), "fig03_rq03_restructuring_flow",
           "Notes: the lock forbids restructuring between last short and bottom confirmation; a favourable signal "
           "produces no trade while the lock is active. Bottom confirmation is a state change, not a return "
           "guarantee: the realised equity trough arrived 0 to 20 weeks after confirmation across the seven "
           "episodes. The complete threshold set is in Appendix C.3.",
           "Source: author's construction from the selected configuration and the executed ledger. "
           "Thresholds: Appendix C.3.", 8.30)


if __name__ == "__main__":
    main()
