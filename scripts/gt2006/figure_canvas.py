"""Publication-grade Matplotlib page canvas faithful to the GT2006 layout."""

from __future__ import annotations

import hashlib
import json
import textwrap
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from . import registry
from .fonts import setup_matplotlib
from .textfmt import PROOF_BANNER

REG = registry.REGISTER
PAGE_SIZES = REG["page"]["sizes_in"]


class FigurePage:
    def __init__(
        self,
        number: int | None,
        title: str,
        paragraph: str | None = None,
        equation: str | None = None,
        size: str = "letter",
        orientation: str = "portrait",
        top_in: float = 1.05,
        proof: bool = False,
        banner: str | None = PROOF_BANNER,
    ) -> None:
        setup_matplotlib()
        width, height = PAGE_SIZES[size]
        if orientation == "landscape":
            width, height = height, width
        self.W, self.H = width, height
        self.number = number
        self.title = title
        self.proof = proof
        self.fig = plt.figure(figsize=(width, height), dpi=100)
        self.fig.patch.set_facecolor("white")
        self.metadata: dict = {}
        self._draw_heading(paragraph, equation, top_in)
        if proof and banner:
            self.add_banner(banner)
        if self.number is not None:
            self.add_page_number()

    # ------------------------------------------------------------------ heading
    def _draw_heading(self, paragraph: str | None, equation: str | None, top_in: float) -> None:
        display = "Times New Roman"
        x_center = 0.5
        y = top_in / self.H
        if self.number is not None:
            self.fig.text(
                x_center,
                1 - y,
                f"Figure {self.number}",
                ha="center",
                va="top",
                fontsize=13,
                fontweight="bold",
                fontfamily=display,
                color="black",
            )
            y += 0.24 / self.H
        title_lines = textwrap.wrap(self.title, 78)
        self.fig.text(
            x_center,
            1 - y,
            "\n".join(title_lines),
            ha="center",
            va="top",
            fontsize=13,
            fontweight="bold",
            fontfamily=display,
            linespacing=1.02,
            color="black",
        )
        y += (len(title_lines) * 0.215 + 0.06) / self.H
        if paragraph:
            para = textwrap.fill(paragraph, 108)
            self.fig.text(
                0.75 / self.W,
                1 - y,
                para,
                ha="left",
                va="top",
                fontsize=11,
                fontfamily="Garamond",
                linespacing=1.18,
                color="black",
            )
            para_lines = para.count("\n") + 1
            y += (para_lines * 11 * 1.18 / 72 + 0.08) / self.H
        if equation:
            self.fig.text(
                x_center,
                1 - y,
                equation,
                ha="center",
                va="top",
                fontsize=11,
                fontfamily="Times New Roman",
                color="black",
            )
            y += 0.28 / self.H
        self.heading_bottom_in = y * self.H

    # ------------------------------------------------------------------ helpers
    def fig_rect(self, x0: float, y_top: float, w: float, h: float) -> list[float]:
        """Convert inches measured from the top-left corner to figure coords."""
        return [x0 / self.W, 1 - (y_top + h) / self.H, w / self.W, h / self.H]

    def add_banner(self, text: str) -> None:
        self.fig.text(
            0.5,
            1 - 0.42 / self.H,
            text,
            ha="center",
            va="center",
            fontsize=7.2,
            fontweight="bold",
            fontfamily="Garamond",
            color="black",
            bbox={"boxstyle": "square,pad=0.25", "facecolor": "white", "edgecolor": "black", "linewidth": 0.5},
        )

    def add_page_number(self) -> None:
        self.fig.text(
            0.5,
            0.18 / self.H,
            str(self.number),
            ha="center",
            va="bottom",
            fontsize=10,
            fontfamily="Garamond",
            color="black",
        )

    def add_box(self, x0: float, y_top: float, w: float, h: float, lw: float = 0.35) -> Rectangle:
        rect = Rectangle(
            (x0 / self.W, 1 - (y_top + h) / self.H),
            w / self.W,
            h / self.H,
            transform=self.fig.transFigure,
            fill=False,
            edgecolor="black",
            linewidth=lw,
            joinstyle="miter",
            clip_on=False,
            zorder=0.5,
        )
        self.fig.add_artist(rect)
        return rect

    def add_panel_title(
        self,
        text: str,
        box_x0: float,
        box_y0: float,
        box_w: float,
        pad_in: float = 0.055,
    ) -> None:
        self.fig.text(
            (box_x0 + box_w / 2) / self.W,
            1 - (box_y0 + pad_in) / self.H,
            text,
            ha="center",
            va="top",
            fontsize=REG["typography"]["sizes_pt"]["figure_panel_title"],
            fontweight="bold",
            fontfamily="Garamond",
            color="black",
        )

    def add_panel_caption(self, text: str, cx_in: float, y_top_in: float) -> None:
        self.fig.text(
            cx_in / self.W,
            1 - y_top_in / self.H,
            text,
            ha="center",
            va="top",
            fontsize=REG["typography"]["sizes_pt"]["figure_panel_caption"],
            fontweight="bold",
            fontfamily="Garamond",
            color="black",
        )

    def add_text(
        self,
        x_in: float,
        y_top_in: float,
        text: str,
        size: float = 8.5,
        bold: bool = False,
        italic: bool = False,
        family: str = "Garamond",
        ha: str = "left",
        va: str = "top",
        linespacing: float = 1.15,
    ):
        return self.fig.text(
            x_in / self.W,
            1 - y_top_in / self.H,
            text,
            ha=ha,
            va=va,
            fontsize=size,
            fontweight="bold" if bold else "normal",
            fontstyle="italic" if italic else "normal",
            fontfamily=family,
            linespacing=linespacing,
            color="black",
        )

    # ------------------------------------------------------------------ axes
    def add_axes(self, x0: float, y_top: float, w: float, h: float):
        ax = self.fig.add_axes(self.fig_rect(x0, y_top, w, h))
        ax.set_facecolor("white")
        return ax

    def format_axis(
        self,
        ax,
        xlabel: str | None = None,
        ylabel: str | None = None,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        xticks: list[float] | None = None,
        yticks: list[float] | None = None,
        xticklabels: list[str] | None = None,
        yticklabels: list[str] | None = None,
        rotation: float = 0,
        heavy_frame: bool = False,
        tick_size: float | None = None,
        label_size: float | None = None,
    ):
        sizes = REG["typography"]["sizes_pt"]
        tick_size = tick_size or sizes["tick_label"]
        label_size = label_size or sizes["axis_label"]
        frame_width = REG["axes"]["heavy_frame_width_pt"] if heavy_frame else REG["axes"]["frame_width_pt"]
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(frame_width)
            spine.set_color("black")
        ax.tick_params(
            axis="both",
            which="major",
            direction="out",
            length=2.5,
            width=0.55,
            pad=2.2,
            labelsize=tick_size,
            colors="black",
        )
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
        if xticks is not None:
            ax.set_xticks(xticks)
        if yticks is not None:
            ax.set_yticks(yticks)
        if xticklabels is not None:
            ax.set_xticklabels(xticklabels, rotation=rotation, ha="center", va="top")
        if yticklabels is not None:
            ax.set_yticklabels(yticklabels)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily("Garamond")
            label.set_fontsize(tick_size)
            label.set_color("black")
        if xlabel:
            ax.set_xlabel(
                xlabel,
                fontsize=label_size,
                fontweight="bold",
                fontfamily="Garamond",
                labelpad=3.0,
                color="black",
            )
        if ylabel:
            ax.set_ylabel(
                ylabel,
                fontsize=label_size,
                fontweight="bold",
                fontfamily="Garamond",
                labelpad=3.0,
                color="black",
            )
        ax.grid(False)
        ax.margins(x=0, y=0)
        return ax

    # ------------------------------------------------------------------ legends
    @staticmethod
    def _line_handle(style_key: str, width: float | None = None) -> Line2D:
        style = REG["lines"]["styles"][style_key]
        dashes = REG["lines"]["dash_patterns"].get(style["linestyle"])
        linestyle = (0, tuple(dashes)) if dashes else "-"
        return Line2D(
            [0],
            [0],
            color=style["color"],
            linestyle=linestyle,
            linewidth=width or style["width_pt"],
            solid_capstyle="butt",
        )

    @staticmethod
    def _area_handle(area_key: str) -> Patch:
        style = REG["stacked_areas"]["styles"][area_key]
        return Patch(
            facecolor=style["facecolor"],
            edgecolor=style["edgecolor"],
            hatch=style["hatch"],
            linewidth=REG["stacked_areas"]["boundary_width_pt"],
        )

    def add_legend(
        self,
        entries: list[dict],
        cx_in: float,
        cy_in: float,
        ncol: int = 4,
        frameon: bool = True,
        size: float | None = None,
        handlelength: float = 2.2,
        columnspacing: float = 1.4,
        handletextpad: float = 0.5,
        borderpad: float = 0.25,
        labelspacing: float = 0.25,
    ):
        handles = []
        labels = []
        for entry in entries:
            if "area_key" in entry:
                handles.append(self._area_handle(entry["area_key"]))
            else:
                handles.append(self._line_handle(entry["style_key"], entry.get("width")))
            labels.append(entry["label"])
        legend = self.fig.legend(
            handles,
            labels,
            loc="center",
            bbox_to_anchor=(cx_in / self.W, 1 - cy_in / self.H),
            bbox_transform=self.fig.transFigure,
            ncol=ncol,
            frameon=frameon,
            fancybox=False,
            edgecolor="black",
            facecolor="white",
            framealpha=1.0,
            borderpad=borderpad,
            columnspacing=columnspacing,
            handlelength=handlelength,
            handletextpad=handletextpad,
            labelspacing=labelspacing,
            fontsize=size or REG["typography"]["sizes_pt"]["legend"],
        )
        for text in legend.get_texts():
            text.set_fontfamily("Garamond")
            text.set_fontsize(size or REG["typography"]["sizes_pt"]["legend"])
            text.set_color("black")
        legend.get_frame().set_linewidth(0.45)
        return legend

    # ------------------------------------------------------------------ notes
    def add_note(self, text: str, y_top_in: float | None = None, source: str | None = None, width: int = 118) -> float:
        y = y_top_in if y_top_in is not None else self.H - 1.05
        wrapped = textwrap.fill(text, width)
        self.add_text(
            0.75,
            y,
            wrapped,
            size=REG["typography"]["sizes_pt"]["figure_note"],
            family="Garamond",
            linespacing=1.12,
        )
        lines = wrapped.count("\n") + 1
        y += lines * 8.5 * 1.12 / 72 + 0.03
        if source:
            src = textwrap.fill(source, width)
            self.add_text(
                0.75,
                y,
                src,
                size=REG["typography"]["sizes_pt"]["source_note"],
                family="Garamond",
                italic=True,
                linespacing=1.1,
            )
            y += (src.count("\n") + 1) * 8.5 * 1.1 / 72 + 0.03
        return y

    # ------------------------------------------------------------------ save
    def save(
        self,
        out_dir: str | Path,
        stem: str,
        formats: tuple[str, ...] = ("png", "pdf", "svg"),
        data: dict | None = None,
        spec: dict | None = None,
        close: bool = True,
        compact: bool = False,
    ) -> dict:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.fig.canvas.draw()
        outputs = []
        fixed_date = datetime(2001, 1, 1, 0, 0, 0)
        for fmt in formats:
            path = out_dir / f"{stem}.{fmt}"
            metadata = {
                "Title": self.title,
                "Author": "GT2006 Academic Presentation Standard",
                "Subject": f"Figure {self.number}",
                "Creator": "GT2006 Academic Presentation Standard",
                "Producer": "Matplotlib",
                "CreationDate": fixed_date,
                "ModDate": fixed_date,
            }
            if fmt == "png":
                self.fig.savefig(path, dpi=REG["export"]["png_dpi"], facecolor="white",
                                 bbox_inches="tight" if compact else None,
                                 pad_inches=0.12 if compact else 0.1,
                                 metadata={"Software": "GT2006 Academic Presentation Standard"})
            elif fmt == "pdf":
                self.fig.savefig(path, format="pdf", facecolor="white", metadata=metadata)
            elif fmt == "svg":
                self.fig.savefig(path, format="svg", facecolor="white", metadata={"Creator": "GT2006 Academic Presentation Standard", "Date": "2001-01-01T00:00:00Z"})
            outputs.append(path)
        if data is not None:
            data_path = out_dir / f"{stem}_source_data.csv"
            _write_mapping_csv(data_path, data)
            outputs.append(data_path)
        if spec is not None:
            spec_path = out_dir / f"{stem}_spec.json"
            spec_path.write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")
            outputs.append(spec_path)
        manifest = {
            "figure": self.number,
            "title": self.title,
            "proof": self.proof,
            "page_size_in": [self.W, self.H],
            "style_register_version": REG["register_version"],
            "files": {p.name: _sha256(p) for p in outputs},
        }
        manifest_path = out_dir / f"{stem}_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        outputs.append(manifest_path)
        if close:
            plt.close(self.fig)
        return {"outputs": outputs, "manifest": manifest}


def _write_mapping_csv(path: Path, data: dict) -> None:
    import csv

    arrays = {key: np.asarray(value) for key, value in data.items() if np.ndim(value) > 0}
    if not arrays:
        path.write_text("key,value\n" + "\n".join(f"{k},{v}" for k, v in data.items()), encoding="utf-8")
        return
    lengths = {len(v) for v in arrays.values()}
    if len(lengths) != 1:
        raise ValueError(f"Source-data columns must share a length, got {lengths}")
    scalar = {k: v for k, v in data.items() if np.ndim(v) == 0}
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(list(arrays) + list(scalar))
        for i in range(next(iter(lengths))):
            writer.writerow([arrays[k][i] for k in arrays] + list(scalar.values()))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
