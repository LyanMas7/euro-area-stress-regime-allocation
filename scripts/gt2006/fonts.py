"""Font discovery and registration for Matplotlib and ReportLab."""

from __future__ import annotations

import os
from pathlib import Path

from matplotlib import font_manager as fm

_WIN_FONTS = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"

FONT_CANDIDATES = {
    ("Garamond", "regular"): ["GARA.TTF", "gara.ttf"],
    ("Garamond", "bold"): ["GARABD.TTF", "garabd.ttf"],
    ("Garamond", "italic"): ["GARAIT.TTF", "garait.ttf"],
    ("Garamond", "bolditalic"): ["GARABI.TTF", "garabi.ttf"],
    ("Times New Roman", "regular"): ["times.ttf"],
    ("Times New Roman", "bold"): ["timesbd.ttf"],
    ("Times New Roman", "italic"): ["timesi.ttf"],
    ("Times New Roman", "bolditalic"): ["timesbi.ttf"],
}

MPL_FAMILY = {
    "display": "Times New Roman",
    "body": "Garamond",
    "numeric": "Times New Roman",
    "figure": "Garamond",
}

_RL_NAMES = {
    ("Garamond", "regular"): "GTGaramond",
    ("Garamond", "bold"): "GTGaramond-Bold",
    ("Garamond", "italic"): "GTGaramond-Italic",
    ("Garamond", "bolditalic"): "GTGaramond-BoldItalic",
    ("Times New Roman", "regular"): "GTTimes",
    ("Times New Roman", "bold"): "GTTimes-Bold",
    ("Times New Roman", "italic"): "GTTimes-Italic",
    ("Times New Roman", "bolditalic"): "GTTimes-BoldItalic",
}


def font_file(family: str, weight: str = "regular") -> Path:
    for candidate in FONT_CANDIDATES[(family, weight)]:
        path = _WIN_FONTS / candidate
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing font {family} {weight} under {_WIN_FONTS}")


def has_font(family: str, weight: str = "regular") -> bool:
    try:
        font_file(family, weight)
        return True
    except FileNotFoundError:
        return False


def setup_matplotlib() -> None:
    import matplotlib as mpl

    available = {f.name for f in fm.fontManager.ttflist}
    fallback = [name for name in ("Garamond", "Times New Roman", "DejaVu Serif") if name in available]
    if not fallback:
        raise RuntimeError("No suitable serif font found")
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": fallback,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "axes.unicode_minus": True,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.hashsalt": "gt2006-academic-style",
        }
    )


def rl_font(role: str = "body", bold: bool = False, italic: bool = False) -> str:
    family = MPL_FAMILY.get(role, "Garamond")
    if family == "Times New Roman":
        key = "bolditalic" if bold and italic else "bold" if bold else "italic" if italic else "regular"
    else:
        if bold and italic:
            key = "bolditalic" if (_WIN_FONTS / "GARABI.TTF").exists() else "bold"
        elif bold:
            key = "bold"
        elif italic:
            key = "italic"
        else:
            key = "regular"
    return _RL_NAMES[(family, key)]


def register_reportlab_fonts() -> None:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    for key, name in _RL_NAMES.items():
        family, weight = key
        if not has_font(family, weight):
            continue
        path = font_file(family, weight)
        try:
            pdfmetrics.registerFont(TTFont(name, str(path)))
        except Exception:
            pass
    # Register families so <b>/<i> markup resolves to the embedded faces.
    for family, base, bold, italic, bolditalic in [
        ("Garamond", "GTGaramond", "GTGaramond-Bold", "GTGaramond-Italic", "GTGaramond-BoldItalic"),
        ("Times New Roman", "GTTimes", "GTTimes-Bold", "GTTimes-Italic", "GTTimes-BoldItalic"),
    ]:
        try:
            pdfmetrics.registerFontFamily(
                base,
                normal=base,
                bold=bold,
                italic=italic,
                boldItalic=bolditalic if has_font(family, "bolditalic") else bold,
            )
        except Exception:
            pass
