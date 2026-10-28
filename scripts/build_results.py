"""Build the DEV/OOS contribution tables and figures from a canonical run.

    python scripts/build_results.py --run <output dir of run_canonical.py> --out results

Everything is computed from artefacts the pipeline wrote; nothing is hard-coded
and figures that cannot be scored are emitted with an explicit status instead of
an invented value.
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
PHASE_VI = {"PRE": "Trước vào", "ENTRY": "Vào", "PEAK": "Đỉnh", "DEESC": "Giảm áp lực",
            "RECOVERY": "Phục hồi", "EXIT": "Thoát"}
BANDS = [("normal", "#e8f1fb", "Bình thường"),
         ("to_short", "#fde7c8", "Bình thường → Last short"),
         ("to_bottom", "#f7c6c7", "Last short → Vùng đáy"),
         ("to_exit", "#d7ecd9", "Vùng đáy → Thoát (bình thường)")]


def _fpr_fnr(y, pred, mask=None):
    y = np.asarray(y).astype(int)
    p = np.asarray(pred).astype(int)
    if mask is not None:
        y, p = y[mask], p[mask]
    tp = int(((y == 1) & (p == 1)).sum()); fn = int(((y == 1) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum()); tn = int(((y == 0) & (p == 0)).sum())
    fnr = 100.0 * fn / (tp + fn) if (tp + fn) else np.nan
    fpr = 100.0 * fp / (fp + tn) if (fp + tn) else np.nan
    return dict(TP=tp, FN=fn, FP=fp, TN=tn, FNR=fnr, FPR=fpr, n=tp + fn + fp + tn)


# ---------------------------------------------------------------- RQ01
def rq01(run: Path, out: Path):
    rows = []
    for period, tape in (("DEV", "RQ01/RQ01_DEV_ISSUE_TAPE.csv"),
                         ("OOS", "RQ01/RQ01_OOS_ISSUE_TAPE.csv")):
        f = run / tape
        if not f.is_file():
            continue
        df = pd.read_csv(f)
        y = (df["label"] != "N").astype(int) if "label" in df else df["stress"].astype(int)
        pool = df["decision_stress"].astype(int)

        if "individual_vote" in df:
            ind = (df["individual_vote"] == "PASS").astype(int)
            ind_eval = df["individual_vote"].notna().values
        else:
            ind, ind_eval = None, None

        for ph in PHASES:
            m = (df["phase"] == ph).values
            if not m.any():
                continue
            rp = _fpr_fnr(y, pool, m)
            ri = _fpr_fnr(y, ind, m & ind_eval) if ind is not None else dict(
                TP=np.nan, FN=np.nan, FP=np.nan, TN=np.nan, FNR=np.nan, FPR=np.nan, n=0)
            rows.append({"period": period, "phase": ph, "phase_vi": PHASE_VI[ph],
                         "n_pool": rp["n"], "pool_FNR": rp["FNR"], "pool_FPR": rp["FPR"],
                         "n_ind": ri["n"], "ind_FNR": ri["FNR"], "ind_FPR": ri["FPR"]})
        ro = _fpr_fnr(y, pool)
        ri = _fpr_fnr(y, ind, ind_eval) if ind is not None else dict(FNR=np.nan, FPR=np.nan, n=0)
        rows.append({"period": period, "phase": "ALL", "phase_vi": "Toàn bộ",
                     "n_pool": ro["n"], "pool_FNR": ro["FNR"], "pool_FPR": ro["FPR"],
                     "n_ind": ri["n"], "ind_FNR": ri["FNR"], "ind_FPR": ri["FPR"]})
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "rq01_fnr_fpr_phase_pool.csv", index=False, encoding="utf-8-sig")

    # pool-operator table (DEV only; the stage OOS tape carries the pooled decision)
    p = run / "RQ01/RQ01_DEV_POOLS.csv"
    if p.is_file():
        pd.read_csv(p).to_csv(out / "rq01_pool_operators_dev.csv", index=False, encoding="utf-8-sig")

    # figure: FNR/FPR by phase, individual vs pool, DEV and OOS side by side
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, metric, title in zip(axes, ("FNR", "FPR"),
                                 ("Tỷ lệ bỏ sót (FNR, %)", "Tỷ lệ báo động sai (FPR, %)")):
        x = np.arange(len(PHASES)); w = 0.2
        for i, (period, colour) in enumerate((("DEV", "#1f4e79"), ("OOS", "#c00000"))):
            sub = tab[(tab.period == period)].set_index("phase").reindex(PHASES)
            ax.bar(x + (i - 0.5) * 2 * w - w / 2, sub[f"ind_{metric}"], w,
                   label=f"{period} – individual", color=colour, alpha=0.45)
            ax.bar(x + (i - 0.5) * 2 * w + w / 2, sub[f"pool_{metric}"], w,
                   label=f"{period} – pool", color=colour)
        ax.set_xticks(x); ax.set_xticklabels([PHASE_VI[p] for p in PHASES], rotation=20)
        ax.set_title(title); ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("phần trăm")
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle("RQ01 — FNR/FPR theo sáu pha: tín hiệu riêng lẻ và quyết định pool (DEV/OOS)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "fig_rq01_phase_pool.png", dpi=170)
    plt.close(fig)
    return tab


# ---------------------------------------------------------------- RQ02
def rq02(run: Path, out: Path):
    rows = []
    for period, f in (("DEV", "RQ02/RQ02_DEV_ONSET_METRICS.csv"),
                      ("OOS", "RQ02/RQ02_OOS_ONSET_METRICS.csv")):
        p = run / f
        if not p.is_file():
            continue
        d = pd.read_csv(p)
        keep = d[d["model"].isin(["M00", "M01", "M10", "M11"])].copy()
        best = keep.sort_values("FNR").groupby("horizon", as_index=False).first()
        for _, r in best.iterrows():
            rows.append({"period": period, "horizon": r["horizon"], "n": r.get("n"),
                         "events": r.get("events"), "auc": r.get("auc"),
                         "FNR": 100 * float(r["FNR"]), "FPR": 100 * float(r["FPR"]),
                         "threshold": r.get("threshold", np.nan)})
    hz = pd.DataFrame(rows)
    hz["h_months"] = hz["horizon"].str.replace("H", "").astype(int)
    hz = hz.sort_values(["period", "h_months"])
    hz.to_csv(out / "rq02_horizon_fnr_fpr.csv", index=False, encoding="utf-8-sig")

    # last-short detection (S5): flag = a dated last-short is present
    ls = []
    for period, f in (("DEV", "RQ02/RQ02_DEV_S5.csv"), ("OOS", "RQ02/RQ02_OOS_S5.csv")):
        p = run / f
        if not p.is_file():
            continue
        d = pd.read_csv(p)
        y = (d["label"] != "N").astype(int).values
        pred = d["s5_last_short"].notna().astype(int).values
        m = _fpr_fnr(y, pred)
        m.update({"period": period})
        ls.append(m)
    ls = pd.DataFrame(ls)
    ls.to_csv(out / "rq02_last_short.csv", index=False, encoding="utf-8-sig")

    # bottom zone
    bt = []
    p = run / "RQ02/RQ02_DEV_BOTTOM_ZONES.csv"
    if p.is_file():
        s = pd.read_csv(p)["status"].value_counts()
        hit, miss = int(s.get("HIT", 0)), int(s.get("MISS", 0))
        bt.append({"period": "DEV", "HIT": hit, "MISS": miss,
                   "hit_rate": 100.0 * hit / max(hit + miss, 1), "status": "SCORED"})
    p = run / "RQ02/RQ02_OOS_BOTTOM.csv"
    if p.is_file():
        d = pd.read_csv(p)
        status = str(d.iloc[0, 0]) if d.shape[1] == 1 else "UNKNOWN"
        bt.append({"period": "OOS", "HIT": np.nan, "MISS": np.nan, "hit_rate": np.nan,
                   "status": status})
    bt = pd.DataFrame(bt)
    bt.to_csv(out / "rq02_bottom_zone.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for period, marker in (("DEV", "o"), ("OOS", "s")):
        d = hz[hz.period == period].sort_values("h_months")
        if not len(d):
            continue
        axes[0].plot(d.h_months, d.FNR, marker=marker, label=period)
        axes[1].plot(d.h_months, d.FPR, marker=marker, label=period)
    for ax, t in zip(axes, ("FNR (%) — H15 → H6", "FPR (%) — H15 → H6")):
        ax.set_title(t); ax.set_xlabel("chân trời (tháng)"); ax.grid(alpha=0.25); ax.legend()
    fig.suptitle("RQ02 — sai số dự báo khởi phát theo chân trời H15 → H6 (DEV/OOS)", fontsize=11)
    fig.tight_layout(); fig.savefig(out / "fig_rq02_horizon.png", dpi=170); plt.close(fig)
    return hz, ls, bt


# ---------------------------------------------------------------- RQ03
def _nav(run: Path, period: str):
    f = run / f"RQ03/RQ03_{period}_NAV_NET.csv"
    tape = run / f"RQ03/RQ03_{period}_ASSET_TAPE.csv"
    if not f.is_file():
        return None
    nav = pd.read_csv(f, parse_dates=["week_end"])
    if tape.is_file():
        t = pd.read_csv(tape, parse_dates=["week_end"])
        eq = t[["week_end", "E1", "E2", "E3"]].copy()
        eq["EQUITY100"] = eq[["E1", "E2", "E3"]].mean(axis=1).add(1).cumprod()
        nav = nav.merge(eq[["week_end", "EQUITY100"]], on="week_end", how="left")
    return nav


def _bands(run: Path, nav: pd.DataFrame, period: str):
    """Four lifecycle stages per week, derived from the regime series plus the
    last-short dates and the trough weeks the pipeline recorded."""
    reg = nav.set_index("week_end")["regime_expost"].astype(str)
    stage = pd.Series("normal", index=reg.index)
    stress = reg != "N"
    if not stress.any():
        return stage

    troughs = []
    p = run / f"RQ02/RQ02_{period}_BOTTOM_TROUGHS.csv"
    if p.is_file():
        troughs = sorted(pd.to_datetime(pd.read_csv(p)["trough_week"]).unique())
    shorts = []
    p = run / f"RQ02/RQ02_{period}_S5.csv"
    if p.is_file():
        s = pd.read_csv(p)
        shorts = sorted(pd.to_datetime(s["s5_last_short"].dropna()).unique())

    idx = reg.index
    eps = []
    on = None
    for d, s in stress.items():
        if s and on is None:
            on = d
        elif not s and on is not None:
            eps.append((on, d)); on = None
    if on is not None:
        eps.append((on, idx[-1]))

    for start, end in eps:
        w = (idx >= start) & (idx <= end)
        tr = [d for d in troughs if start <= d <= end]
        sh = [d for d in shorts if start <= d <= end]
        t_bottom = tr[0] if tr else None
        t_short = min(sh) if sh else None
        if t_short is not None and t_short >= start:
            stage[w & (idx <= t_short)] = "to_short"
        if t_bottom is not None and t_bottom >= start:
            lo = t_short if (t_short is not None and t_short < t_bottom) else start
            stage[w & (idx > lo) & (idx <= t_bottom)] = "to_bottom"
            stage[w & (idx > t_bottom)] = "to_exit"
        elif t_short is not None:
            stage[w & (idx > t_short)] = "to_bottom"
    return stage


STRATS = ["POLICY", "EW", "IVOL", "MVP", "ERC", "MaxDiv", "EQUITY100"]
COLOURS = {"POLICY": "#c00000", "EW": "#1f4e79", "IVOL": "#2e75b6", "MVP": "#70ad47",
           "ERC": "#7030a0", "MaxDiv": "#bf9000", "EQUITY100": "#404040"}
LABELS = {"POLICY": "Mô hình chính", "EW": "EW (trọng số đều)", "IVOL": "IVOL",
          "MVP": "MVP", "ERC": "ERC", "MaxDiv": "MaxDiv", "EQUITY100": "100% cổ phiếu"}


def rq03(run: Path, out: Path):
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    summary = []
    for ax, period in zip(axes, ("DEV", "OOS")):
        nav = _nav(run, period)
        if nav is None:
            ax.text(0.5, 0.5, f"{period}: không có dữ liệu", ha="center"); continue
        stage = _bands(run, nav, period)
        have = [c for c in STRATS if c in nav.columns]
        ymax = float(nav[have].max().max()) * 1.02
        for key, colour, _ in BANDS:
            m = (stage == key).values
            if m.any():
                ax.fill_between(nav["week_end"], 0, ymax,
                                where=m, color=colour, step="mid", linewidth=0, zorder=0)
        if "EQUITY100" not in have:
            ax.text(0.99, 0.02, "100% cổ phiếu: không có tape tài sản trong khối OOS",
                    transform=ax.transAxes, ha="right", fontsize=7, style="italic")
        for s in STRATS:
            if s in nav:
                ax.plot(nav["week_end"], nav[s], color=COLOURS[s],
                        lw=2.0 if s == "POLICY" else 1.2,
                        ls="-" if s == "POLICY" else "--", label=LABELS[s], zorder=3)
        ax.set_title(f"{period}: NAV của mô hình chính so với năm đối chứng và 100% cổ phiếu")
        ax.set_ylabel("NAV (vốn ban đầu = 1,0)"); ax.grid(alpha=0.25)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.legend(fontsize=7, ncol=4, loc="upper left")
        last = nav.iloc[-1]
        summary.append({"period": period, "n_weeks": len(nav),
                        **{s: float(last[s]) for s in STRATS if s in nav}})
        for key, _, _ in BANDS:
            summary[-1][f"weeks_{key}"] = int((stage == key).sum())
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c, _ in BANDS]
    axes[0].legend(handles=handles, labels=[v for _, _, v in BANDS], fontsize=7,
                   loc="lower right", title="Pha vòng đời")
    fig.suptitle("RQ03 — NAV theo tuần, DEV 2005–2019 và OOS 2020–2024", fontsize=12)
    fig.tight_layout(); fig.savefig(out / "fig_rq03_nav_dev_oos.png", dpi=170); plt.close(fig)

    sm = pd.DataFrame(summary)
    sm.to_csv(out / "rq03_nav_terminal.csv", index=False, encoding="utf-8-sig")

    # excess-return bars
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, period in zip(axes, ("DEV", "OOS")):
        r = sm[sm.period == period]
        if not len(r):
            continue
        r = r.iloc[0]
        base = r["POLICY"]
        names = [s for s in STRATS if s != "POLICY"]
        vals = [100.0 * (base / r[s] - 1.0) for s in names]
        ax.bar([LABELS[s] for s in names], vals, color="#1f4e79")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(f"{period}: mô hình chính vượt đối chứng (%)")
        ax.set_ylabel("% cao hơn về NAV cuối kỳ"); ax.grid(axis="y", alpha=0.25)
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=8)
    fig.tight_layout(); fig.savefig(out / "fig_rq03_excess.png", dpi=170); plt.close(fig)
    return sm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    run, out = Path(a.run), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    r1 = rq01(run, out)
    r2 = rq02(run, out)
    r3 = rq03(run, out)
    print("RQ01 rows:", len(r1))
    print(r1.to_string(index=False))
    print("\nRQ02 horizons:\n", r2[0].to_string(index=False))
    print("\nRQ02 last short:\n", r2[1].to_string(index=False))
    print("\nRQ02 bottom:\n", r2[2].to_string(index=False))
    print("\nRQ03 terminal:\n", r3.to_string(index=False))


if __name__ == "__main__":
    main()
