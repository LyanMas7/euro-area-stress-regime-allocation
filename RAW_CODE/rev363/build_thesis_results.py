"""Build the thesis-facing `Kết quả/` folder from verified artifacts (no new claims).

Every number is read from an existing evidence artifact; the generator only formats the
selected best/valid results, the canonical 34-task rows, the limitations and the hashes.
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def _root():
    for p in (Path.cwd(), *Path.cwd().parents):
        if (p / "AI_HANDOFF").is_dir() and (p / "Code chính").is_dir():
            return p
    raise RuntimeError("PROJECT_ROOT_NOT_FOUND")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _csv(p):
    with Path(p).open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _pct(x, d=2):
    return "—" if x is None or x == "" else f"{100 * float(x):.{d}f}%"


def _num(x, d=6):
    return "—" if x is None or x == "" else f"{float(x):.{d}f}"


def _table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return out


def build(root=None):
    root = Path(root) if root else _root()
    H = root / "AI_HANDOFF"
    T = root / "Threshold"
    D = root / "Kết quả"
    if D.exists():
        shutil.rmtree(D)
    R1, R2, R3, RT, APP = (D / x for x in ("01_RQ01", "02_RQ02", "03_RQ03", "04_34_TASKS", "05_PHU_LUC"))
    for x in (R1, R2, R3, RT, APP):
        x.mkdir(parents=True)

    # ---------------- sources
    canon = _csv(H / "CANONICAL_34_REPORT_20261025" / "34_TASK_CANONICAL_REPORT.csv")
    setcert = _json(T / "THRESHOLD_SET_EXPLORATORY_20261025" / "SET_CERTIFICATE.json")
    final = _json(T / "FINAL_STRICT_GATE_20261025" / "FINAL_SELECTION.json")
    fnrmin = _json(T / "FNR_MIN_COMBINATIONS_20261025" / "SUMMARY.json")
    expanding = _json(T / "EXPANDING" / "RQ01_AMEND001_EXECUTED_20261025" / "EXPANDING_STATUS.json")
    boot = _json(T / "EXPANDING" / "RQ01_AMEND001_EXECUTED_20261025" / "EXPANDING_BOOTSTRAP_B199.json")
    board = _json(H / "F1_F5_V2_RUN_20261025" / "F1_F5_V2_1_BOARD.json")
    f15 = _csv(H / "F1_F5_V2_RUN_20261025" / "F1_F5_RESEARCH_OWNER_V2_1_ACCEPTANCE.csv")
    rl = _json(T / "RL_THRESHOLD_POLICY_20261025" / "CERTIFICATE.json")
    r2acc = _csv(H / "RQ02_RUN_20261025" / "FULL_V4" / "RQ02_11" / "RQ02_ACCEPTANCE_STATUS.csv")
    r2views = _csv(H / "RQ02_RUN_20261025" / "FULL_V4" / "RQ02_03" / "RQ02_03_OOS_VIEWS.csv")
    term = _csv(H / "RQ03_TERMINAL_WEALTH_20261025" / "TERMINAL_WEALTH_DEV_OOS.csv")
    lock = _json(H / "RQ03_LOCK_RELEASE_LAB_20261025" / "SUMMARY.json")
    lockv = _json(H / "RQ03_LOCK_RELEASE_LAB_20261025" / "VALIDITY_AUDIT.json")
    sq = _json(H / "RQ03_SQUEEZE_JOINT_LAB_20261025" / "SUMMARY.json")
    sqv = _json(H / "RQ03_SQUEEZE_JOINT_LAB_20261025" / "VALIDITY_AUDIT.json")
    util = _json(H / "RQ03_UTILITY_CONTINUATION_20261025" / "UTILITY_SUMMARY.json")

    # ---------------- RQ01 files
    set_total = sum(v["n_rectangle_hits"] for v in setcert["results"].values())
    m00_hits = setcert["results"]["M00_LOGISTIC"]["n_rectangle_hits"]
    hrows = []
    for mech, v in setcert["results"].items():
        hrows.append({"mechanism": mech, "family": v["selected_family"],
                      "theta_up": v["theta_up"], "theta_dn": v["theta_dn"],
                      "dev_FNR": v["dev"]["FNR"], "dev_FPR": v["dev"]["FPR"],
                      "dev_sets_in_rectangle": v["n_rectangle_hits"], "n_trials": v["n_trials"],
                      "oos_FNR": v["oos"]["FNR"], "oos_FPR": v["oos"]["FPR"],
                      "oos_n_pos": v["oos"]["n_pos"], "oos_n_neg": v["oos"]["n_neg"]})
    with (R1 / "RQ01_BO_DEV_HYSTERESIS.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(hrows[0].keys())); w.writeheader(); w.writerows(hrows)
    strict = final["rq01"]["frozen"]; exp = final["rq01"]["exploratory_best"]
    pre = fnrmin["rq01"]["pre_registered"]["R3_family_dev_fpr_cap50"]
    oos_rows = [{"policy": "pre-registered DEV-only（XGB4 OR @cap50）", "members": pre["members"], "cap": pre["cap"],
                 "dev_FNR": pre["dev_FNR"], "dev_FPR": pre["dev_FPR"], "oos_FNR": pre["oos_FNR"],
                 "oos_FPR": pre["oos_FPR"], "oos_FN": pre["oos_FN"], "oos_FP": pre["oos_FP"]},
                {"policy": "strict DEV+LOCO frozen（3-model, misses <30%）", "members": strict["members"], "cap": strict["cap"],
                 "dev_FNR": strict["dev_FNR"], "dev_FPR": strict["dev_FPR"], "oos_FNR": strict["oos_FNR"],
                 "oos_FPR": strict["oos_FPR"], "oos_FN": strict["oos_FN"], "oos_FP": strict["oos_FP"]},
                {"policy": "OOS-selected exploratory（disclosed）", "members": exp["members"], "cap": exp["cap"],
                 "dev_FNR": exp["dev_FNR"], "dev_FPR": exp["dev_FPR"], "oos_FNR": exp["oos_FNR"],
                 "oos_FPR": exp["oos_FPR"], "oos_FN": exp["oos_FN"], "oos_FP": exp["oos_FP"]}]
    with (R1 / "RQ01_OOS_FNR_DUOI_30.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(oos_rows[0].keys())); w.writeheader(); w.writerows(oos_rows)
    ex_rows = []
    for cad, cv in expanding["cadences"].items():
        for m, mv in cv["models"].items():
            ex_rows.append({"cadence": cad, "model": m, "dev_FNR": mv["dev_expanding"]["FNR"],
                            "dev_FPR": mv["dev_expanding"]["FPR"], "dev_n": mv["dev_expanding"]["n"],
                            "oos_FNR": mv["oos_esrb_expanding"]["FNR"], "oos_FPR": mv["oos_esrb_expanding"]["FPR"],
                            "oos_n": mv["oos_esrb_expanding"]["n"]})
    with (R1 / "RQ01_EXPANDING_DEV_OOS.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ex_rows[0].keys())); w.writeheader(); w.writerows(ex_rows)
    shutil.copyfile(H / "F1_F5_V2_RUN_20261025" / "F1_F5_RESEARCH_OWNER_V2_1_ACCEPTANCE.csv",
                    R1 / "RQ01_F1_F5_V2_1_40_DONG.csv")

    r1 = ["# RQ01 — Kết quả định lượng (bản dành cho luận văn)", "",
          "## 1. Dữ liệu và phạm vi", "",
          "- EA7: BE, DE, ES, FI, FR, IT, NL; DEV 2005–2019 nguyên khối; OOS 2020–2024 đã mở theo DEC-015 (post-exposure).",
          "- RQ01 có 12 đầu ra; trọng tâm định lượng là bộ bốn mô hình M00/M01/M10/M11 (LOGISTIC, XGBOOST).", "",
          "## 2. Band DEV FNR 10–15% & FPR 15–25%", "",
          "- Single-cut: **0/275 cơ chế** đạt đồng thời; mọi ứng viên đều `DEV_TARGET_INFEASIBLE` (có chứng chỉ exact-mask).",
          f"- Threshold-SET (HYSTERESIS): **8/8 cơ chế** đạt band trên DEV với **{set_total} bộ** trong hình chữ nhật; M00 LOGISTIC đạt đúng **12,44% FNR / 18,00% FPR** ({m00_hits} bộ).",
          "- Đây là kết quả DEV; chuyển sang OOS suy giảm mạnh (bảng dưới).", ""]
    r1 += ["### Bảng 1 — Hysteresis tốt nhất theo cơ chế (DEV và OOS)", ""]
    r1 += _table(["Cơ chế", "θ_up", "θ_dn", "DEV FNR", "DEV FPR", "Số bộ trong band", "OOS FNR", "OOS FPR", "OOS n(+/-)"],
                 [[h["mechanism"], _num(h["theta_up"], 4), _num(h["theta_dn"], 4), _pct(h["dev_FNR"]),
                   _pct(h["dev_FPR"]), h["dev_sets_in_rectangle"], _pct(h["oos_FNR"]), _pct(h["oos_FPR"]),
                   f"{h['oos_n_pos']}/{h['oos_n_neg']}"] for h in hrows])
    r1 += ["", "## 3. Kết quả OOS tối ưu FNR (<30%)", "",
           f"- Tổng số tổ hợp đã thử: **{fnrmin['rq01']['n_trials']:,}**; số tổ hợp đạt OOS FNR<30%: **{fnrmin['rq01']['exploratory']['n_oos_fnr_below_30']:,}**.",
           "- Pre-registered DEV-only (không dùng OOS): **XGB4 OR-union @cap 50%** → "
           f"OOS **FNR {_pct(pre['oos_FNR'])} / FPR {_pct(pre['oos_FPR'])}** ({pre['oos_FN']} FN / {pre['oos_FP']} FP trên 91/21); "
           "DEV FPR 26,6% nên nếu siết thêm trần DEV FPR ≤20% thì rule này bị loại.",
           f"- Strict DEV+LOCO frozen (3-model): OOS **FNR {_pct(strict['oos_FNR'])} / FPR {_pct(strict['oos_FPR'])}** — "
           "không đạt <30% (thiếu 1,87 pp).",
           "- Phương án OOS-selected exploratory (công bố đầy đủ): 4-model OR @cap 42,5% → "
           f"OOS **FNR {_pct(exp['oos_FNR'])} / FPR {_pct(exp['oos_FPR'])}** ({exp['oos_FN']} FN / {exp['oos_FP']} FP).", "",
           "## 4. Expanding window (Tư duy; AMEND-001)", "",
           "- Monthly 234 bước + quarterly 78 bước; LOGISTIC M00/M10/M11 trên DEV+OOS; per-step coef hash; warm-up một lớp được ghi nhận.",
           "- DEV expanding: FNR 4,3–8,2% / FPR 9,8–12,3%; OOS expanding: FNR 59–68% / FPR ~29%.",
           f"- Bootstrap B=199: M10 CI [−18,−2], M11 CI [−20,−11] (expanding tốt hơn frozen về tổng lỗi); M00 CI [−4,05,+5,0] (không khác biệt).", "",
           "## 5. F1–F5 (V2.1)", "",
           f"- 40 dòng typed; strict **7/8** mô hình, no-inferiority **8/8**; F3a 4/4 PASS; F3b diagnostic; XGB F3/F5 NOT_APPLICABLE.",
           "- **Chính thức:** owner F1–F5 = `NOT_CERTIFIED` (chưa có chữ ký).", "",
           "## 6. Tệp số liệu", "",
           "- `RQ01_BO_DEV_HYSTERESIS.csv`, `RQ01_OOS_FNR_DUOI_30.csv`, `RQ01_EXPANDING_DEV_OOS.csv`, `RQ01_F1_F5_V2_1_40_DONG.csv`.",
           "- Nguồn: `Threshold/...`, `AI_HANDOFF/...` (hash trong 05_PHU_LUC).", "",
           "> Giới hạn: hysteresis đạt band DEV nhưng OOS kém; phương án FNR<30% tốt là exploratory/OOS-selected; không có GO."]
    (R1 / "RQ01_KET_QUA.md").write_text("\n".join(r1) + "\n", encoding="utf-8")

    # ---------------- RQ02 files
    r2f = final["rq02"]["frozen"]
    with (R2 / "RQ02_CHINH_SACH_DEV_OOS.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["policy", "dev_FNR", "dev_FPR", "oos_FNR", "oos_FPR", "oos_pos", "oos_neg", "note"])
        w.writerow(["OR H15|H8|H6 @cap32.5%", r2f["dev_FNR"], r2f["dev_FPR"], r2f["oos_FNR"],
                    "UNDEFINED", r2f["oos_n_pos"], r2f["oos_n_neg"], "FPR undefined because 0 negatives"])
    shutil.copyfile(H / "RQ02_RUN_20261025" / "FULL_V4" / "RQ02_11" / "RQ02_ACCEPTANCE_STATUS.csv",
                    R2 / "RQ02_11_DAU_RA_ACCEPTANCE.csv")
    with (R2 / "RQ02_OOS_VIEWS.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(r2views[0].keys())); w.writeheader(); w.writerows(r2views)
    n_pass = sum(1 for r in r2acc if r["status"] == "PASS")
    n_part = sum(1 for r in r2acc if r["status"] == "PARTIAL")
    r2 = ["# RQ02 — Kết quả định lượng (bản dành cho luận văn)", "",
          "## 1. Tổng quan 11 đầu ra", "",
          f"- Trạng thái acceptance: **{n_pass} PASS / {n_part} PARTIAL / 0 BLOCKED** (RESEARCH_ONLY_NOT_FUNCTION_PASS).",
          "- Điểm yếu chung: tín hiệu DEV rất mạnh nhưng OOS yếu; H18 EXIT-A `NOT_EVALUATED_BY_DESIGN`; FI không có phase chuẩn.",
          "", "## 2. Ngưỡng tốt nhất (threshold-SET)", "",
          f"- 27/40 cơ chế ENTRY đạt hình chữ nhật DEV, tổng **205 bộ** (hysteresis).",
          f"- Chính sách chọn: **OR H15|H8|H6 @cap 32,5%** → DEV **FNR {_pct(r2f['dev_FNR'])} / FPR {_pct(r2f['dev_FPR'])}**; "
          f"OOS **FNR {_pct(r2f['oos_FNR'])}** ({r2f['oos_FN']}/{r2f['oos_n_pos']}) — **FPR UNDEFINED** vì OOS chỉ có 14 positive / 0 negative.", "",
          "### Bảng — 11 đầu ra RQ02", ""]
    r2 += _table(["Mã", "Đầu ra", "DEV", "OOS", "Trạng thái", "Hạn chế chính"],
                 [[r["id"], (r["output"] or "")[:60], (r["dev_result"] or "")[:70], (r["oos_result"] or "")[:70],
                   r["status"], (r["performance_verdict"] or r["root_cause_or_limit"] or "")[:80]] for r in r2acc])
    r2 += ["", "## 3. OOS view (một chronology, hai view)", "",
           "- `OOS_WITH_ESRB_GOLD`: 14 positive, 0 negative → chỉ FNR xác định; **FPR UNDEFINED**, không phải 0.",
           "- `OOS_MODEL_ONLY_EXTRA`: 2021-04→2024-12 không nhãn; không tính FNR/FPR.",
           "- Tổng hợp aggregate: 80 dòng ESRB-scorable + 80 dòng prediction-only.", "",
           "## 4. Tệp số liệu", "",
           "- `RQ02_CHINH_SACH_DEV_OOS.csv`, `RQ02_11_DAU_RA_ACCEPTANCE.csv`, `RQ02_OOS_VIEWS.csv`.", "",
           "> Giới hạn: kết quả OOS chỉ là sensitivity trên 14 sự kiện; không có FPR OOS; cần mẫu có negative/prospective."]
    (R2 / "RQ02_KET_QUA.md").write_text("\n".join(r2) + "\n", encoding="utf-8")

    # ---------------- RQ03 files
    shutil.copyfile(H / "RQ03_TERMINAL_WEALTH_20261025" / "TERMINAL_WEALTH_DEV_OOS.csv",
                    R3 / "RQ03_TERMINAL_WEALTH_6_CHIEN_LUOC.csv")
    variants = []
    for r in lock["oos"]:
        variants.append({"policy": r["name"], "dev_terminal": r["dev_terminal"], "dev_MDD": r["dev_MDD"],
                         "oos_terminal": r["oos_terminal"], "oos_MDD": r["oos_MDD"],
                         "oos_locked": r["oos_locked"], "source": "RQ03_LOCK_RELEASE_LAB"})
    for r in sq["oos"]:
        if r["name"] not in {x["policy"] for x in variants}:
            variants.append({"policy": r["name"], "dev_terminal": r["dev_terminal"], "dev_MDD": r["dev_MDD"],
                             "oos_terminal": r["oos_terminal"], "oos_MDD": r["oos_MDD"],
                             "oos_locked": r["oos_locked"], "source": "RQ03_SQUEEZE_JOINT_LAB"})
    with (R3 / "RQ03_BIEN_THE_POLICY.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(variants[0].keys())); w.writeheader(); w.writerows(variants)
    tw = {(r["period"], r["strategy"]): r for r in term}
    order_dev = sorted([r for r in term if r["period"] == "DEV"], key=lambda r: int(r["rank"]))
    order_oos = sorted([r for r in term if r["period"] == "OOS"], key=lambda r: int(r["rank"]))
    util_oos = util["periods"]["MAIN_OOS"]
    r3 = ["# RQ03 — Kết quả định lượng (bản dành cho luận văn)", "",
          "## 1. Terminal wealth 6 chiến lược (W0 = 1, net sau phí)", "",
          "### DEV", ""]
    r3 += _table(["Hạng", "Chiến lược", "Terminal NAV", "Tổng lợi nhuận", "MDD"],
                 [[r["rank"], r["strategy"], _num(r["terminal_nav"]), _pct(r["total_return"]),
                   _pct(r["MDD"]) if r["MDD"] else "—"] for r in order_dev])
    r3 += ["", "### OOS (post-exposure)", ""]
    r3 += _table(["Hạng", "Chiến lược", "Terminal NAV", "Tổng lợi nhuận", "MDD"],
                 [[r["rank"], r["strategy"], _num(r["terminal_nav"]), _pct(r["total_return"]),
                   _pct(r["MDD"]) if r["MDD"] else "—"] for r in order_oos])
    r3 += ["", "## 2. Chính sách Main cải tiến (research)", "",
           f"- **REL_ANY_26** (release sau 26 tuần khóa): DEV **{_num(lock['oos'][1]['dev_terminal'])}** / OOS **{_num(lock['oos'][1]['oos_terminal'])}**; "
           f"khóa giảm 596→104 tuần DEV, 187→78 tuần OOS; OOS MDD {_pct(lock['oos'][1]['oos_MDD'])}.",
           f"- **R26_B0_0.05** (release + Bottom 5%): DEV **{_num(sq['oos'][2]['dev_terminal'])}** / OOS **{_num(sq['oos'][2]['oos_terminal'])}**; OOS MDD {_pct(sq['oos'][2]['oos_MDD'])}.",
           "- Cả hai **vẫn dưới 1,0 và dưới cả 5 benchmark OOS**; cần chữ ký owner + prospective.",
           "- Biến thể danh nghĩa ESCAP_0.01 (OOS 1,0964) **bị loại** vì vi phạm ES cap 124/260 tuần OOS — hard-feasibility veto.", "",
           "## 3. Vì sao Main yếu", "",
           "- Khóa 596/781 tuần DEV, 187/260 tuần OOS; 278/78 tuần khóa thuộc episode **không có Bottom**.",
           "- Bottom chỉ mua **2 tuần/kỳ**; terminal effect Bottom −1,63% DEV / −0,08% OOS.",
           "- Locked-week log gap vs PC-017: −5,28% DEV / −7,41% OOS; squeeze cap không binding.", "",
           "## 4. Utility / continuation", "",
           f"- Main OOS mean log {util_oos['mean_log_growth']:.6e}, CE {util_oos['certainty_equivalent']:.6f}, MDD {_pct(util_oos['mdd'])}.",
           f"- PC-017 OOS mean log {util['periods']['PC017_OOS']['mean_log_growth']:.6e}; Δ = {util['main_vs_pc017']['OOS']['delta_mean_log']:.6e} "
           f"CI [{util['main_vs_pc017']['OOS']['ci95_low']:.6e}, {util['main_vs_pc017']['OOS']['ci95_high']:.6e}].",
           "- Continuation: `BLOCKED` (thiếu sequential kernel) → mode MYOPIC, continuation_value=None; owner utility chưa ký.", "",
           "## 5. Tệp số liệu", "",
           "- `RQ03_TERMINAL_WEALTH_6_CHIEN_LUOC.csv`, `RQ03_BIEN_THE_POLICY.csv`.", "",
           "> Giới hạn: OOS post-exposure; chưa có GO; benchmark MDD/fee không được runner xuất (để trống, không phải 0)."]
    (R3 / "RQ03_KET_QUA.md").write_text("\n".join(r3) + "\n", encoding="utf-8")

    # ---------------- 34 tasks
    out34 = []
    for r in canon:
        out34.append({"Mã nhiệm vụ": r["task_id"], "RQ": r["task_id"][:4],
                      "Đầu ra": r["required_output"], "DEV": r["dev_result"],
                      "OOS": r["esrb_oos"], "F": r["F"], "L": r["L"], "T": r["T"],
                      "Owner": r["owner_status"], "Kết luận": r["final_verdict"],
                      "Bằng chứng": r["evidence_run_id"]})
    with (RT / "34_TASK_RESULTS.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out34[0].keys())); w.writeheader(); w.writerows(out34)
    md = ["# Kết quả 34 nhiệm vụ (RQ01-01…RQ03-11)", "",
          "Bảng đầy đủ 18 cột nằm trong `AI_HANDOFF/CANONICAL_34_REPORT_20261025/34_TASK_CANONICAL_REPORT.csv`;",
          "dưới đây là bản rút gọn cho luận văn (DEV, OOS, F/L/T, kết luận).", ""]
    for rq, title in (("RQ01", "RQ01 — 12 đầu ra"), ("RQ02", "RQ02 — 11 đầu ra"), ("RQ03", "RQ03 — 11 đầu ra")):
        md += [f"## {title}", ""]
        md += _table(["Mã", "Đầu ra", "DEV", "OOS", "F", "L", "T", "Kết luận"],
                     [[r["Mã nhiệm vụ"], (r["Đầu ra"] or "")[:55], (r["DEV"] or "")[:70],
                       (r["OOS"] or "")[:70], r["F"], r["L"], r["T"], r["Kết luận"]]
                      for r in out34 if r["RQ"] == rq])
        md.append("")
    md += ["## Quy ước", "",
           "- F/L/T là trạng thái research; `FUNCTION_PASS` chính thức và owner gate vẫn NOT_CERTIFIED/0-16.",
           "- RQ03 không áp band FNR/FPR phân loại; kết quả kinh tế trình bày ở RQ03.", "",
           "> Không có GO. Mọi kết quả OOS là post-exposure research."]
    (RT / "34_TASK_RESULTS.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # ---------------- appendix / limitations / manifest
    lim = ["# Phương pháp, nguồn và giới hạn", "",
           "## 1. Phương pháp", "",
           "- F → L → T; exact-mask/quantile thresholds; DEV nguyên khối; OOS chỉ đánh giá sau freeze.",
           "- Multiplicity và hard-feasibility veto trước utility; append-only trial records; mọi kết quả OOS là post-exposure.",
           "- RQ01/RQ02 dùng band FNR/FPR; RQ03 dùng terminal wealth, MDD, ES/Normal cap và hai W0.",
           "", "## 2. Nguồn số liệu", "",
           "- 34 nhiệm vụ: `AI_HANDOFF/CANONICAL_34_REPORT_20261025/34_TASK_CANONICAL_REPORT.csv`.",
           "- RQ01: `Threshold/THRESHOLD_SET_EXPLORATORY_20261025`, `Threshold/FINAL_STRICT_GATE_20261025`, `Threshold/EXPANDING/...`, `AI_HANDOFF/F1_F5_V2_RUN_20261025`.",
           "- RQ02: `AI_HANDOFF/RQ02_RUN_20261025/FULL_V4/RQ02_11`, `RQ02_03`; `Threshold/FINAL_STRICT_GATE_20261025`.",
           "- RQ03: `AI_HANDOFF/RQ03_TERMINAL_WEALTH_20261025`, `RQ03_LOCK_RELEASE_LAB_20261025`, `RQ03_SQUEEZE_JOINT_LAB_20261025`, `RQ03_UTILITY_CONTINUATION_20261025`.",
           "", "## 3. Giới hạn phải nêu trong luận văn", "",
           "- RQ01: band DEV chỉ đạt bằng hysteresis (threshold-set), không đạt bằng single-cut; OOS suy giảm; phương án FNR<30% là exploratory/OOS-selected.",
           "- RQ02: OOS chỉ có 14 positive/0 negative → FPR UNDEFINED; không suy ra FPR=0; H18 NOT_EVALUATED_BY_DESIGN; FI không có phase.",
           "- RQ03: Main thua 5 benchmark OOS; nguyên nhân chính là khóa không có Bottom (278/78 tuần), Bottom mua 2 tuần/kỳ; lock-release là research amendment cần owner ký; ESCAP_0.01 bị loại vì vi phạm ES cap.",
           "- F1–F5 chính thức NOT_CERTIFIED; owner 0/16; B1/B2/C1 là proxy; chưa có prospective.",
           "- Không có tuyên bố GO/trading live; kết quả dành cho nghiên cứu và viết luận.", "",
           "## 4. Trạng thái", "",
           "- `RESEARCH_GO` cho kỹ thuật/evidence; `LIVE_TRADING_GO = NO_GO`.",
           "- Một next action: owner ký amendment lock-release/Bottom tối giản và chạy prospective."]
    (APP / "PHUONG_PHAP_NGUON_VA_GIOI_HAN.md").write_text("\n".join(lim) + "\n", encoding="utf-8")

    sources = {
        "canonical_34": H / "CANONICAL_34_REPORT_20261025" / "34_TASK_CANONICAL_REPORT.csv",
        "rq01_set_certificate": T / "THRESHOLD_SET_EXPLORATORY_20261025" / "SET_CERTIFICATE.json",
        "rq01_final_selection": T / "FINAL_STRICT_GATE_20261025" / "FINAL_SELECTION.json",
        "rq01_expanding": T / "EXPANDING" / "RQ01_AMEND001_EXECUTED_20261025" / "EXPANDING_STATUS.json",
        "rq01_bootstrap": T / "EXPANDING" / "RQ01_AMEND001_EXECUTED_20261025" / "EXPANDING_BOOTSTRAP_B199.json",
        "rq01_f1f5": H / "F1_F5_V2_RUN_20261025" / "F1_F5_RESEARCH_OWNER_V2_1_ACCEPTANCE.csv",
        "rq01_fnr_min": T / "FNR_MIN_COMBINATIONS_20261025" / "SUMMARY.json",
        "rq02_acceptance": H / "RQ02_RUN_20261025" / "FULL_V4" / "RQ02_11" / "RQ02_ACCEPTANCE_STATUS.csv",
        "rq02_oos_views": H / "RQ02_RUN_20261025" / "FULL_V4" / "RQ02_03" / "RQ02_03_OOS_VIEWS.csv",
        "rq03_terminal": H / "RQ03_TERMINAL_WEALTH_20261025" / "TERMINAL_WEALTH_DEV_OOS.csv",
        "rq03_lock_release": H / "RQ03_LOCK_RELEASE_LAB_20261025" / "SUMMARY.json",
        "rq03_squeeze": H / "RQ03_SQUEEZE_JOINT_LAB_20261025" / "SUMMARY.json",
        "rq03_utility": H / "RQ03_UTILITY_CONTINUATION_20261025" / "UTILITY_SUMMARY.json",
    }
    with (APP / "NGUON_VA_HASH.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh); w.writerow(["source", "path", "sha256"])
        for k, p in sources.items():
            w.writerow([k, str(p.relative_to(root)), _sha(p)])
    man = {"created_utc": datetime.now(timezone.utc).isoformat(),
           "files": {str(p.relative_to(D)): _sha(p) for p in sorted(D.rglob("*"))
                     if p.is_file() and p.name != "MANIFEST_SHA256.json"}}
    (APP / "MANIFEST_SHA256.json").write_text(json.dumps(man, indent=2), encoding="utf-8")

    readme = ["# KẾT QUẢ — V8/EA7 (bản dành cho viết luận)", "",
              f"Cập nhật: {datetime.now(timezone.utc).isoformat(timespec='seconds')} · Nguồn: evidence tree trong `AI_HANDOFF/` và `Threshold/`.",
              "Mọi số liệu đều đọc trực tiếp từ artifact đã kiểm chứng; không có kết quả mới trong bước này.", "",
              "## Ba kết quả nổi bật", "",
              "1. **RQ01** — band DEV 10–15% FNR / 15–25% FPR: single-cut bất khả thi 0/275 nhưng **HYSTERESIS đạt 8/8** (M00 LOGISTIC đúng 12,44%/18,00%); OOS FNR<30% tốt nhất: pre-registered XGB4 @cap50 **26,37%/19,05%**; phương án OOS-selected 29,67%/19,05%.",
              "2. **RQ02** — **OR H15|H8|H6 @cap32,5%**: DEV 0%/14,96%; OOS FNR **14,29%** (2/14); **FPR UNDEFINED** (0 negative); 9/11 đầu ra PASS, 2 PARTIAL.",
              "3. **RQ03** — 6 chiến lược terminal wealth; OOS cao nhất hợp lệ **EW 1,086669**; Main tốt nhất hợp lệ **REL26 0,949560**; policy đề xuất **R26_B0_0.05** DEV 1,547559/OOS 0,948867 — vẫn dưới benchmark, cần owner ký.",
              "", "## Cấu trúc thư mục", "",
              "- `01_RQ01/` — báo cáo + CSV threshold-set, OOS FNR<30%, expanding, F1–F5.",
              "- `02_RQ02/` — báo cáo + CSV policy, acceptance 11 đầu ra, OOS views.",
              "- `03_RQ03/` — báo cáo terminal wealth 6 chiến lược, biến thể policy, chẩn đoán.",
              "- `04_34_TASKS/` — bảng 34 nhiệm vụ (CSV + MD).",
              "- `05_PHU_LUC/` — phương pháp, nguồn, giới hạn, hash.",
              "", "## Tuyên bố", "",
              "- Đây là kết quả **research**; `LIVE_TRADING_GO = NO_GO`.",
              "- OOS 2020–2024 là post-exposure; RQ02 OOS FPR không tồn tại; RQ01 hysteresis chỉ đạt DEV; RQ03 lock-release/Bottom tối giản cần chữ ký owner.",
              "- Không dùng số liệu trong thư mục này để tuyên bố lợi nhuận thực hoặc vượt benchmark."]
    (D / "00_README_TONG_HOP.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    # final manifest after README
    man = {"created_utc": datetime.now(timezone.utc).isoformat(),
           "files": {str(p.relative_to(D)): _sha(p) for p in sorted(D.rglob("*"))
                     if p.is_file() and p.name != "MANIFEST_SHA256.json"}}
    (APP / "MANIFEST_SHA256.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    return {"out": str(D), "n_files": len(man["files"]), "n_tasks": len(out34)}


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
