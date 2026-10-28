# RQ01–RQ03 research code — DEV / OOS evidence

Mã nguồn cho chuỗi ba câu hỏi nghiên cứu trên bảy quốc gia khu vực đồng euro:
xác nhận trạng thái căng thẳng (RQ01) → dự báo chuyển trạng thái và vùng đáy (RQ02)
→ phân bổ danh mục theo trạng thái (RQ03).

```
RAW_CODE/     entrypoint run_canonical.py, rev363/, research_adapters/, rq02_source/, rq03_standard/
THRESHOLD/    RQ01, RQ02, RQ03 selected configurations (JSON)
scripts/      build_results.py, build_rq03_standard.py
results/      bảng CSV và hình PNG của DEV/OOS
```

## Chạy lại

```bash
pip install -r requirements.txt
# RQ01–RQ02 và bộ 34 artefact
python RAW_CODE/run_canonical.py --dataset-root <ASSET03> --out <out> --mode dev
python RAW_CODE/run_canonical.py --dataset-root <ASSET03> --out <out> --mode oos     --oos-authorization <DEC-015 owner token>
python scripts/build_results.py --run <out> --out results

# RQ03 chuẩn S4BOT (cần cây AI_HANDOFF của dự án)
RQ_PROJECT_ROOT=<project> RQ_OUT=results/rq03_standard     python RAW_CODE/rq03_standard/build_chosen_standard.py
python scripts/build_rq03_standard.py --ledger-dir results/rq03_standard     --handoff <project>/AI_HANDOFF --dataset <ASSET03> --out results
```

## RQ01 — FNR/FPR theo sáu pha, tín hiệu riêng lẻ và quyết định pool

Sáu pha: trước vào, vào, đỉnh, giảm áp lực, phục hồi, thoát. "Individual" là phiếu của tầng
chỉ báo riêng lẻ; "pool" là quyết định sau khi hợp phiếu individual/joint.

| period | phase_vi | n_pool | pool_FNR | pool_FPR | n_ind | ind_FNR | ind_FPR |
|---|---|---|---|---|---|---|---|
| DEV | Trước vào | 148 | 0.00 | 0.68 | 78 | 0.00 | 66.23 |
| DEV | Vào | 44 | 0.00 | 7.89 | 44 | 0.00 | 100.00 |
| DEV | Đỉnh | 62 | 6.67 | 0.00 | 62 | 0.00 | 93.62 |
| DEV | Giảm áp lực | 86 | 11.11 | 2.00 | 86 | 0.00 | 92.00 |
| DEV | Phục hồi | 429 | 2.19 | 3.98 | 429 | 14.04 | 84.08 |
| DEV | Thoát | 449 | 7.58 | 2.21 | 449 | 70.45 | 28.71 |
| DEV | Toàn bộ | 1218 | 4.78 | 2.50 | 1148 | 29.90 | 60.14 |
| OOS | Trước vào | 12 | — | 0.00 | 0 | — | — |
| OOS | Vào | 31 | 42.86 | 40.00 | 0 | — | — |
| OOS | Đỉnh | 22 | 85.71 | 75.00 | 0 | — | — |
| OOS | Giảm áp lực | 25 | 85.71 | 72.73 | 0 | — | — |
| OOS | Phục hồi | 286 | 95.24 | 57.79 | 0 | — | — |
| OOS | Thoát | 44 | — | 18.18 | 0 | — | — |
| OOS | Toàn bộ | 420 | 80.22 | 50.76 | 0 | — | — |

Mô hình đạt FNR 4.78% / FPR 2.50% trên DEV,
và FNR 80.22% / FPR 50.76% trên khối OOS —
độ chính xác giảm mạnh ngoài mẫu, đặc biệt ở các pha phục hồi và thoát.

![RQ01](results/fig_rq01_phase_pool.png)

## RQ02 — H15 → last short, và vùng đáy

| period | horizon | n | events | auc | FNR | FPR |
|---|---|---|---|---|---|---|
| DEV | H6 | 716 | 108 | 1.00 | 1.85 | 0.00 |
| DEV | H8 | 674 | 144 | 1.00 | 2.08 | 0.00 |
| DEV | H10 | 632 | 180 | 1.00 | 3.89 | 1.11 |
| DEV | H12 | 595 | 209 | 1.00 | 1.44 | 0.00 |
| DEV | H15 | 546 | 230 | 1.00 | 0.00 | 0.00 |
| OOS | H6 | 203 | 14 | 0.36 | 64.29 | 64.55 |
| OOS | H8 | 161 | 14 | 0.38 | 57.14 | 60.54 |
| OOS | H10 | 119 | 14 | 0.24 | 64.29 | 70.48 |
| OOS | H12 | 77 | 14 | 0.12 | 64.29 | 85.71 |
| OOS | H15 | 14 | 14 | — | 50.00 | — |

Tín hiệu last short (S5) so với nhãn căng thẳng:

| period | TP | FN | FP | TN | FNR | FPR | n |
|---|---|---|---|---|---|---|---|
| DEV | 418 | 0 | 499 | 343 | 0.00 | 59.26 | 1260 |
| OOS | 65 | 26 | 245 | 84 | 28.57 | 74.47 | 420 |

Vùng đáy:

| period | HIT | MISS | hit_rate | status |
|---|---|---|---|---|
| DEV | 15.00 | 21.00 | 41.67 | SCORED |
| OOS | — | — | — | OOS_BOTTOM_SEALED_NOT_SCORED |

`OOS_BOTTOM_SEALED_NOT_SCORED`: khối OOS của vùng đáy bị niêm phong theo hợp đồng nghiên cứu,
hệ thống ghi trạng thái thay vì tạo ra số.

![RQ02](results/fig_rq02_horizon.png)

## RQ03 — NAV của mô hình chính (chuẩn S4BOT) so với năm đối chứng và 100% cổ phiếu

Chuẩn được chọn: **W16_S4BOT_Z2 + ENTRY H12 + EXIT-3m theo pha RQ03** (ngưỡng EXIT 0,027174,
phí 10 bp một chiều). Kết quả tái lập được từ `build_chosen_standard.py`:

| Chỉ tiêu | DEV | OOS | Toàn mẫu |
|---|---:|---:|---:|
| NAV | **3,9015** | **1,8059** | **5,6981** |
| MDD | 0,3283 | 0,1430 | 0,3283 |
| ES95 tuần | 0,0452 | 0,0395 | — |
| NAV cuối kỳ của 5 đối chứng | 1,293 – 1,678 | 0,989 – 1,087 | — |
| 100% cổ phiếu (EW ba chỉ số NTR) | 2,596 | 1,261 | — |

| period | weeks | POLICY | EW | IVOL | MVP | ERC | MaxDiv | EQUITY100 |
|---|---|---|---|---|---|---|---|---|
| DEV | 781 | 3.90 | 1.68 | 1.36 | 1.29 | 1.45 | 1.37 | 2.60 |
| OOS | 260 | 1.81 | 1.09 | 1.05 | 0.99 | 1.05 | 1.04 | 1.26 |

Số tuần theo từng pha vòng đời:

| period | weeks_normal | weeks_to_short | weeks_to_bottom | weeks_to_exit |
|---|---|---|---|---|
| DEV | 52 | 42 | 596 | 91 |
| OOS | 30 | 13 | 187 | 30 |


![RQ03 NAV](results/fig_rq03_nav_dev_oos.png)
![RQ03 terminal](results/fig_rq03_terminal.png)

Cách đọc hình: nền xanh nhạt = bình thường; cam = bình thường → last short; đỏ = last short →
vùng đáy; xanh lá = vùng đáy → thoát. Nét liền đỏ là mô hình chính; nét đứt là năm danh mục
đối chứng; nét chấm xám là 100% cổ phiếu. Ở panel DEV các đối chứng chỉ có NAV cuối kỳ nên
được vẽ bằng nét ngang; panel OOS có đường theo tuần.

Mô hình chính vượt cả năm đối chứng và 100% cổ phiếu ở cả hai giai đoạn: DEV 3,9015 so với
1,2934–1,6784 (đối chứng) và 2,5960 (100% cổ phiếu); OOS 1,8059 so với 0,9892–1,0867 và
1,2611. Đánh đổi: MDD DEV 32,8% và ES95 4,52%, cao nhất trong nhóm — lợi nhuận đến từ việc
duy trì tỷ trọng cổ phiếu lớn chứ không phải từ việc cắt rủi ro.

## Hai thế hệ mã trong repo

| Thành phần | Vai trò | Kết quả |
|---|---|---|
| `RAW_CODE/rev363` + `run_canonical.py` | pipeline REV363: RQ01, RQ02 và đủ 34 artefact (`34_OUTPUT_REPORT`) | RQ01/RQ02 ở trên |
| `RAW_CODE/rq03_standard` | bộ sinh chuẩn RQ03 (W16 S4BOT + overlay H12/EXIT-3m) | NAV 3,9015 / 1,8059 / 5,6981 |

`rev363` có engine phân bổ riêng (NAV DEV 1,982) **không** phải chuẩn đã chốt; chuẩn đã chốt
được sinh bởi `rq03_standard` và cần cây `AI_HANDOFF` của dự án. Hai bộ không được trộn số.

## Dữ liệu

Dataset ASSET03 (Euro_Area_Dataset_Mentor_Ready) không được đóng gói trong repo; đường dẫn là
tham số bắt buộc. Không có ngưỡng nào bị suy diễn tại chỗ: mọi giá trị nằm trong `THRESHOLD/`.
