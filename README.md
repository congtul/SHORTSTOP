# ShortStop

Ghi chú tái lập cá nhân cho paper *Counterexample-Guided Short-Horizon
Shielding of Generative Robot Policies*. Kế hoạch nghiên cứu đầy đủ (lý
thuyết) nằm ở [docs/ShortStop_Research_Plan.tex](docs/ShortStop_Research_Plan.tex);
lộ trình thực thi theo stage (Track A/Track B ghép cặp, bản mới nhất) nằm ở
[docs/ShortStop_Report_1.tex](docs/ShortStop_Report_1.tex). README này chỉ
tóm tắt phần cần để setup/chạy code và nắm nhanh lý thuyết.

## 1. Lý thuyết tóm tắt

**Bài toán**: một generative robot policy (diffusion/flow) đã huấn luyện sẵn,
đông cứng (frozen, không retrain), cần được làm an toàn *tại runtime* — trong
khi verify hình thức toàn bộ policy là bất khả thi. Giải pháp: chỉ verify
*hậu quả ngắn hạn* (short-horizon) của một action chunk cụ thể mà policy vừa
đề xuất, rồi lọc/sửa chunk đó trước khi thực thi.

**4 bước P-R-C-S** (Algorithm 1 trong paper):

1. **Propose** — sinh $K$ action chunk ứng viên (độ dài $H$) từ policy.
2. **Reach** — lan truyền reachable-set *over-approximation* cho mỗi chunk qua
   horizon $H$ (công thức 1):
   $\hat R_{k+1} = \hat f(\hat R_k, a_{t+k}) \oplus B(\varepsilon_k + \bar w)$
3. **Certify** — tính STL robustness-to-go trên cả tube, không chỉ 1
   trajectory nominal (công thức 2):
   $\rho(\phi,\hat R) = \min_{0\le k\le H} \inf_{x\in\hat R_k} \mathrm{dist}(x,X_u)$
4. **Select / repair** — chọn chunk tốt nhất trong tập an toàn (admissible);
   nếu không chunk nào an toàn, tìm counterexample rồi sửa bằng gradient step
   có trust region; nếu vẫn không sửa được, gọi fallback controller
   $\pi_{fb}$ (braking/retreat).

**Lộ trình thực nghiệm (Track B)** chia thành 11 stage, đi từ dễ đến khó,
đúng theo ablation của paper (Table III): `Reach-only` → `+STL-to-go` →
`+CE search` → `+Repair` → `baselines`, tất cả build/debug trên
`Reach-Avoid-2D` (point-mass 2D tự dựng) với một **policy Gaussian giả lập**
(rẻ, dễ debug). Chỉ sau khi toàn bộ shield logic đã chạy đúng mới **train/import
một generative policy 2D thật** (stage 6a) để validate lại giả định
"black-box proposer" (stage 6b), rồi mới leo lên 3 benchmark thật — LIBERO,
ManiSkill2, RLBench — mỗi env tách thành một stage riêng (7a/7b/7c, cần GPU +
Linux/WSL2). Chi tiết ghép cặp với lý thuyết Track A: xem bảng trong
[docs/ShortStop_Report_1.tex](docs/ShortStop_Report_1.tex), section
"Track A \& Track B".

Repo hiện tại (nhánh `2d-prototype`) đã có **Stage 0--4**: setup, Reach-only
rejector, STL-to-go, CE search, và full repair loop, tất cả trên
`Reach-Avoid-2D` với policy Gaussian giả lập.

## 2. Cấu trúc thư mục

```
SHORTSTOP/
├── docs/                        # Tài liệu tham khảo (không phải code)
│   ├── ShortStop_Research_Plan.tex
│   ├── main (3).pdf              # Paper gốc
│   └── main (3).txt
├── shortstop/                    # Package chính
│   ├── env.py                    # ReachAvoid2D: point-mass 2D + obstacle tròn
│   ├── reach.py                  # Box (interval) + propagate() theo công thức (1)
│   ├── policy.py                 # GaussianChunkPolicy: sinh K chunk giả lập (chưa phải diffusion thật)
│   ├── stl.py                    # robustness_to_go() + find_counterexample() theo công thức (2)-(3)
│   ├── shield.py                 # ReachOnlyShield / STLShield / CEShield / RepairShield (Stage 1-4)
│   ├── experiment.py             # run_episode() dùng chung cho mọi stage (cùng seed để so sánh công bằng)
│   ├── calibration.py            # calibrate_w_bar(): Table VII's high-quantile calibration recipe
│   └── metrics.py                # 7 metric + latency mean/p95: violation/success/activation/latency/precision/recovery/cons.cost
├── scripts/
│   ├── run_phase1.py             # Thực nghiệm Stage 1 riêng (tên file giữ từ "Phase 1" cũ) + vẽ hình
│   ├── run_ablation.py           # Bảng so sánh Unshielded + Stage 1-4 cùng lúc (config = dict STAGES)
│   └── run_horizon_sweep.py      # Quét certified horizon H (conservatism-horizon trade-off, Fig. 4)
├── tests/                        # pytest cho từng module
│   ├── test_env.py
│   ├── test_reach.py
│   ├── test_stl.py
│   ├── test_shield.py
│   ├── test_calibration.py
│   └── test_metrics.py
├── results/                      # Output khi chạy script (gitignore, tự sinh)
├── requirements.txt
└── .gitignore
```

Mỗi module ứng với đúng một phần trong Algorithm 1: `policy.py` là bước
Propose, `reach.py` là bước Reach, `stl.py` + `shield.py` là bước Certify +
Select/Repair. Bốn shield class trong `shield.py` là 4 stage lồng nhau
(`ReachOnlyShield` → `STLShield` → `CEShield` → `RepairShield`, mỗi stage kế
thừa trực tiếp từ stage ngay trước, đúng theo thứ tự ablation của paper):

| Class | Stage | Certify bằng | Repair? |
|---|---|---|---|
| `ReachOnlyShield` | 1 | giao reachtube/obstacle (nhị phân) | không |
| `STLShield` | 2 | STL robustness-to-go $\ge\varepsilon$ | không |
| `CEShield` | 3 | giống Stage 2 + trả về counterexample cho chunk bị reject | không (chỉ định vị) |
| `RepairShield` | 4 | giống Stage 2 + sửa chunk bị reject bằng gradient step trong trust region rồi certify lại | có |

`RepairShield` có tham số `max_repair_iters` (mặc định **1**): default này khớp
đúng Algorithm 1 của paper — sửa đúng 1 bước gradient, certify lại đúng 1
lần, không đạt thì bỏ candidate luôn, không thử lại. Đặt `max_repair_iters >
1` là mở rộng **vượt ra ngoài paper** (theo tinh thần CEGIS cổ điển: lặp lại
tìm-counterexample-rồi-sửa nhiều vòng cho tới khi hết counterexample) — dùng
để thử nghiệm, không phải hành vi mặc định khi so khớp với paper.

Công thức (4) của paper thật ra có **2** tham số riêng biệt (Table VII:
`η=0.05`, `δ=0.1`), không phải một: `a' = Π_{A,δ}(a + η·d)`. `RepairShield`
tách đúng như vậy — `step_size` (η) là độ lớn mỗi bước gradient,
`trust_region` (δ) là bán kính chặn **tổng độ lệch tích lũy** so với
candidate gốc, không phải độ lớn mỗi bước.

Vì scale hành động của prototype 2D này (radius obstacle ~0.4–0.8, action
magnitude ~1.0) không cùng đơn vị với paper (cm), giá trị `η=0.05`/`δ=0.1`
"đúng paper" khá nhỏ so với những gì cần để sửa một va chạm (sweep 150
episode: recovery chỉ ~6%). `run_ablation.py` nên mặc định
**`step_size=0.3`, `trust_region=1.0`** — tune riêng cho scale prototype này
(recovery ~17%, violation rate không đổi vì mọi chunk sửa xong vẫn bị
certify lại như cũ) — không phải số của paper. Đặt lại `0.05`/`0.1` trong
file đó nếu muốn tái lập đúng hyperparameter của paper (xem comment ngay
trên khai báo `STEP_SIZE` để biết bảng sweep đầy đủ).

**Lưu ý quan trọng**: trong bản 2D này, Stage 2 và Stage 3 ra **cùng** kết quả
an toàn (`CEShield` chỉ thêm chẩn đoán counterexample, không đổi quyết định
accept/reject) — vì obstacle tròn + reachtube box nên "tìm counterexample" có
nghiệm dạng đóng (closed-form), không cần adversarial search thật sự để thay
đổi kết quả. Số liệu chỉ thay đổi thêm lần nữa ở Stage 4, khi repair thực sự
sửa được chunk thay vì chỉ loại bỏ nó. Xem docstring của từng class trong
`shortstop/shield.py` để biết chi tiết.

## 3. Setup (Stage 0)

Yêu cầu: Python 3.10+ (đã test với 3.14.7). Thuần Python, không phụ thuộc
OS — chạy được trên Windows, Linux, hoặc WSL2.

```bash
# Tạo virtual env
python -m venv .venv

# Kích hoạt
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Windows cmd:
.venv\Scripts\activate.bat
# Linux / WSL2 / Git Bash:
source .venv/bin/activate

# Cài dependencies
pip install -r requirements.txt
```

Thư viện chính trong [requirements.txt](requirements.txt): `numpy`, `scipy`,
`matplotlib` (vẽ hình), `cvxpy` (QP solver, dùng cho baseline
MPC-Filter/CBF-Shield ở Stage 5), `pytest` (test).

## 4. Cách chạy

**Chạy test** (kiểm tra box propagation, circle intersection, env dynamics,
shield reject/fallback logic):

```bash
pytest tests/ -v
```

Kỳ vọng: 19/19 test pass.

**Chạy thực nghiệm Stage 1 riêng** (có vẽ hình trajectory + decision snapshot):

```bash
python scripts/run_phase1.py
```

Script sẽ:
- Chạy 500 episode ngẫu nhiên, in ra metrics so sánh `Unshielded` vs
  `Reach-only rejector` (violation rate, success rate, shield-activation
  rate).
- Lưu hình trajectory mẫu + decision snapshot vào `results/phase1_*.png`.

**Chạy bảng so sánh Stage 1--4 (ablation, để so kết quả qua các stage)**:

```bash
python scripts/run_ablation.py
```

Script chạy `Unshielded` + `Stage 1` (`ReachOnlyShield`) + `Stage 2`
(`STLShield`) + `Stage 3` (`CEShield`) + `Stage 4` (`RepairShield`) trên
**cùng một bộ seed** (nhờ `shortstop/experiment.py` dùng chung), in bảng
(violation/success/activation/lat_median/lat_mean/lat_p95/precision/recovery/cons.cost —
3 cột latency vì median một mình có thể che mất đuôi phân phối đắt, xem giải
thích ở docstring `RepairShield` trong `shortstop/shield.py`) và lưu bản
JSON vào `results/ablation.json`. Thêm stage mới chỉ cần thêm 1 dòng
vào dict `STAGES` ở đầu file — không phải sửa logic chạy episode.

Mặc định (`USE_CALIBRATED_W_BAR=True`), script còn tự **calibrate** disturbance
bound `w_bar` mà shield được certify theo (`shortstop/calibration.py`, đúng
công thức Table VII "high-quantile residual × safety factor"), thay vì đưa
thẳng giá trị `w_bar` thật (ground-truth) cho shield như trước — vì một
shield triển khai thực tế không biết trước nhiễu thật của môi trường. Tham
số `CALIBRATION_EPISODES`/`CALIBRATION_QUANTILE`/`CALIBRATION_SAFETY_FACTOR`
ở đầu file điều khiển việc này; đặt `USE_CALIBRATED_W_BAR=False` để quay lại
hành vi cũ (privileged w_bar).

**Quét certified horizon H (conservatism–horizon trade-off, Fig. 4)**:

```bash
python scripts/run_horizon_sweep.py
```

Paper chứng minh (Prop. 1) và quan sát (Fig. 4) rằng khi H tăng, violation
giảm mạnh rồi bão hòa, còn activation/latency tăng đơn điệu — tồn tại một
"điểm ngọt" (paper: H≈8). Script này quét `HORIZONS` (mặc định
`[4,6,8,10,12,16]`) cho các stage trong `SWEEP_STAGES` (mặc định Stage 1 +
Stage 4) để xem prototype 2D này có tái hiện đúng hình dạng đường cong đó
không — H tối ưu không nhất thiết trùng H=8 của paper vì scale
obstacle/policy khác nhau. Kết quả lưu vào `results/horizon_sweep.json`.

**Lưu ý về số liệu**: violation/success rate in ra sẽ không khớp hệt Table
III của paper (3.8% / 64.1% / 2.9% / 1.7% / 1.2% qua từng stage), vì paper
không công bố đầy đủ cấu hình env 2D gốc (layout obstacle, $\bar w$, action
bound). Các hằng số trong `make_scenario()` (`shortstop/experiment.py`) và
`EPSILON`/`TRUST_REGION`/`MAX_REPAIR_ITERS` (`scripts/run_ablation.py`) là
placeholder hợp lý để pipeline chạy đúng logic; có thể tinh chỉnh khi cần bám
sát số liệu paper hơn.

## 5. Trạng thái / roadmap

Stage theo đúng bảng Track A/Track B trong
[docs/ShortStop_Report_1.tex](docs/ShortStop_Report_1.tex) (bản mới nhất,
thay cho Phase 0--6 cũ):

- [x] **Stage 0** — Setup: env `Reach-Avoid-2D`, interval arithmetic,
      `cvxpy`; policy Gaussian giả lập chỉ để dựng khung.
- [x] **Stage 1** — Reach-only shield: reachtube, binary reject (Gaussian).
- [x] **Stage 2** — STL-to-go: thay reject nhị phân bằng STL robustness
      lower bound (Gaussian) — `shortstop/stl.py` + `STLShield`.
- [x] **Stage 3** — CE search: định vị counterexample (closed-form, đứng
      vai trò của adversarial search) trên box 2D (Gaussian) — `CEShield`.
- [x] **Stage 4** — Repair: gradient step + trust region, certify lại, có
      recovery-rate metric (Gaussian) — `RepairShield`.
- [ ] **Stage 5** — Baselines: MPC-Filter, CBF-Shield, STL-Monitor,
      Conf-Thresh (vẫn trên Gaussian).
- [ ] **Stage 6a** — 2D policy thật: train một diffusion/flow policy nhỏ
      trên `Reach-Avoid-2D`, hoặc import policy có sẵn (thay Gaussian).
- [ ] **Stage 6b** — Metrics + stress test: 7 metric, horizon sweep,
      bootstrap — chạy lại với policy 2D thật.
- [ ] **Stage 7a** — LIBERO (swap sang env thật, long-horizon manipulation).
- [ ] **Stage 7b** — ManiSkill2 (swap sang env thật, contact-rich precision).
- [ ] **Stage 7c** — RLBench (swap sang env thật, diverse manipulation tasks).

Stage 7a--7c cần GPU + Linux/WSL2. Chi tiết đầy đủ từng module lý thuyết
(Track A, 7 nhóm chủ đề ghép với từng stage) và lộ trình theo tuần xem
[docs/ShortStop_Research_Plan.tex](docs/ShortStop_Research_Plan.tex) và
[docs/ShortStop_Report_1.tex](docs/ShortStop_Report_1.tex).
