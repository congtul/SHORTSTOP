# ShortStop

Ghi chú tái lập cá nhân cho paper *Counterexample-Guided Short-Horizon
Shielding of Generative Robot Policies*. Kế hoạch nghiên cứu đầy đủ nằm ở
[docs/ShortStop_Research_Plan.tex](docs/ShortStop_Research_Plan.tex); README
này chỉ tóm tắt phần cần để setup/chạy code và nắm nhanh lý thuyết.

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

**Lộ trình thực nghiệm (Track B)** đi từ dễ đến khó, đúng theo ablation của
paper (Table III): `Reach-only` → `+STL-to-go` → `+CE search` → `+Repair`,
test trước trên môi trường `Reach-Avoid-2D` (point-mass 2D tự dựng) rồi mới
leo lên LIBERO/ManiSkill2/RLBench (Phase 6, cần GPU + Linux/WSL2).

Repo hiện tại (nhánh `2d-prototype`) mới có **Phase 0** (setup) và
**Phase 1** (Reach-only rejector trên Reach-Avoid-2D).

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
│   ├── shield.py                 # ReachOnlyShield: Phase 1 — reject theo giao reachtube/obstacle
│   └── metrics.py                # violation rate / success rate / shield-activation rate
├── scripts/
│   └── run_phase1.py             # Thực nghiệm Phase 1: Unshielded vs Reach-only rejector
├── tests/                        # pytest cho từng module
│   ├── test_env.py
│   ├── test_reach.py
│   └── test_shield.py
├── results/                      # Output khi chạy script (gitignore, tự sinh)
├── requirements.txt
└── .gitignore
```

Mỗi module ứng với đúng một phần trong Algorithm 1: `policy.py` là bước
Propose, `reach.py` là bước Reach, `shield.py` là bước Certify + Select
(Phase 1 mới chỉ certify bằng điều kiện nhị phân "giao obstacle hay không",
chưa có STL robustness — sẽ thêm ở Phase 2).

## 3. Setup (Phase 0)

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
MPC-Filter/CBF-Shield ở Phase 4), `pytest` (test).

## 4. Cách chạy

**Chạy test** (kiểm tra box propagation, circle intersection, env dynamics,
shield reject/fallback logic):

```bash
pytest tests/ -v
```

Kỳ vọng: 7/7 test pass.

**Chạy thực nghiệm Phase 1**:

```bash
python scripts/run_phase1.py
```

Script sẽ:
- Chạy 500 episode ngẫu nhiên, in ra metrics so sánh `Unshielded` vs
  `Reach-only rejector` (violation rate, success rate, shield-activation
  rate).
- Lưu 1 hình trajectory mẫu vào `results/phase1_example.png`.

**Lưu ý về số liệu**: violation/success rate in ra (vd. ~1% / ~32% trong lần
chạy mẫu) sẽ không khớp hệt Table III của paper (3.8% / 64.1%), vì paper
không công bố đầy đủ cấu hình env 2D gốc (layout obstacle, $\bar w$, action
bound). Các hằng số trong `make_scenario()` / `run_episode()` của
`scripts/run_phase1.py` là placeholder hợp lý để pipeline chạy đúng logic;
có thể tinh chỉnh khi cần bám sát số liệu paper hơn.

## 5. Trạng thái / roadmap

- [x] Phase 0 — Setup tooling
- [x] Phase 1 — Reach-Avoid-2D + Reach-only rejector
- [ ] Phase 2 — STL robustness-to-go
- [ ] Phase 3 — Counterexample search
- [ ] Phase 4 — Repair loop + đủ 7 metrics + baselines (Unshielded,
      Conf-Thresh, MPC-Filter, STL-Monitor, CBF-Shield)
- [ ] Phase 5 — Stress test + horizon sensitivity + bootstrap significance
- [ ] Phase 6 — Leo lên LIBERO/ManiSkill2/RLBench (cần GPU, Linux/WSL2)

Chi tiết đầy đủ từng phase, module lý thuyết, và lộ trình theo tuần xem
[docs/ShortStop_Research_Plan.tex](docs/ShortStop_Research_Plan.tex).
