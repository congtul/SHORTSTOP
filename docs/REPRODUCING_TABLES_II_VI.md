# Reproducing Tables II–VI của paper ShortStop — setup notes

Nguồn: đọc trực tiếp `docs/main (3).txt` (Sec. V–VII, Appendix B). Không suy diễn — mọi số liệu/trích dẫn dưới đây lấy đúng nguyên văn từ file đó (đối chiếu số dòng để tự kiểm tra lại).

## 0. Phát hiện quan trọng nhất — đọc trước khi làm gì khác

Ngay trước Sec. VI (Results), paper có đoạn "Reproducibility note" (dòng 384–390):

> *"Reported numbers in Sec. VI–VII are produced by the released prototype and figure scripts. Values on LIBERO/ManiSkill2/RLBench are **model-derived projections** calibrated to published unshielded success/violation rates and to the 2D-sim measurements; PLAN.md specifies exactly how to replace each with a full simulator run."*

Nghĩa là: **số liệu cho LIBERO/ManiSkill2/RLBench trong Table II–VI KHÔNG đến từ việc thật sự chạy policy + ShortStop qua simulator của 3 benchmark đó** — chúng là số được tính/hiệu chỉnh (calibrate) dựa trên:
1. Tỷ lệ unshielded success/violation đã **công bố sẵn** ở các paper khác (paper gốc của LIBERO/ManiSkill2/RLBench, hoặc paper dùng Diffusion Policy làm baseline trên chúng), và
2. Số liệu **thật** đo được trên `Reach-Avoid-2D` (env 2D của chính nhóm tác giả).

Chỉ có cột/dòng `Reach-Avoid-2D` là simulator run thật. `PLAN.md` được nhắc tới như một file hướng dẫn cách thay projection bằng full simulator run — **file này không có trong repo hiện tại**, chỉ là tham chiếu nội bộ của paper.

$\Rightarrow$ **Không thể "tái lập" đúng số của Table II–VI bằng cách chạy simulator thật** — nếu bạn chạy thật, số sẽ khác (đúng như Report 2 của chúng ta đã thấy: chạy Reach-Avoid-2D thật cho ra 0.2%/38.4% thay vì 1.2%/68.4% của paper — khác env, khác policy, khác nguồn số).

### 0b. Công thức calibration cụ thể là gì? — đã tìm, không có

Đã đọc trực tiếp cả 9 trang PDF (`main (3).pdf`, không chỉ bản `.txt` trích chữ), tìm mọi đoạn có "calibrat/project/scale/derive" — **không có công thức, tỉ lệ, hay thuật toán nào được công bố** cho việc tính projection. Cả bài chỉ nhắc đúng 2 lần (Sec VI trang 6, Sec VIII trang 8), gần như y nguyên câu chữ, và luôn dẫn chiếu tới `PLAN.md` — file **không có trong bộ tài liệu này**.

Hai chi tiết liên quan tìm thêm được (đọc trực tiếp trang PDF, không có trong bản `.txt`):

- **Sec V-A (trang 5)**: *"The protected policy is a diffusion policy [1] (a flow-matching [4] variant is used in an ablation) trained by imitation on each suite's demonstrations following standard recipes [59]. All shields wrap the same frozen checkpoint."* ([59] = Mandlekar et al., CoRL 2021 — recipe BC kiểu robomimic.) Đây là **protocol lý tưởng/dự định** (train 1 policy/suite, dùng chung checkpoint cho cả 6 method) — nhưng theo Reproducibility note ở mục 0, protocol này **không thực sự được chạy** để ra số báo cáo cho 3 benchmark thật; nó mô tả cái nên làm, không phải cái đã làm.
- **Trang 2, mục "Artifact" (Contributions)**: *"A runnable prototype... regenerating every figure and table."* — khớp với cách `scripts/run_ablation.py` trong repo này hoạt động (chạy lại, ghi JSON, in bảng) — nhiều khả năng nghĩa là script tái sinh **định dạng** bảng từ số đã lưu (bao gồm phần projection), không phải re-run full simulator mỗi lần.

**Kết luận**: nếu bạn cần công thức calibration để tự làm lại đúng con số của paper — **không tồn tại để lấy**, phải tự định nghĩa cách "chiếu" số 2D sang số manipulation nếu muốn làm theo đúng ý paper, hoặc (khuyến nghị hơn) bỏ qua việc tái lập số cũ, tự chạy full simulator thật để có bộ số của riêng mình (xem mục 3).

---

## 1. Từng bảng: report gì, cần dữ liệu gì để tạo ra số đó

### Table II — Main comparison (dòng 403–415)
- **Nội dung**: Violation/Success/Activation/Precision/Latency/Conservatism-cost/Recovery cho 6 method (Unshielded, Conf-Thresh, MPC-Filter[33], CBF-Shield[38], STL-Monitor[53], ShortStop).
- **Caption gốc**: *"AGGREGATED OVER LIBERO, ManiSkill2, RLBench, AND Reach-Avoid-2D (NOMINAL + ALL STRESS TESTS), MEAN ± STD OVER 5 SEEDS."*
- **Cần gì để tạo ra**: rollout của cả 6 method × 4 env × (nominal + 3 stress test) × 5 seed × 500 episode.
- **Baseline có bị stress test không?** → **CÓ**, ở mức aggregate — caption nói rõ "aggregated over... nominal + all stress tests" cho toàn bộ 6 method, không riêng ShortStop. Nhưng **không có bảng breakdown riêng theo từng stress condition cho baseline** (xem Table V).

### Table III — Component ablation (dòng 419–441)
- **Nội dung**: Reach-only → +STL-to-go → +CE-search → +Repair(=ShortStop), Violation/Success/Recovery.
- Số dòng cuối (`1.2 / 68.4 / 0.78`) **khớp đúng** dòng ShortStop trong Table II → xác nhận Table III dùng **cùng bộ dữ liệu aggregate** với Table II (không phải riêng Reach-Avoid-2D).
- Không có baseline nào ở đây — chỉ so các thành phần nội tại của ShortStop với nhau.

### Table IV — Per-environment breakdown (dòng 531–541)
- **Nội dung**: Unshielded vs ShortStop **only**, Violation%/Success%, chia theo từng environment (LIBERO/ManiSkill2/RLBench/Reach-Avoid-2D) thay vì gộp chung như Table II.
- **Không có baseline nào khác** (Conf-Thresh/MPC-Filter/CBF-Shield/STL-Monitor) trong bảng này.
- Số liệu:

| Environment | Viol. Unshld | Viol. ShortStop | Succ. Unshld | Succ. ShortStop |
|---|---|---|---|---|
| LIBERO~[56] | 16.4 | 1.0 | 78.1 | 72.9 |
| ManiSkill2~[57] | 21.9 | 1.6 | 69.5 | 63.2 |
| RLBench~[58] | 19.8 | 1.4 | 71.0 | 64.8 |
| Reach-Avoid-2D | 16.7 | 0.5 | 88.5 | 84.0 |

### Table V — Stress-test breakdown (dòng 498–511)
- **Nội dung**: Unshielded vs ShortStop **only** (+ Precision), chia theo condition: Nominal / Object-placement / Timing / Visual-shift. Chỉ có cột Violation%, **không có cột Success%**.
- **Baseline khác không xuất hiện ở đây** — đây là điểm trả lời trực tiếp câu hỏi "có stress test với baseline không": **có ở Table II (aggregate), không có breakdown riêng ở Table V**.
- Định nghĩa 3 stress test (đã có trong Report 1, xác nhận lại): Object-placement (obstacle/target lệch ±6cm, đặt đối nghịch trên nominal path), Timing (control-latency jitter + dropped observation), Visual-shift (lighting/texture/camera perturbation ở input của policy).
- Số liệu:

| Condition | Unshielded↓ | ShortStop↓ | Precision↑ |
|---|---|---|---|
| Nominal | 9.1 | 0.6 | 0.90 |
| Object-placement | 24.7 | 1.7 | 0.95 |
| Timing | 19.5 | 1.3 | 0.94 |
| Visual-shift | 21.3 | 1.2 | 0.95 |

### Table VI — Latency breakdown (dòng 515–525)
- **Nội dung**: median per-decision latency ($K{=}8,H{=}8,M{=}16$), chia theo stage của P-R-C-S:

| Stage | Latency (ms) |
|---|---|
| Reachtube propagation (Eq. 1) | 3.1 |
| STL robustness-to-go (Eq. 2) | 1.4 |
| Counterexample search (Eq. 3) | 3.0 |
| Repair, amortized (Eq. 4) | 1.4 |
| **Total** | **8.9** |

- **Cần gì**: chỉ cần profile chính pipeline ShortStop — không cần dataset/policy ngoài. Đo trên *"a single-core reference implementation"* (dòng 545–551, CPU đơn lõi, không batch GPU) trên *"a workstation-class machine"* (dòng 380–382).

### Table VII — Hyperparameters (dòng 743–753) — config chung cho mọi bảng trên
| Hyperparameter | Value |
|---|---|
| Candidate chunks $K$ | 8 |
| Certified horizon $H$ | 8 |
| CE-search iterations $M$ | 16 |
| Safety margin $\varepsilon$ | 2 cm |
| Repair trust region $\delta$ | 0.1 |
| Repair step size $\eta$ | 0.05 |
| Model-error quantile | 99th ($\times$1.25) |
| Disturbance bound $\bar w$ | 0.5 cm/step |
| Episodes $\times$ seeds | 500 $\times$ 5 |
| Bootstrap resamples | $10^4$ |

Kèm thêm (Appendix B, dòng 787–789 và đoạn tương ứng ở dòng 731–738): diffusion policy dùng **1D-convolutional U-Net denoiser**, **100 training diffusion steps / 10 inference DDIM steps**, action-chunk length **16** (shield chỉ certify $H{=}8$ đầu). Reachtube: **axis-aligned interval trên 6 tọa độ an toàn task-space** (end-effector + elbow position) cho các env robot; **zonotope bậc 2 (2nd-order)** chỉ dùng riêng cho `Reach-Avoid-2D` (ngược lại với giả định tôi từng nói trước đó trong hội thoại — cần đính chính: zonotope là của **2D study**, không phải của robot 3D). $\hat f$ (dynamics model): *"a locally linearized rigid-body model in sim and a learned residual one-step model for the visual-shift setting"* — nghĩa là $\hat f$ **đổi theo điều kiện stress test**, không phải 1 model cố định cho mọi trường hợp.

---

## 2. Trả lời trực tiếp 3 câu hỏi

### Dataset ở đâu?
**Không được paper nêu cụ thể** cho LIBERO/ManiSkill2/RLBench — vì (theo mục 0) paper không chạy simulator thật cho 3 benchmark này, chỉ hiệu chỉnh theo số đã công bố sẵn ở nơi khác. `Reach-Avoid-2D` là simulator riêng của nhóm tác giả, không phải dataset công khai (giống cách `Reach-Avoid-2D` của chúng ta là code tự viết, không phải dataset tải về).

### Policy được train từ đâu?
**Không nêu cụ thể theo từng environment.** Paper chỉ cho kiến trúc/hyperparameter *chung* (U-Net 1D-conv, 100 train/10 DDIM steps, chunk 16) — không nói checkpoint nào, train trên demo nào, cho từng benchmark. Nếu muốn tái lập **thật** (không phải projection), cần tự tìm nguồn cho từng env (đã research ở phần trước của hội thoại):
- **LIBERO**: ứng viên tốt nhất là checkpoint $\pi_0$ (Physical Intelligence, flow-matching) fine-tune sẵn trên LIBERO-Spatial/Object/Goal/10, host qua repo `openpi`.
- **ManiSkill2**: chỉ có script train + 4M+ frame demo data chính thức, **không có checkpoint pretrained sẵn** — phải tự train Diffusion Policy.
- **RLBench**: không có dataset cố định (RLBench tự sinh demo qua motion-planner theo yêu cầu) — checkpoint generative-policy thật gần nhất là "Mini Diffuser" (arXiv:2505.09430); PerAct/RVT có checkpoint nhưng là transformer 1-action, không phải generative chunk policy.

### Có stress test với baseline không?
**Có, nhưng chỉ ở mức aggregate (Table II)** — caption Table II xác nhận cả 4 baseline (Conf-Thresh/MPC-Filter/CBF-Shield/STL-Monitor) đều được gộp qua nominal + 3 stress test. **Không có bảng breakdown per-condition riêng cho baseline** — Table V (breakdown theo condition) chỉ show Unshielded vs ShortStop, không có 4 baseline kia.

---

## 3. Ý nghĩa cho việc setup thực tế (nối với phần "prepare tools" đã làm trước đó)

Nếu mục tiêu là **tái lập đúng số của paper** → không khả thi, vì phần lớn là projection, không phải simulator run.

Nếu mục tiêu là **tự tạo ra Table II–VI thật của riêng mình** (khuyến nghị hơn, và khớp hướng Stage 7a–7c đã note trong Report 2):
1. Cần Linux (xem `ShortStop_Report_2.tex`, section "Chuẩn bị hạ tầng") để chạy LIBERO/ManiSkill2/RLBench thật.
2. Cần policy per-env (xem mục "Policy được train từ đâu?" trên) — LIBERO có sẵn checkpoint tốt nhất để bắt đầu.
3. Cần tự implement 3 stress test (object-placement/timing/visual-shift) trên từng benchmark — paper không cho code, chỉ cho định nghĩa bằng lời.
4. Cần tự implement $\hat f$ theo đúng kiểu paper dùng (rigid-body linearized cho nominal, learned residual riêng cho visual-shift) — không phải 1 model dùng chung cho tất cả condition.
5. Table VI (latency) là bảng duy nhất **không cần** dataset/policy ngoài — chỉ cần profile lại đúng pipeline `shortstop/shield.py` hiện có trên máy thật, với $K,H,M$ giống Table VII.
