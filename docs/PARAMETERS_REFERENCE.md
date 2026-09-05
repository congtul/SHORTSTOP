# Tunable Parameters Reference — baselines & ShortStop

Mục tiêu file này: với mỗi tham số có thể tune trong repo — **nó là gì**, và **cần nhìn vào output/metric nào để biết nên tăng hay giảm**, cho đúng với env thực tế (CALVIN/arm) đang chạy. Giá trị trong paper (Table VII, xem `docs/REPRODUCING_TABLES_II_VI.md`) chỉ là 1 điểm khởi đầu người ta từng đề xuất cho setup của họ — **không phải chuẩn phải đạt tới**, không dùng để so sánh "đúng/sai". Mọi giá trị mặc định trong code hiện tại đều là điểm khởi đầu để tune tiếp, không phải kết luận.

7 metric chính (`shortstop/metrics.py::aggregate`, `shortstop/calvin_metrics.py`) dùng làm tín hiệu tune xuyên suốt file này: `violation_rate`, `success_rate`, `shield_activation_rate`, `intervention_precision`, `latency_ms_*`, `conservatism_cost`, `recovery_rate`.

---

## 1. Khái niệm & cách tune (đọc phần này trước)

Mỗi tham số dưới đây dùng chung ở nhiều class (2D lẫn arm/CALVIN) — giải thích 1 lần, mục 2 trở đi chỉ tra "class nào dùng, default hiện tại bao nhiêu".

### K — số candidate chunk mỗi bước (`n_candidates`)
- **Ý nghĩa**: Propose sinh ra K chunk, shield chọn 1 trong số đó (best-of-K).
- **Output cần theo dõi**: `shield_activation_rate` (bao nhiêu % bước không tìm được candidate nào an toàn → fallback/braking) và `success_rate`.
- **K quá nhỏ**: dễ rơi vào "0 candidate an toàn" → fallback nhiều → `success_rate` giảm dù không phải vì nguy hiểm mà vì không đủ lựa chọn. Dấu hiệu: `shield_activation_rate` cao nhưng `violation_rate` đã thấp sẵn.
- **K quá lớn**: tốn compute (mỗi candidate phải propagate reachtube riêng) → `latency_ms_mean` tăng tuyến tính theo K, trong khi `success_rate` sớm bão hoà (plateau).
- **Cách tune**: sweep K, vẽ `success_rate` (hoặc `shield_activation_rate`) theo K — dừng ở điểm đường cong bắt đầu phẳng, không cần K lớn hơn nữa.

### H — horizon chứng nhận (certified horizon / `horizon`)
- **Ý nghĩa**: Reachtube chỉ propagate và certify H bước đầu của chunk (chunk có thể dài hơn).
- **Output cần theo dõi**: `violation_rate` (H quá ngắn: bỏ sót nguy hiểm nằm sau bước H) và `conservatism_cost`/`shield_activation_rate` (H quá dài: box nhiễu cộng dồn qua nhiều bước → tube phình to → candidate hợp lệ bị từ chối oan).
- **Cách tune**: sweep H, tìm điểm `violation_rate` đã đủ thấp (không giảm thêm khi tăng H nữa) nhưng `conservatism_cost` chưa tăng mạnh.

### w_bar — bound nhiễu shield certify chống lại (`w_bar`, khác với nhiễu thật của env)
- **Ý nghĩa**: Reachtube giả định nhiễu mỗi bước nằm trong `[-w_bar, w_bar]`. Đây là giá trị shield *tin*, không nhất thiết bằng nhiễu *thật*.
- **Output cần theo dõi**: `violation_rate` (đo trên rollout thật) là tín hiệu chính.
- **w_bar bị đặt quá nhỏ so với nhiễu thật**: `violation_rate` > 0 dù shield "tưởng" đã certify an toàn — đây là dấu hiệu rõ ràng nhất cần tăng `w_bar`.
- **w_bar quá lớn**: `shield_activation_rate`/`conservatism_cost` tăng, `success_rate` giảm mà `violation_rate` không giảm thêm được nữa (đã ở mức sàn).
- **Cách tune đúng bài** (không dùng giá trị thật đặc quyền): `shortstop/calibration.py::calibrate_w_bar` — chạy rollout thật, đo residual `||x_{t+1} - f_hat(x_t, a_t)||`, lấy quantile cao (mặc định 0.99) nhân safety_factor (mặc định 1.25). Nếu sau khi certify theo giá trị này `violation_rate` vẫn > 0 đáng kể, tăng quantile hoặc safety_factor rồi đo lại; nếu `violation_rate` đã bằng 0 nhưng `success_rate` tệ hơn hẳn baseline, thử giảm.

### model_error — sai số mô hình động lực học ($\hat f$ vs $f$ thật)
- **Ý nghĩa**: Cộng thêm vào bán kính inflate của reachtube, bù cho việc $\hat f$ (mô hình shield dùng để propagate) không khớp 100% động lực thật.
- **2D prototype**: `model_error=0.0` là **chính xác**, không phải xấp xỉ — vì $\hat f$ ở đó bằng đúng $f$ thật (`reach.propagate` docstring). Không cần tune ở 2D.
- **Arm/CALVIN**: `model_error` khác 0 vì phép biến đổi task-space→joint-space qua Jacobian pseudo-inverse chỉ chính xác cục bộ (linearize quanh 1 điểm) — luôn có sai số thật, không tránh được.
- **Output cần theo dõi**: giống `w_bar` — `violation_rate` > 0 dù shield chấp nhận chunk ⇒ tăng `model_error`.
- **✅ Script đo đã có (2026-09-05)**: `scripts/calibrate_arm_model_error.py` — đo `shortstop.arm_reach.step_prediction_residual` (Cartesian, `||panda_frames(q_thật_sau_bước) - panda_frames(q_dự_đoán_qua_Jacobian)||`) trên rollout thật, không obstacle/shield, rồi tính `model_error = quantile(residuals, 0.99) * 1.25` — đúng recipe Table VII, chỉ khác plumbing so với `shortstop/calibration.py::calibrate_w_bar` (module đó hard-wire vào interface `ReachAvoid2D`, không tái dùng được cho arm). Kết quả chạy xong KHÔNG tự động wire vào đâu — phải tự set thủ công làm `model_error` cho `ArmReachOnlyShield`/`ArmSTLShield`/`ArmRepairShield` (không áp dụng cho `ArmSTLMonitorShield`, `model_error=0.0` cố định theo định nghĩa).

- **⚠️ Lần chạy đầu tiên (2026-09-05) cho ra `model_error = 0.928m` — KHÔNG dùng số này**, đây là dấu hiệu của 1 bug thật, không phải kết quả calibrate hợp lệ: `mean=0.32m`, `p99=0.74m` — phi lý cho 1 bước di chuyển thật (gần bằng cả tầm với tay máy). **Root cause đã tìm ra và SỬA XONG**: `arm_reach._step_joint_config` coi `task_space_delta_pos` (cột position của 1 hàng task_chunk) như đã là mét thật, nhưng CALVIN thật nhân action thô với `max_rel_pos=0.02` trước khi áp dụng (`mdt_policy/calvin_env/calvin_env/robot/robot.py::Robot.relative_to_absolute`, xác nhận `panda.yaml` không override) — mọi lời gọi `propagate_arm_tube`/`nominal_joint_trajectory` (tức TOÀN BỘ Reach step của mọi Arm shield) từng dự đoán cánh tay đi xa hơn thật ~50 lần (1/0.02). Đã fix: thêm hằng số `CALVIN_ACTION_SCALE=0.02` trong `arm_reach.py`, áp dụng trong `_step_joint_config` (fix nền tảng, xem module docstring). 2 chỗ khác tự tính Jacobian riêng (không qua `_step_joint_config`) cũng cần fix tương ứng: `ArmMPCFilterShield._solve_qp` và `ArmRepairShield._repair`'s `step_size`/`trust_region` (giờ coi 2 tham số này là mét thật, tự quy đổi sang raw units bên trong).
- **Cần chạy lại `calibrate_arm_model_error.py` sau fix này** để có số `model_error` thật, đáng tin cậy — số `0.928` ở trên chỉ để lưu lại làm bằng chứng phát hiện bug, không phải giá trị dùng được.
- **Hệ quả lan rộng cần biết** (đã xác nhận qua đọc lại code, không phải suy đoán): `calvin_experiment._clearance` (ground truth quyết định `violation_rate`/`success_rate` của MỌI baseline) đọc thẳng joint_angles THẬT từ obs, KHÔNG gọi `propagate_arm_tube` — **hoàn toàn miễn nhiễm bug này**, số liệu `radius=0.08` sweep không sai theo hướng này. Nhưng **`calvin_obstacle.sample_obstacle_from_reference_chunk` (đặt obstacle) và `calvin_experiment._candidate_clearance` (intervention_precision) đều dùng `propagate_arm_tube` — BỊ ảnh hưởng**: obstacle từng được đặt xa hơn thật ~50 lần theo hướng dự đoán — nên dù `radius=0.08` tự nó không sai, **vị trí đặt** đã thay đổi sau fix, khả năng cần sweep lại radius (không chỉ chạy lại). `ArmConfThreshShield._endpoint` cũng dùng `propagate_arm_tube` — **`CHOSEN_THRESHOLD=0.9` đã chốt trên 1 scale bị lệch ~50 lần, cần sweep lại từ đầu**, không chỉ chạy lại 1 lần ở threshold cũ.

### epsilon — biên an toàn STL ($\rho \geq \varepsilon$)
- **Ý nghĩa**: Không chỉ chặn "chạm biên obstacle" (robustness = 0) mà đòi hỏi cách biên ít nhất `epsilon`.
- **Output cần theo dõi**: `violation_rate` và `success_rate`/`conservatism_cost` cùng lúc — đây là 1 tham số đánh đổi kinh điển (Pareto).
- **epsilon quá nhỏ**: candidate được chấp nhận sát biên obstacle → mọi sai số hình học/xấp xỉ (Jacobian, box thay vì hình dạng thật, ...) dễ biến "vừa đủ an toàn theo shield" thành "va chạm thật" → `violation_rate` nhích lên.
- **epsilon quá lớn**: nhiều candidate hợp lệ bị từ chối dù thực ra an toàn → `shield_activation_rate` tăng, `success_rate`/`conservatism_cost` xấu đi.
- **Cách tune**: sweep epsilon, vẽ `violation_rate` vs `success_rate` — chọn điểm "khuỷu tay" (knee) của đường cong, hoặc điểm `violation_rate` vừa chạm mức chấp nhận được cho ứng dụng của bạn.

### trust_region ($\delta$) & step_size ($\eta$) — Repair (Eq. 4)
- **Ý nghĩa**: `step_size` là độ dài 1 bước gradient đẩy candidate ra khỏi obstacle; `trust_region` là biên tối đa cho phép candidate đã sửa lệch khỏi bản gốc.
- **Output cần theo dõi**: `recovery_rate` (repair có thành công certify lại không) song song với `success_rate`/`conservatism_cost` (candidate sửa xong còn "hữu ích" không, hay đã lệch quá xa mục tiêu ban đầu).
- **step_size/trust_region quá nhỏ**: 1 bước (với `max_repair_iters=1`) không đủ để thoát vùng vi phạm → `recovery_rate` thấp, nhiều candidate rơi vào fallback → `success_rate` giảm.
- **trust_region quá lớn**: candidate sau khi sửa có thể lệch rất xa ý định ban đầu (với arm: có thể phá vỡ giới hạn khớp thật, dù code hiện chưa check joint limit) — không có metric hiện tại bắt trực tiếp "lệch xa mục tiêu", phải tự quan sát qua log quỹ đạo hoặc thêm 1 signal đo khoảng cách tới candidate gốc.
- **Cách tune**: sweep `step_size` trước (giữ `trust_region` đủ rộng để không giới hạn), tìm điểm `recovery_rate` bắt đầu bão hoà; sau đó thu hẹp `trust_region` dần tới khi `recovery_rate` bắt đầu giảm rõ rệt — đó là biên hẹp nhất còn chấp nhận được.

### max_repair_iters — số vòng lặp CEGIS-style
- **Ý nghĩa**: `=1` đúng Algorithm 1 gốc (1 bước, không retry). `>1` là mở rộng ngoài paper (lặp counterexample-search→repair tới khi hết vi phạm hoặc hết số vòng).
- **Output cần theo dõi**: `recovery_rate` (tăng theo iters, nhưng bão hoà) so với `latency_ms_p95`/`latency_ms_mean` (KHÔNG dùng median — xem caveat trong `arm_shield.py`/`shield.py`: activation rate thấp có thể làm median "ẩn" mất nhánh chậm).
- **Cách tune**: tăng dần từ 1, dừng khi `recovery_rate` không cải thiện thêm đáng kể so với chi phí latency tăng thêm.

### disagreement_threshold (Conf-Thresh baseline)
- **Ý nghĩa**: Ngưỡng "độ lệch giữa các candidate" để coi là không đáng tin.
- **Output cần theo dõi**: `intervention_precision` — đo được: sweep 0.5→0.06 trên `GaussianChunkPolicy` cho `violation_rate` phẳng ~0.81 ở MỌI ngưỡng, tức proxy này không mang tín hiệu gì cho policy tổng hợp hiện tại. **Trước khi tốn công sweep tiếp trên CALVIN/policy thật**, kiểm tra `intervention_precision` có nhích lên theo threshold không — nếu vẫn phẳng, đừng đầu tư thêm vào baseline này, ghi nhận đó là kết quả thật (giống paper cũng chỉ báo 0.43 precision — yếu, không phải 0).
- **⚠️ Cập nhật 2026-09-05 — thêm 1 lý do nữa cần sweep lại (ngoài g(a)/A.4 đã biết)**: `ArmConfThreshShield._endpoint()` dùng `propagate_arm_tube` — bị bug scale action (xem mục "model_error" ở trên), nghĩa là toàn bộ sweep dưới đây đo disagreement trên 1 scale bị lệch ~50 lần so với thật. Cần sweep lại từ đầu, không chỉ chạy lại đúng threshold cũ.
- **✅ Đã chốt trên CALVIN thật (trước fix, cần sweep lại) (`scripts/run_calvin_shielded.py`)**: `CHOSEN_THRESHOLD = 0.9` — sweep thật trên tuning cohort (`[0.15, 0.35, 0.6, 0.9]`) cho `violation_rate` phẳng (~0.13-0.142) bất kể `activation_rate` (0.086-0.990), xác nhận đúng phát hiện của paper ("disagreement is a poor safety proxy"); 0.9 là Pareto-best (activation thấp nhất mà vẫn có hoạt động thật, không phải no-op). **Xác nhận sạch trên eval cohort (idx 100-199, 2026-09-05)**: `violation_rate=0.140`, `success_rate=0.466`, `shield_activation_rate=0.074`, `avg_seq_len=2.33/5` (n=307 subtask, clearance mean=0.204 median=0.184 p10=-0.057 p90=0.515 min=-0.204 max=0.638) — gần như trùng số liệu tuning cohort (violation 0.134→0.140, success 0.474→0.466), cùng pattern lệch nhỏ đã thấy ở unshielded. **Đây là số liệu final cần trích dẫn cho Conf-Thresh**, không phải số liệu tuning cohort. So với unshielded-with-obstacle (violation=0.136, success=0.476, xem mục "radius" ở trên) — Conf-Thresh **không cải thiện** violation_rate (0.140 vs 0.136, còn TỆ hơn chút) trong khi success_rate cũng thấp hơn (0.466 vs 0.476) — khớp với kết luận "disagreement không phải proxy an toàn tốt" đã ghi nhận ở trên, không phải bug.

### alpha (CBF-Shield gain)
- **Ý nghĩa**: Class-$\mathcal{K}$ gain trong điều kiện đạo hàm CBF — alpha lớn cho phép giá trị barrier giảm nhanh hơn trước khi can thiệp (can thiệp trễ/gần biên hơn); alpha nhỏ can thiệp sớm/xa biên hơn.
- **Output cần theo dõi**: `violation_rate` (alpha quá lớn: can thiệp trễ so với tốc độ hệ rời rạc hoá thật → dễ vọt qua biên giữa 2 bước) vs `success_rate`/`conservatism_cost` (alpha quá nhỏ: can thiệp quá sớm/thường xuyên).
- **Cách tune**: sweep alpha, vẽ `violation_rate` vs `success_rate`, chọn điểm cân bằng.

### max_action_norm — giới hạn biên độ hành động
- Đây **không phải tham số nên tune theo metric** — phải khớp giới hạn vật lý thật của actuator/robot (2D: giả định abstract; arm thật: giới hạn tốc độ khớp/EE thật). Đặt sai (nhỏ hơn khả năng thật) sẽ làm mọi shield trông "tệ hơn" một cách giả tạo vì tự giới hạn hành động không liên quan gì đến an toàn.

### MDT sampler config (`sampler_type`, `num_sampling_steps`, `sigma_min/max`, `noise_scheduler`)
- **Ý nghĩa**: Chất lượng vs tốc độ của bước denoise diffusion — không phải tham số của ShortStop, mà của chính policy MDT (ảnh hưởng cả unshielded lẫn shielded).
- **Output cần theo dõi**: chạy **không obstacle** trước, xem `success_rate` (chất lượng chunk thô, độc lập với shield) và latency của riêng bước `forward()`.
- **num_sampling_steps quá nhỏ**: chunk kém chất lượng hơn → có thể vừa giảm `success_rate` vừa tăng `violation_rate` cùng lúc (không phải đánh đổi, mà xấu cả 2 phía) — đây là dấu hiệu rõ ràng cần tăng, không phải tune như 1 tradeoff thật.
- **Cách tune**: sweep `num_sampling_steps` khi CHƯA có obstacle, tìm điểm `success_rate` bão hoà (không cần bước cao hơn nữa) — số này lấy từ chính checkpoint MDT, không liên quan gì đến ShortStop.

### multistep / replan_steps — số bước thực thi trước khi propose lại
- **Ý nghĩa**: Chunk dài hơn `replan_steps` chỉ thực thi `replan_steps` bước đầu rồi propose lại (nhưng reachtube certify cả H bước — xem tương tác với H ở trên).
- **Output cần theo dõi**: `violation_rate` (replan thưa: nguy hiểm phát sinh giữa chunk không được phát hiện tới lần propose sau) vs `latency_ms_mean`×tần suất gọi (replan dày: gọi policy/shield nhiều hơn, tốn hơn).
- **Cách tune**: sweep cùng lúc với H (2 tham số liên quan chặt) — `replan_steps <= H` là ràng buộc hợp lý (không thực thi vượt quá phần đã certify).

**Protocol thật của paper là `replan_steps = 1` cho MỌI baseline, không riêng ShortStop** — grounded trực tiếp từ paper text (`docs/main (3).txt`): Alg. 1 line 17 `return first action of a?`; Thm. 1's proof: *"Only the first action of a\* is committed before re-deciding (receding-horizon execution), so certification is refreshed every step even though chunks span H"*; MPC-Filter baseline's mô tả cũng nói *"minimally corrects the first action"*. `H` chỉ là độ sâu **certify** (nhìn xa để chứng minh an toàn), KHÔNG phải số bước được **thực thi** trước khi quyết định lại — 2 khái niệm paper tách bạch rõ.

Pipeline 2D của repo này (`shortstop/experiment.py::run_episode`) đã luôn làm đúng `replan_steps=1` (hardcode trong cấu trúc loop, không tham số hoá — mỗi lần lặp = 1 lần `policy.propose()` mới + `env.step(action_chunk[0])`). CALVIN harness (`replan_steps=10`, kế thừa `cfg.multistep` gốc của MDT vì lý do compute) là **deviation có chủ đích** khỏi protocol thật này, chưa từng được ghi nhận rõ trước phiên làm việc này.

**Ý tưởng "tách tần suất filter khỏi tần suất policy"** (để giảm chi phí compute mà vẫn gần với `replan_steps=1` thật của paper): Propose (đắt, cần K lần diffusion sample) chạy thưa (mỗi `replan_steps` bước như hiện tại), nhưng Certify/Filter (nếu rẻ — không cần sample lại) chạy MỖI BƯỚC, re-verify phần đuôi còn lại của chunk đã chọn từ state THẬT (không phải state dự đoán lúc propose) — bắt drift/model-error sớm hơn, gần với recursive-feasibility của Thm 1 hơn `replan_steps=10` "mù" hiện tại. Khả thi hay không tuỳ cơ chế mỗi baseline:

| Baseline | Tách filter freq > policy freq được không? | Vì sao |
|---|---|---|
| Conf-Thresh | **Không** | Disagreement gắn chặt với lần sample K candidate cụ thể đó — không có gì rẻ để re-check giữa 2 lần propose, không có "phần đuôi" nào mang tính certificate để re-verify |
| MPC-Filter / CBF-Shield | **Được** | QP correction chỉ cần state hiện tại + $\hat f$, không cần sample K candidate mới |
| STL-Monitor | **Được** | Robustness-to-go là rollout trên $\hat f$, không cần sample mới |
| ShortStop (Repair) | **Được, tốt nhất** | Reach/Certify/Repair vốn thiết kế để rẻ (không cần diffusion sample), tái dùng phần đuôi candidate đã chọn — đúng tinh thần recursive-feasibility của Cor. 1 |

**Tỉ lệ `replan_steps`/`H` tham khảo từ literature action-chunking** (không phải paper ShortStop, nhưng liên quan trực tiếp vì cùng bài toán "dự đoán H bước, thực thi bao nhiêu trước khi replan"): **Diffusion Policy (Chi et al., RSS 2023)** — paper action-chunking được cite nhiều nhất — dùng `Tp=16` (prediction horizon), `Ta=8` (execution horizon) → **tỉ lệ 0.5**, tune thực nghiệm ("action horizon 8 tối ưu cho hầu hết task"), không phải suy ra từ 1 theorem an toàn nào. Có 2 truyền thống khác nhau, không nên nhầm lẫn: MPC cổ điển/Thm 1 của ShortStop paper cần `replan_steps=1` (lý thuyết yêu cầu, để giữ recursive feasibility), còn action-chunking literature thực dụng (Diffusion Policy, RTC) dùng tỉ lệ ~0.5 (đánh đổi compute/reactivity, không gắn với an toàn). Setup hiện tại của CALVIN (`replan_steps=10=H`, tỉ lệ **1.0**) còn "mở" hơn cả mức thực dụng 0.5 — chưa từng thấy paper nào chọn tỉ lệ này có chủ đích.

**Đã thử đổi `replan_steps` 10→5 (tỉ lệ 0.5, khớp Diffusion Policy's Ta/Tp) ngày 2026-09-04, ĐÃ REVERT lại 10 CÙNG NGÀY — phát hiện thật quan trọng hơn cả câu hỏi lý thuyết ban đầu:**

Chạy sweep thật (n_sequences=100, tuning cohort) ở `REPLAN_STEPS=5` cho baseline **"without obstacle"** (không obstacle nào cả, độc lập với radius): `success_rate` tụt từ **0.930** (ở `replan_steps=10`) xuống **0.834** (ở `replan_steps=5`) — mất ~10 điểm % chỉ từ việc replan nhanh hơn, chưa tính obstacle. Lý do: mỗi lần `propose()` vẽ noise độc lập hoàn toàn mới (không "nối tiếp" chunk trước) — replan thường xuyên hơn = nhiều "seam"/điểm giật giữa các chunk độc lập hơn = dễ trật task hơn. Đây đúng là lý do Diffusion Policy's own ablation KHÔNG chọn `Ta` nhỏ nhất có thể, dù có vẻ "phản ứng nhanh hơn" trên giấy.

**Kết luận**: tỉ lệ 0.5 của Diffusion Policy được tune cho checkpoint/setup CỦA HỌ, không transfer sang checkpoint CALVIN này. Và không có gì trong project hiện tại thật sự cần replan nhanh hơn để được lợi (Conf-Thresh không tách được filter/policy freq dù gì — xem bảng trên; baseline nào tách được thì còn chưa tồn tại) đủ để bù lại cái giá ~10pp này. **Quay lại `REPLAN_STEPS=10` (tỉ lệ 1.0)** — số liệu thật (trên đúng checkpoint đang dùng) thắng heuristic mượn từ paper khác. `radius=0.08` (tune ở `replan_steps=10` từ trước) **valid lại, không cần sweep lại** — `RADII_TO_SWEEP` trong `run_calvin_unshielded.py` đã trim về lại `[0.08]`.

**Vấn đề "công bằng" giữa các baseline đã resolve**: `freq_filter = freq_policy` của Conf-Thresh **không phải bất công cần sửa** — đó là hạn chế cố hữu của chính cơ chế Conf-Thresh (disagreement gắn chặt với lần sample K candidate cụ thể, không có gì rẻ để re-check giữa 2 lần propose — xem bảng trên), giống như paper's Table II tự nhiên có latency khác nhau giữa các baseline (Conf-Thresh 1.6ms vs MPC-Filter 12.4ms vs ShortStop 8.9ms) vì mỗi cơ chế vốn khác nhau, không phải setup thực nghiệm thiên vị. Baseline nào sau này tách được filter/policy freq (MPC-Filter/CBF-Shield/STL-Monitor/ShortStop) thì tận dụng — không cần kéo Conf-Thresh xuống cùng mức để "công bằng giả tạo".

**Đã implement (2026-09-05), bắt đầu từ STL-Monitor**: `ArmReachOnlyShield.recertify(joint_angles, remaining_chunk)` (`shortstop/arm_shield.py`) — tái dùng đúng `_admissible()` mà `select()` đã dùng, chỉ khác là gọi trên phần đuôi còn lại + state THẬT. Kế thừa tự động xuống `ArmSTLShield`/`ArmSTLMonitorShield`/`ArmRepairShield` — không cần sửa gì thêm ở các class con. `run_calvin_shielded_subtask` (`shortstop/calvin_experiment.py`) giờ gọi `shield.recertify(...)` sau MỖI hàng đã thực thi thật (không chỉ ở ranh giới `replan_steps`) — nếu fail, bỏ ngay phần đuôi chunk hiện tại và propose lại sớm thay vì đợi đủ `replan_steps` bước. `hasattr(shield, "recertify")` là no-op cho `ArmConfThreshShield` (không có method này, đúng bảng trên) — hành vi của nó không đổi. Test: `tests/test_arm_shield.py::test_recertify_matches_admissible_and_reacts_to_a_drifted_real_state`, `tests/test_calvin_experiment.py::test_shielded_subtask_recertifies_every_step_and_reproposes_early_on_failure`.

**Nâng cấp tiếp, cùng ngày, riêng cho MPC-Filter**: `recertify` (binary recheck) không đủ đúng bản chất "predictive safety filter" — literature gốc (Wabersich & Zeilinger) là **re-solve QP mỗi bước điều khiển thật**, không chỉ verify lại nghiệm cũ. `ArmMPCFilterShield` (`shortstop/arm_shield.py`) định nghĩa riêng `resolve(joint_angles, remaining_chunk) -> remaining_chunk mới hoặc None` — re-solve ĐÚNG QP đó (tái dùng `_solve_qp`, tách ra từ `select()`) nhưng linearize lại quanh state THẬT + coi `remaining_chunk` (đuôi đã sửa lần trước, không phải nominal gốc) làm reference. `run_calvin_shielded_subtask` ưu tiên `resolve` hơn `recertify` khi shield có cả 2: nếu có `resolve`, kết quả của nó được **swap thẳng vào `chunk`** (`chunk[row_idx+1:window] = resolved`, dựa vào việc slice numpy là view chia sẻ bộ nhớ — vòng `for` đang chạy sẽ đọc đúng giá trị mới, đã verify bằng code thật không chỉ suy luận), không chỉ dùng làm cổng đi/không đi như `recertify`. `None` xử lý giống `recertify` trả `False` (bỏ chunk, propose lại ngay). Test: `tests/test_arm_shield.py`'s 3 test `resolve` mới + `tests/test_calvin_experiment.py::test_shielded_subtask_executes_resolves_result_not_the_stale_chunk`/`test_shielded_subtask_abandons_the_chunk_when_resolve_returns_none`.

**Lưu ý đã đo được**: chain `select()` rồi `resolve()` (đúng như harness thật làm) là 2 lần linearize riêng biệt chồng lên nhau — verify bằng số thật cho 1 case cụ thể: robustness sau khi chain chỉ còn **-0.00068m** (âm nhẹ, dưới 1mm) thay vì ≥0 tuyệt đối — sai số cộng dồn nhỏ, THẬT nhưng không đáng kể, đã ghi nhận rõ trong docstring `ArmMPCFilterShield`, không che giấu.

**Nâng cấp tiếp, cùng ngày, cho ShortStop (`ArmRepairShield`)**: cùng lý do như MPC-Filter — shield này đã CÓ SẴN 1 cơ chế sửa thật (`_repair()`, gradient-step theo Eq. 4), nên chỉ dùng `recertify` (nhị phân, kế thừa) ở bước drift-check giữa chừng là lãng phí đúng khả năng làm nó khác STL-Monitor. `ArmRepairShield.resolve(joint_angles, remaining_chunk)` giờ tái dùng đúng logic per-candidate của `select()` (đã admissible thì giữ nguyên; không thì `arm_find_counterexample` + `_repair()`), áp cho state thật thay vì 1 trong K candidate nominal. Trả `None` nếu repair thất bại (giống `recertify` trả `False`). Test: `tests/test_arm_shield.py`'s 3 test `arm_repair_shield_resolve_*` mới (no-op, sửa thành công verify bằng FK thật, thất bại đúng tham số với `test_arm_repair_shield_falls_back_when_repair_cannot_fix_it_in_time`).

CBF-Shield chưa có bản cho tay máy (`ArmCBFShield`) — nếu build sau này, cùng bản chất QP 1-bước như MPC-Filter, áp dụng lại đúng pattern `resolve` này, không cần thiết kế lại từ đầu.

Chưa làm: `propose()` vẫn chạy đúng nhịp `replan_steps` cố định như trước (chưa rút ngắn nhịp Propose theo tỉ lệ Diffusion Policy nào cả) — thay đổi lần này CHỈ ở tần suất Certify/Filter, đúng phạm vi ý tưởng ở trên, không đụng tới câu hỏi `replan_steps` (nhịp Propose) đã chốt =10 ở mục trên.

**Ngoại lệ mới, 2026-09-05 (TOTAL fallback, không phải decoupling ở trên)**: khi `select()` trả về `fallback=True` (KHÔNG candidate nào admissible, khác với chỉ 1 phần bị reject), `run_calvin_shielded_subtask` giờ chỉ đứng yên **1 bước thật** rồi `propose()` lại ngay, thay vì đứng yên trọn `replan_steps`. Lý do: `propose()` luôn vẽ K sample độc lập mới (không seed lại — xem `mdt_policy_client.py`), nên đứng yên cả `replan_steps` bước trước khi thử lại chỉ lãng phí ngân sách `ep_len` trên 1 state gần như không đổi (phát hiện thật từ 1 sweep trước fix A/B: ~90% episode kết thúc do hết `ep_len`, không phải violate/success — nghi ngờ đúng là 1 dạng "kẹt vòng lặp fallback"). Quyết định (partial rejection, còn ít nhất 1 candidate qua) KHÔNG đổi — vẫn chạy trọn `replan_steps` như cũ. Trường hợp fallback lặp lại liên tục sẽ làm tăng mạnh số lần gọi `propose()` (đắt — chạy diffusion sample) trong cùng `ep_len` — đánh đổi có chủ đích (nhiều cơ hội thoát kẹt hơn, nhưng tốn compute hơn khi thật sự kẹt). Metric mới `n_fallback` (đếm riêng số quyết định `n_admissible==0`, khác `n_activated` vốn đếm cả partial rejection) thêm vào kết quả trả về để đo trực tiếp tần suất path này và hiệu quả của fix, thay vì chỉ suy luận từ `steps_taken`. Test: `tests/test_calvin_experiment.py::test_shielded_subtask_fallback_holds_the_gripper_instead_of_forcing_it_closed`.

### replan_steps trong g(a)/disagreement (Conf-Thresh) — đã fix
`ArmConfThreshShield._endpoint()` và `calvin_progress.calvin_progress_scores()` giờ đo predicted endpoint tại `chunk[:replan_steps]`, KHÔNG phải toàn bộ chunk (`H`/`act_window_size`) — cả 2 đều nhận `replan_steps` làm tham số **bắt buộc** (không default), để không âm thầm rơi lại về "đo cả chunk". Trước khi fix, code đo `tube[-1]` của TOÀN BỘ chunk — chỉ "đúng một cách trùng hợp" vì `act_window_size == replan_steps == 10` theo config default hiện tại, không phải vì code có chủ đích liên kết 2 giá trị này.

### epsilon (STL-Monitor baseline trên CALVIN, `ArmSTLMonitorShield`) — ĐANG TUNE, chưa chốt
Khác với `shortstop/baselines.py::STLMonitorShield` (bản 2D, ngưỡng cố định 0 theo đúng nghĩa đen "rejects if negative"), bản CALVIN/arm (`shortstop/arm_shield.py::ArmSTLMonitorShield`) coi `epsilon` là tham số **cần tune thật, không hardcode** — lý do: chính văn bản paper (`docs/main (3).txt` Sec. IV) mâu thuẫn nhau ở đúng chỗ này:
- *"STL-Monitor, which computes nominal STL robustness on the fˆ rollout and rejects if negative"* → đọc theo nghĩa đen: `epsilon=0.0`.
- Ngay câu kế tiếp: *"All model-based baselines use the identical fˆ, ε and fallback for a fair comparison"* → đọc theo nghĩa "epsilon dùng chung": `epsilon` = margin đã calibrate của chính ShortStop (paper Sec. VI: `margin ε = 2cm`, tức 0.02 trong codebase này).

Không thể resolve chỉ từ text (nhiều khả năng do PDF 2 cột bị xáo khi convert sang `.txt`, như từng gặp ở chỗ khác trong file này) — quyết định: **tune thực nghiệm để tự trả lời**, đúng cách `disagreement_threshold` của Conf-Thresh từng được resolve (xem mục trên), thay vì đoán 1 trong 2 cách đọc.
- **Output cần theo dõi**: giống mọi tham số STL/epsilon khác — `violation_rate` vs `success_rate`/`shield_activation_rate` (mục "epsilon — biên an toàn STL" ở trên).
- **Cách tune**: `python scripts/run_calvin_stl_monitor.py --tuning` — sweep `EPSILONS_TO_SWEEP` (hiện là placeholder `[0.0, 0.02, 0.05, 0.1]`, neo tại đúng 2 cách đọc trên + 2 điểm rộng hơn để thấy xu hướng), Phase 1 (`epsilon=-inf`, mọi candidate admissible) log percentile thật của nominal robustness để tinh chỉnh lại list này nếu cần — cùng khuôn với `THRESHOLDS_TO_SWEEP` của Conf-Thresh (`scripts/run_calvin_shielded.py`). Chốt `CHOSEN_EPSILON` trên tuning cohort (idx 0-99), xác nhận 1 lần trên eval cohort (idx 100-199) — xem [[feedback_tuning_cohort_split]].
- **Chưa có số liệu thật** — `RUN_DIAGNOSTIC=True`/`CHOSEN_EPSILON=0.0` trong script hiện là placeholder, chờ lần chạy WSL2 đầu tiên.

### radius — obstacle ảo $X_u$ trên CALVIN (`shortstop/calvin_obstacle.py`)
- **Ý nghĩa**: Tham số riêng của CALVIN (không có trong paper) — độ "khó" của bài toán an toàn tổng hợp mình tự tạo ra.
- **Output cần theo dõi**: `violation_rate` của baseline **unshielded** (chưa có shield) khi bật obstacle.
- **radius quá nhỏ**: obstacle gần như không bao giờ bị chạm → `violation_rate` (unshielded, with obstacle) gần 0 → không có gì để shield chứng minh cải thiện (floor effect, vô nghĩa).
- **radius quá lớn**: gần như chunk nào cũng chạm → `violation_rate` gần 100% → cũng vô nghĩa (ceiling effect), và không còn phản ánh "obstacle đặt trên đường đi thật" nữa mà đơn giản là chặn hết.
- **Cách tune**: sweep radius, mục tiêu tìm khoảng `violation_rate` (unshielded) vừa đủ cao để có "chỗ" cho shield cải thiện rõ rệt (ví dụ 30–70%, không có con số chuẩn — tự quyết theo câu chuyện muốn kể trong paper) nhưng không sát 0% hay 100%.
- **Đã implement**: `scripts/run_calvin_unshielded.py::RADII_TO_SWEEP` (mặc định `[0.02, 0.05, 0.08, 0.12]`, sửa tay list này khi cần thử giá trị khác) chạy baseline unshielded với từng radius, in `violation_rate`/`success_rate` mỗi giá trị. Khi `cfg.debug=True` (mặc định hiện tại), in thêm thống kê `min_clearance` (mean/median/p10/p90/min/max) trên toàn bộ subtask đã thử — tín hiệu tinh hơn nhị phân violated/not-violated: nếu p10 luôn dương xa 0 dù radius đã lớn -> floor effect thật sự, còn nếu median rất âm -> ceiling effect.
- **`min_clearance`/violation check đã tính đúng vật lý**: `_clearance()` (`calvin_experiment.py`) và `propagate_arm_tube()` (`arm_reach.py`) giờ trừ thêm bán kính vật lý của chính sphere (`SPHERE_RADII`), không chỉ `obstacle.radius` — trước đó tay bị coi như 1 điểm, thiếu hẳn kích thước thật, khiến "radius" đang sweep nhỏ hơn "vùng cản thật" tương đương. Bug này được phát hiện từ chính câu hỏi "radius to quá thì khó tránh" của user. `SPHERE_RADII` bản thân cũng đã được đo lại từ mesh thật (`[0.09, 0.10, 0.06, 0.05]`, xem mục 8) thay vì placeholder `[0.08, 0.08, 0.08, 0.06]` cũ.
- **Caveat hình học liên quan** (xem `arm_reach.py`'s docstring): việc inflate box bằng `SPHERE_RADIUS` dùng cùng cơ chế `Box.inflate()` (khối vuông, không phải khối cầu thật) như `w_bar`/`model_error` — nếu độ dịch chuyển mỗi bước (`replan_steps`/chunk) **nhỏ hơn** tổng lượng inflate (sphere_radius + w_bar + model_error), 2 bước liên tiếp trong tube có thể không phân biệt được (box "nuốt" điểm obstacle của bước trước). Không phải lỗi bỏ sót va chạm (vẫn luôn over-approximate, không under-approximate), nhưng `step`/`sphere` báo cáo trong counterexample có thể sai lệch (chỉ vào bước không thực sự là bước gây nguy hiểm nhất) khi motion mỗi bước quá nhỏ so với radius tổng.
- **4 sphere-điểm không đủ để check va chạm dọc theo link** (câu hỏi user đặt ra trực tiếp): elbow/forearm/wrist chỉ là 1 ĐIỂM + bán kính cắt ngang tại 1 đầu của link, nhưng link đó thật ra dài 0.14–0.35m (đo từ mesh) — obstacle nằm giữa link (không trúng đúng điểm sphere) sẽ bị bỏ sót. Thêm nữa, link0/1/2/4/6 không có sphere nào đại diện cả. **Đã sửa cho ground-truth check** (`calvin_experiment._clearance`): giờ dùng `robot_geometry.capsule_segments()` — 8 capsule (đoạn thẳng + bán kính) nối đủ 9 frame (0..8), phủ toàn bộ chiều dài tay. **CHƯA sửa cho `arm_reach.py`'s reachtube** (dùng bởi Certify step của shield thật) — vẫn chỉ check 4 điểm cũ. Đây là mismatch cần xử lý trước khi wiring shield thật vào CALVIN, đã flag rõ trong `arm_reach.py`'s docstring.
- **Ngón tay (vượt ra ngoài flange) cũng bị bỏ sót ban đầu** (user hỏi tiếp: "objs đặt ngẫu nhiên không rơi vào phạm vi ngón tay à?"): obstacle được đặt tại vị trí TƯƠNG LAI của gripper (endpoint reach-tube), tay di chuyển dần tới đó — nhưng ngón tay thò ra PHÍA TRƯỚC flange theo đúng hướng di chuyển, nên chạm sớm hơn lúc flange (sphere gripper cũ, bán kính 0.05m) tới đủ gần để bị phát hiện → **under-count thật, không phải rủi ro thấp** (sửa lại nhận định ban đầu). Đã sửa: thêm `robot_geometry.gripper_tip_position()` (điểm TCP, cách flange đúng `GRIPPER_TIP_OFFSET=0.1m` dọc theo hướng flange đang chỉ — suy ra từ `panda.urdf`'s `tcp_joint`, không cần biết góc xoay trung gian vì tịnh tiến dọc trục z không bị ảnh hưởng bởi xoay quanh chính trục đó) + capsule flange→TCP, bán kính `GRIPPER_TIP_RADIUS=0.06` (0.04m độ mở ngón tay tối đa mỗi bên + 0.02m độ dày ngón tay, cận trên cố ý bảo toàn cho cả trường hợp gripper mở hết — không mô phỏng động trạng thái đóng/mở gripper).

**⚠️ Cập nhật 2026-09-05 — cần sweep lại**: bảng dưới đây đo TRƯỚC khi phát hiện+sửa bug scale action (mục "model_error" ở trên) — `sample_obstacle_from_reference_chunk` (đặt obstacle) dùng `propagate_arm_tube`, từng đặt obstacle xa hơn thật ~50 lần theo hướng dự đoán. Radius=0.08 tự nó (bán kính obstacle) không sai, nhưng VỊ TRÍ đặt đã đổi sau fix — bảng dưới **không còn đáng tin cậy để chốt lại radius=0.08 mà không sweep lại**.

**✅ ĐÃ CHỐT (trước fix, cần re-sweep): radius = 0.08m** — kết quả sweep thật, chạy trên checkpoint/dataset thật (n_sequences=100 mỗi radius, sau khi capsule-chain fix ở trên đã áp dụng):

| radius | violation_rate | success_rate | avg_seq_len | clearance p10 | clearance min | min + radius |
|---|---|---|---|---|---|---|
| 0 (baseline, không obstacle) | — | 0.930 | 4.65/5 | — | — | — |
| 0.02 | 0.076 | 0.704 | 3.52/5 | +0.0042 | -0.1289 | -0.1089 |
| 0.05 | 0.108 | 0.586 | 2.93/5 | -0.0084 | -0.1589 | -0.1089 |
| **0.08 (chốt)** | **0.128** | **0.482** | **2.41/5** | **-0.0286** | **-0.1889** | **-0.1089** |
| 0.12 | 0.148 | 0.412 | 2.06/5 | -0.0732 | -0.2289 | -0.1089 |

Nhận xét khi tune: (1) `min_clearance + radius` ra **đúng hằng số -0.1089 ở mọi radius** — vì rollout dừng ngay khi violated, nên "overshoot" tối đa chỉ phụ thuộc chuyển động 1 bước của policy, không phụ thuộc radius; xác nhận `_clearance()`/dừng-khi-violated hoạt động đúng thiết kế, không phải bug. (2) Không có floor effect ở 0.02 (violation_rate 7.6% đã là tín hiệu thật, không gần 0%) và chưa chạm ceiling ở 0.12 (14.8%, còn xa 100%) — toàn bộ range 0.02–0.12 đều "dùng được", 0.08 được chọn vì cân bằng: violation_rate đủ cao để shield có việc làm, success_rate còn 48.2% (chưa về 0, còn khoảng để shield show cải thiện). (3) success_rate giảm rất nhanh dù radius nhỏ (93%→70.4% chỉ với r=0.02) — do cấu trúc chained-sequence (1 subtask violated làm mất hết điểm các subtask sau trong `build_fixed_cohort_slots`), không phải bug, nhưng cần nhớ khi so sánh giữa CALVIN và LIBERO nếu LIBERO không chain task như vậy.

Muốn sweep lại (checkpoint mới, hoặc muốn xem ceiling effect ở radius > 0.12): sửa `RADII_TO_SWEEP` trong `scripts/run_calvin_unshielded.py`.

**✅ Xác nhận sạch trên eval cohort (idx 100-199, 2026-09-05)** — bảng trên đo trên tuning cohort (idx 0-99), có bias tối ưu nhẹ theo `feedback_tuning_cohort_split`. Chạy lại đúng 1 lần trên eval cohort (`python scripts/run_calvin_unshielded.py`, không cờ, `REPLAN_STEPS=10`, `radius=0.08`):

| | violation_rate | success_rate | avg_seq_len |
|---|---|---|---|
| without obstacle | — | **0.926** | 4.63/5 |
| with obstacle r=0.08 | **0.136** | **0.476** | 2.38/5 |

Gần như y hệt bảng tuning (0.128→0.136, 0.482→0.476) — xác nhận bias tuning-on-eval trước đó nhỏ, đúng như dự đoán. **Dùng số eval này (violation=0.136, success=0.476) làm baseline unshielded cuối cùng để báo cáo**, không phải số ở bảng tuning phía trên.

### num_sequences / n_episodes — cỡ mẫu
- **Không phải tham số đánh đổi chất lượng** — chỉ ảnh hưởng độ tin cậy thống kê của các metric đo được.
- **Output cần theo dõi**: độ rộng của confidence interval (paper's own recipe: bootstrap 10^4 resamples) hoặc đơn giản là chạy lại với seed khác, xem các metric có dao động nhiều không.
- **Cách tune**: tăng tới khi metric ổn định giữa các lần chạy lặp lại (không đổi seed vẫn ra số gần giống nhau).

### Sphere radii / DH table (`shortstop/robot_geometry.py`) — KHÔNG tune theo metric
- Khác hẳn mọi tham số trên: đây là **hình học vật lý**, phải khớp robot thật (đo từ mesh CAD/URDF), không phải thứ "tune cho ra metric đẹp". Nếu tune sai theo hướng "đặt sphere radii nhỏ lại để success_rate cao hơn", số liệu sẽ đẹp nhưng vô nghĩa (an toàn giả). Chỉ sửa khi có số đo thật từ `franka_description`/mesh CAD.

---

## 2. Tra cứu nhanh — 2D env/episode-level (`shortstop/env.py`, `shortstop/experiment.py`)

| Param | Default | Khái niệm ở mục 1 |
|---|---|---|
| `dt` | 0.1 | — (bước rời rạc hoá, không phải tham số tune theo metric, chỉ cần đủ nhỏ so với động lực thật) |
| `w_bar` (thật, sinh nhiễu env) | 0.02 | privileged ground-truth, không phải cái để tune |
| `shield_w_bar` (certify) | `None` → dùng `w_bar` | **w_bar** ở mục 1 |
| `horizon` | 8 | **H** |
| `n_candidates` | 8 | **K** |
| `max_steps` | 200 | giới hạn cứng, không tune |
| `goal_radius` | 0.3 | định nghĩa "reached", không tune cho shield |
| `max_action_norm` | 1.0 | **max_action_norm** |

## 3. Tra cứu nhanh — ShortStop shield stages 2D (`shortstop/shield.py`)

| Class | Param | Default |
|---|---|---|
| ReachOnlyShield (Stage 1) | `w_bar`, `model_error` | (mục 2) / 0.0 |
| STLShield (Stage 2) | +`epsilon` | 0.05 |
| CEShield (Stage 3) | (không thêm — chỉ thêm chẩn đoán) | — |
| RepairShield (Stage 4) | +`trust_region`, `step_size`, `max_repair_iters`, `max_action_norm` | 0.3 / 0.05 / 1 / 1.0 |

## 4. Tra cứu nhanh — Table II baselines (`shortstop/baselines.py`)

| Baseline | Param | Default |
|---|---|---|
| Conf-Thresh | `disagreement_threshold` | 0.15 |
| STL-Monitor | — (ngưỡng cố định 0, không có param riêng) | — |
| MPC-Filter | `w_bar`, `max_action_norm` | (mục 2) / 1.0 |
| CBF-Shield | `alpha`, `max_action_norm` | 1.0 / 1.0 |

## 5. Tra cứu nhanh — Propose-step policies (`shortstop/policy.py`)

| Policy | Param | Default |
|---|---|---|
| `GaussianChunkPolicy` | `horizon`, `n_candidates`, `noise_std`, `max_speed` | 8 / 8 / 0.3 / 1.0 |
| `DiffusionChunkPolicy` | `n_candidates`, `num_inference_steps` | 8 / 10 |

## 6. Tra cứu nhanh — Calibration recipe (`shortstop/calibration.py`)

| Param | Default |
|---|---|
| `quantile` | 0.99 |
| `safety_factor` | 1.25 |
| `n_episodes` | 200 |

## 7. Tra cứu nhanh — Arm/CALVIN shield stages (`shortstop/arm_shield.py`, Stage 7a/7b)

| Class | Param | Default |
|---|---|---|
| ArmReachOnlyShield | `w_bar`, `model_error` | (truyền vào) / 0.02 |
| ArmSTLShield | +`epsilon` | 0.02 |
| ArmRepairShield | +`trust_region`, `step_size`, `max_repair_iters` | 0.05 / 0.02 / 1 |

Chưa có script nào đo residual thật (Jacobian pseudo-inverse) để calibrate `model_error` kiểu `calibration.py` — nếu cần, phải viết thêm (rollout thật, so `sphere_centers` dự đoán vs đo được, lấy quantile).

## 8. Tra cứu nhanh — Robot geometry constants (`shortstop/robot_geometry.py`)

**Sửa lại 2026-09-05**: 2 dòng dưới đây trước đó claim "đã verify khớp URDF thật" cho `PANDA_DH` — **claim đó không chính xác/stale lúc viết ra**, và bảng này còn trích dẫn API cũ (`SPHERE_RADII`/`SPHERE_FRAME_INDICES`) không còn tồn tại trong code (đã thay bằng `LINK_RADIUS`/`FRAME_RADIUS`/`capsule_segments()`, phủ toàn bộ 8 link chứ không chỉ 4 điểm named). Cái "verify" cũ chỉ là so tay vài hằng số DH (d/a) với `<origin>` của URDF — không phải so sánh FK/pose thật.

**Cập nhật tiếp, cùng ngày — đã verify thật, PASS tuyệt đối**: `scripts/verify_robot_geometry_against_pybullet.py` đã chạy thật trên WSL2 (100 sample từ rollout thật, so `panda_frames()` với PyBullet's real link/base pose). Lần chạy đầu tiên gặp 1 bug RIÊNG của chính script (không phải DH sai): PyBullet's `getLinkState(...)[0]`/`getBasePositionAndOrientation(...)[0]` trả về khung **center-of-mass**, không phải khung joint/link URDF mà `panda_frames()` đại diện — 2 khung chỉ trùng khi link không có offset quán tính (đúng cho flange ảo, sai cho joint thật có khối lượng motor), gây sai lệch giả ~5-13cm ở joint 1-7 + base. Sau khi fix (đọc `getLinkState(..., computeForwardKinematics=1)[4]` + phục hồi khung base qua `getDynamicsInfo`, xem `[[calvin_ga_base_frame_bug_fixed]]`), **kết quả thật: sai lệch = 0.00000m ở CẢ 9 frame** (base, 7 joint, flange) — `PANDA_DH`/`FLANGE_OFFSET` khớp chính xác tuyệt đối với FK thật của CALVIN's PyBullet. `shortstop/robot_geometry.py`'s docstring đã cập nhật để phản ánh đúng hiện trạng (đã verify thật, không còn "chưa chạy với robot/sim thật").

| Const | Giá trị | Ghi chú |
|---|---|---|
| `LINK_RADIUS` | `[0.13, 0.11, 0.10, 0.09, 0.09, 0.10, 0.11, 0.06]` (link0..link7) | **Đã đo từ mesh collision thật** (`mdt_policy/calvin_env/data/franka_panda/meshes/collision/link{0..7}.obj`) — max bán kính cắt ngang (vuông góc trục dài nhất qua SVD), làm tròn lên cm. Chi tiết + số liệu thô: comment ngay trên hằng số này trong code. |
| `FRAME_RADIUS` | dẫn xuất từ `LINK_RADIUS` (9 giá trị, 1/frame) | Frame nội bộ = max bán kính 2 link kề; frame flange = `GRIPPER_TIP_OFFSET + GRIPPER_TIP_RADIUS` (gộp cả phần ngón tay vượt quá flange, vì `propagate_arm_tube` không track orientation để tách riêng). |
| `capsule_segments()` | 8 capsule (link0..link7) | Phủ **toàn bộ** chiều dài từng link (không chỉ 1 điểm/link như `SPHERE_FRAME_INDICES` cũ) — dùng cho ground-truth check (`_clearance`) và (từ 2026-09-05) reachtube's link-Capsule (`propagate_arm_tube`, xem `arm_reach.Capsule`). |
| `JOINT_LIMITS` | (7,2), soft limit từ `mdt_policy/calvin_env/conf/robot/panda.yaml` | **Mới 2026-09-05** — trước đây không có check joint-limit nào cả. Lưu ý: `q=0` (dùng làm placeholder tiện tính toán ở nhiều chỗ trước đây) **tự nó không hợp lệ** cho khớp 4 (`[-3.0718, -0.0698]`, toàn âm) — test suite đã đổi sang 1 pose "ready" thật (`Q_HOME` trong các file test) sau khi phát hiện điều này. |
| `PANDA_DH` | bảng modified-DH | **Đã verify thật (2026-09-05)** — `scripts/verify_robot_geometry_against_pybullet.py` so trực tiếp `panda_frames()` với PyBullet thật của sim CALVIN (100 sample, rollout thật), sau khi fix 1 bug COM-vs-URDF-frame riêng của chính script (xem note ngay trên bảng). Kết quả: sai lệch = 0.00000m ở cả 9 frame — khớp tuyệt đối, không cần chỉnh `PANDA_DH`/`FLANGE_OFFSET`. |
| `HAND_YAW_OFFSET`/`FINGER_JOINT_Z_OFFSET`/`FINGER_RADIUS`/`finger_tip_capsules()` | đọc trực tiếp từ `panda.urdf` | **Mới 2026-09-05** — thay `gripper_tip_position()`/`GRIPPER_TIP_RADIUS` (1 capsule cố định, luôn giả định gripper mở hết cỡ) bằng 2 finger-capsule THẬT, dùng `gripper_opening_width` thật (`obs["robot_obs_raw"][6]`) — xem mục "Category A.2/A.4" bên dưới. |

### Category A.2 (Box không phải sphere) và A.4 (gripper không track mở/đóng) — ĐÃ SỬA (2026-09-05)

- **A.2**: `shortstop/arm_reach.py` không còn dùng `Box` (`shortstop/reach.py`, hình lập phương) — thay bằng `Capsule` (đoạn thẳng `a`-`b` + bán kính vô hướng, `a==b` cho 1 điểm), khoảng cách tính qua `robot_geometry.point_to_segment_distance`/`closest_point_on_segment` — **chính xác tuyệt đối** (không còn over-approximation hình học), đồng thời làm luôn cho link-capsule (trước đó chỉ là AABB bao 2 box, xấp xỉ lỏng hơn capsule thật). **Phát hiện phụ khi sửa**: công thức cũ (Box) từng "quên" trừ bán kính riêng của chính box khi obstacle nằm ngay trong box (chỉ trừ `obstacle.radius`, cho kết quả nông hơn thật) — công thức Capsule mới trừ **cả 2** bán kính (capsule + obstacle), đúng bản chất signed-distance hơn, khiến 1 số giá trị robustness âm sâu hơn trước (vd. obstacle đặt đúng tâm flange: cũ báo `-0.05`, giờ đúng là `-(FRAME_RADIUS[8]+0.05)=-0.21`). Test đã cập nhật theo số liệu mới, verify lại bằng script trước khi sửa assertion.
- **A.4**: `_clearance()`/`_candidate_clearance()` (ground truth, `calvin_experiment.py`) giờ dùng `finger_tip_capsules(joint_angles, gripper_width)` — 2 capsule ngón tay THẬT, lấy `gripper_width` thật từ `obs["robot_obs_raw"][6]` (`gripper_opening_width`, xác nhận qua chính code `calvin_env.robot.Robot.get_observation`), thay vì luôn giả định mở hết cỡ (`GRIPPER_TIP_RADIUS=0.06` cũ). **Chưa sửa cho reachtube của shield** (`propagate_arm_tube`'s `FRAME_RADIUS[8]` vẫn dùng model cũ, 1 sphere bảo thủ) — quyết định phạm vi có chủ đích: threading `gripper_width` real-time qua toàn bộ `_admissible`/`select()`/harness là thay đổi interface lớn hơn hẳn, trong khi ground-truth (quyết định `violation_rate` thật cho MỌI baseline) mới là chỗ giá trị cao nhất. Vẫn conservative (an toàn), chỉ là gap còn lại, không phải bug.

### Category obstacle-shape (X_u chỉ hỗ trợ hình cầu) — CHƯA COVER, note 2026-09-05

- **Gap**: `shortstop/env.py::Obstacle` chỉ là `(center, radius)` — 1 hình cầu duy nhất. `arm_reach._signed_distance` (capsule-vs-sphere, exact) đúng bởi vì phép trừ `- obstacle.radius` chỉ hợp lệ khi obstacle đối xứng mọi hướng quanh tâm — công thức này **không tổng quát** cho 1 X_u không đối xứng (box, plane, mesh thật). Paper gốc định nghĩa $X_u$ trừu tượng, không ràng buộc hình dạng (`report/ShortStop_Report_1.tex` dòng 580) — nên đây là 1 phạm vi implementation **hẹp hơn paper**, dù bản thân không phải bug (obstacle ở CALVIN là virtual/privileged, không mô phỏng vật thể thật nào cụ thể — xem `calvin_obstacle.py`'s docstring).
- **Hệ quả**: `arm_find_counterexample`/CESearch hiện tại exact & sound CHO obstacle hình cầu, nhưng chưa từng được thiết kế/test cho 1 X_u hình dạng khác. Muốn mở rộng (vd. obstacle hình hộp mô phỏng cạnh bàn/tường, một stress-test khó hơn vì có góc/cạnh) cần viết công thức distance mới (capsule-vs-box), không chỉ đổi tham số `radius`.
- **Chưa sửa** — ghi nhận là 1 limitation còn lại so với paper, ngoài phạm vi Category A.2/A.4 đã fix ở trên.

## 9. Tra cứu nhanh — CALVIN/MDT policy sampling (`mdt_policy/conf/mdt_evaluate.yaml`, qua `patches/mdt_policy_shortstop.patch`)

| Param | Default hiện tại |
|---|---|
| `sampler_type` | `ddim` |
| `num_sampling_steps` | 10 |
| `multistep` (= `replan_steps` phía harness) | 10 |
| `sigma_min` / `sigma_max` | 1.0 / 80 |
| `noise_scheduler` | `exponential` |
| `cond_lambda`, `cfg_value` | 1 / 1 |
| `ep_len` | 360 |
| `num_sequences` | 100 (đổi từ gốc 1000 để chạy nhanh khi debug) |
| `n_candidates` (K) | **1** trong `_ForwardOnlyPolicy` (`scripts/run_calvin_unshielded.py`) — hợp lý cho baseline unshielded hiện tại, cần tăng khi wiring shield thật |

## 10. Tra cứu nhanh — CALVIN obstacle injection (`shortstop/calvin_obstacle.py`)

| Param | Default |
|---|---|
| `radius` | 0.05 m |
| `sphere_name` | `SPHERE_NAMES[-1]` = `"gripper"` |

## 11. Tra cứu nhanh — CALVIN experiment harness (`scripts/run_calvin_unshielded.py`)

| Param | Default |
|---|---|
| `SEQUENCE_SEED_BASE` | 1000 (reseed mỗi sequence, xem mục 1's ghi chú về RNG-alignment) |
| `subtasks_per_sequence` (fixed-cohort denominator, `calvin_metrics.py`) | 5 — cố định theo convention CALVIN, không phải tham số nên đổi |
