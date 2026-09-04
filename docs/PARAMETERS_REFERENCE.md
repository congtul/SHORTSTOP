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
- **Output cần theo dõi**: giống `w_bar` — `violation_rate` > 0 dù shield chấp nhận chunk ⇒ tăng `model_error`. Cách đo cụ thể cho arm: log lại `||sphere_centers(q_thật) - sphere_centers(q_ước_lượng_từ_Jacobian)||` trên rollout thật, lấy quantile cao — chưa có script nào làm việc này, cần viết thêm (xem mục 7).

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

**TODO (chưa implement, chỉ note lại cho việc sau)**: khi implement MPC-Filter/CBF-Shield/STL-Monitor/ShortStop cho CALVIN, áp dụng ý tưởng "tách tần suất filter khỏi policy" ở trên — propose thưa (mỗi `replan_steps` bước, giữ tỉ lệ ~0.5 theo Diffusion Policy làm điểm khởi đầu tham khảo, hoặc =1 nếu muốn khớp đúng Thm 1), nhưng certify/filter chạy mỗi bước thật trên phần đuôi chunk còn lại + state thật. Cần: (1) 1 method mới trên mỗi shield class kiểu `recertify(real_state, remaining_chunk_suffix)` tách biệt với `select(joint_angles, candidates, scores)`, (2) sửa lại cấu trúc loop trong harness (`run_calvin_shielded_subtask`) để gọi `recertify` mỗi bước giữa 2 lần `propose()`.

### replan_steps trong g(a)/disagreement (Conf-Thresh) — đã fix
`ArmConfThreshShield._endpoint()` và `calvin_progress.calvin_progress_scores()` giờ đo predicted endpoint tại `chunk[:replan_steps]`, KHÔNG phải toàn bộ chunk (`H`/`act_window_size`) — cả 2 đều nhận `replan_steps` làm tham số **bắt buộc** (không default), để không âm thầm rơi lại về "đo cả chunk". Trước khi fix, code đo `tube[-1]` của TOÀN BỘ chunk — chỉ "đúng một cách trùng hợp" vì `act_window_size == replan_steps == 10` theo config default hiện tại, không phải vì code có chủ đích liên kết 2 giá trị này.

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

**✅ ĐÃ CHỐT: radius = 0.08m** — kết quả sweep thật, chạy trên checkpoint/dataset thật (n_sequences=100 mỗi radius, sau khi capsule-chain fix ở trên đã áp dụng):

| radius | violation_rate | success_rate | avg_seq_len | clearance p10 | clearance min | min + radius |
|---|---|---|---|---|---|---|
| 0 (baseline, không obstacle) | — | 0.930 | 4.65/5 | — | — | — |
| 0.02 | 0.076 | 0.704 | 3.52/5 | +0.0042 | -0.1289 | -0.1089 |
| 0.05 | 0.108 | 0.586 | 2.93/5 | -0.0084 | -0.1589 | -0.1089 |
| **0.08 (chốt)** | **0.128** | **0.482** | **2.41/5** | **-0.0286** | **-0.1889** | **-0.1089** |
| 0.12 | 0.148 | 0.412 | 2.06/5 | -0.0732 | -0.2289 | -0.1089 |

Nhận xét khi tune: (1) `min_clearance + radius` ra **đúng hằng số -0.1089 ở mọi radius** — vì rollout dừng ngay khi violated, nên "overshoot" tối đa chỉ phụ thuộc chuyển động 1 bước của policy, không phụ thuộc radius; xác nhận `_clearance()`/dừng-khi-violated hoạt động đúng thiết kế, không phải bug. (2) Không có floor effect ở 0.02 (violation_rate 7.6% đã là tín hiệu thật, không gần 0%) và chưa chạm ceiling ở 0.12 (14.8%, còn xa 100%) — toàn bộ range 0.02–0.12 đều "dùng được", 0.08 được chọn vì cân bằng: violation_rate đủ cao để shield có việc làm, success_rate còn 48.2% (chưa về 0, còn khoảng để shield show cải thiện). (3) success_rate giảm rất nhanh dù radius nhỏ (93%→70.4% chỉ với r=0.02) — do cấu trúc chained-sequence (1 subtask violated làm mất hết điểm các subtask sau trong `build_fixed_cohort_slots`), không phải bug, nhưng cần nhớ khi so sánh giữa CALVIN và LIBERO nếu LIBERO không chain task như vậy.

Muốn sweep lại (checkpoint mới, hoặc muốn xem ceiling effect ở radius > 0.12): sửa `RADII_TO_SWEEP` trong `scripts/run_calvin_unshielded.py`.

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

| Const | Giá trị | Ghi chú |
|---|---|---|
| `SPHERE_RADII` | `[0.09, 0.10, 0.06, 0.05]` m | **Đã đo từ mesh collision thật** (`mdt_policy/calvin_env/data/franka_panda/meshes/collision/{link3,link5,link7,hand}.obj`) — max bán kính cắt ngang (vuông góc trục dài nhất qua SVD, không phải khoảng cách thô từ gốc vì sẽ lẫn chiều dài link), làm tròn lên cm gần nhất. Chi tiết phương pháp + số liệu thô: xem comment ngay trên `SPHERE_RADII` trong code. Vẫn là 1 sphere/link (không phủ hết toàn bộ chiều dài link, links 1/2/4/6 không có sphere riêng) — số đã đúng vật lý hơn nhưng độ phủ hình học (coverage) vẫn thô như cũ. |
| `SPHERE_FRAME_INDICES` | `[3, 5, 7, 8]` | elbow/forearm/wrist/gripper — **đã verify khớp URDF thật**: mọi hàng `PANDA_DH` đối chiếu đúng với `<origin>` của từng `<joint>` trong `panda.urdf` (d=0.333/a=0.0825/d=0.384/a=0.088/flange 0.107 khớp chính xác), và `positions[3]/[5]/[7]` đúng là gốc của `panda_link{3,5,7}`, `positions[8]` (flange) trùng vị trí gốc của `panda_hand` (chỉ lệch góc xoay). |
| `PANDA_DH` | bảng modified-DH | **đã verify với URDF thật** (xem dòng trên) — không còn "chưa verify". |

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
