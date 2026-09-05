# Tuning Workflow — CALVIN, làm mẫu cho LIBERO (và các env sau)

Mục đích file này: **CALVIN (Stage 7b) đang ở giai đoạn thử-sai** (chưa có shield thật, đang tune obstacle radius) — ghi lại đúng trình tự các bước + vai trò từng file ngay trong lúc làm, để khi chuyển qua LIBERO (Stage 7a) việc lặp lại nhanh hơn: biết ngay cái gì tái dùng được 100%, cái gì phải viết lại, và theo đúng pattern nào.

Không lặp lại nội dung tham số (xem `docs/PARAMETERS_REFERENCE.md`) — file này chỉ nói về **trình tự** và **file nào làm gì**.

---

## 0. Quy tắc cohort tuning/eval — áp dụng cho MỌI tham số, mọi lần tune sau này

Mọi lần tune 1 tham số bằng cách chạy nhiều giá trị trên 1 tập sequence rồi
chọn giá trị "tốt nhất" (radius, `disagreement_threshold`, `epsilon`,
`model_error`, ...) **phải tách 2 cohort disjoint**, quy ước cố định:

- **Tuning cohort: sequence idx 0..N-1** (mặc định N=100 — `get_sequences(2*N)[0:N]`,
  tương đương `SEQUENCE_SEED_BASE=1000` + idx 0..99 đã dùng từ trước). Dùng
  cho *mọi* vòng sweep/so sánh giá trị, và cho diagnostic pass nếu có — chọn
  ra giá trị cuối (τ*, radius*, ...) ở đây.
- **Eval cohort: sequence idx N..2N-1 ("100..199")** — disjoint với tuning
  cohort, chạy **đúng 1 lần** với giá trị đã chốt, số ra từ lần chạy này mới
  là số "final" để report/đưa vào paper. Không chọn lại giá trị dựa trên số
  ở eval cohort — nếu eval cho kết quả không ổn, đó là tín hiệu để nghĩ lại
  cách tune, không phải để nhích giá trị cho đẹp số (nhích vậy biến eval
  thành tuning, mất hết ý nghĩa tách cohort).

**Lý do có quy tắc này (rút từ 1 lần đã lỡ không làm)**: bộ sweep radius=0.08
(mục 6 dưới, đã chốt *trước khi* quy tắc này được đặt ra) chọn giá trị VÀ
report số cuối trên **cùng 100 sequence** — không sai về code/logic (số tính
đúng), nhưng có thiên lệch optimistic nhẹ do tuning-on-eval leakage. Mức độ
lệch có thể nhỏ (radius chỉ chọn giữa 4 giá trị rời rạc theo tiêu chí định
tính, không phải tối ưu số liệu chặt) nhưng về nguyên tắc không nên gọi đó là
số "final unbiased". Muốn số radius=0.08 sạch hoàn toàn cần 1 lần chạy xác
nhận riêng trên idx 100..199 — chưa làm, follow-up optional, không chặn việc
khác.

Từ giờ mọi tham số mới đều theo quy tắc 2 cohort này ngay từ đầu —
`disagreement_threshold` của `ArmConfThreshShield` (Conf-Thresh baseline) là
tham số đầu tiên áp dụng đúng theo quy tắc này.

**Cơ chế CLI cụ thể (đã implement trong `scripts/run_calvin_unshielded.py`,
áp dụng cho MỌI script sau này có cơ chế sweep tham số — không cần cho
script chỉ chạy 1 lần như `render_obstacle_video.py`)**:

```bash
python scripts/run_calvin_unshielded.py            # không flag -> eval cohort (mặc định)
python scripts/run_calvin_unshielded.py --tuning    # tuning cohort
```

Implementation pattern (copy nguyên cho script mới):
```python
TUNING_MODE = "--tuning" in sys.argv   # pop ra khỏi sys.argv TRƯỚC khi hydra.main() parse,
if TUNING_MODE:                          # vì hydra chỉ hiểu key=value, --tuning lạ sẽ làm hydra lỗi
    sys.argv.remove("--tuning")
...
N = cfg.num_sequences
COHORT_OFFSET = 0 if TUNING_MODE else N
eval_sequences = get_sequences(2 * N)[COHORT_OFFSET:COHORT_OFFSET + N]
SEQUENCE_SEED_BASE = 1000 + COHORT_OFFSET   # tránh 2 cohort trùng seed
```
`RUN_OUTPUT_DIR` nên gắn suffix `_tuning`/`_eval`, và `results.json` nên ghi
`tuning_mode`/`cohort_sequence_idx_range` để không nhầm khi đọc lại sau.

---

## 1. Trình tự đã đi qua cho CALVIN (làm lại y vậy cho LIBERO)

1. **Xác nhận I/O contract thật** của policy + env bằng cách đọc code thật (không đoán) — action space, obs keys, cách gọi model đúng (CALVIN: `MDTVAgent.forward()` không phải `.step()`, xem `shortstop/mdt_policy_client.py`'s docstring).
2. **Viết harness unshielded**: Propose → execute trực tiếp, không shield, đo baseline risk exposure trước khi có gì để so sánh (`shortstop/calvin_experiment.py`).
3. **Thiết kế obstacle ảo $X_u$**: privileged/geometric, không render vào vision input (quyết định đã chốt, xem `docs/STAGE7B_CALVIN_PIPELINE_DESIGN.md`) — đặt tại endpoint của chunk **thực sự sắp thực thi** (không phải 1 lần sample riêng, để không lệch RNG giữa nhánh with/without).
4. **Viết fixed-cohort metrics** đúng theo convention riêng của benchmark đó (CALVIN: 100 sequences × 5 subtask cố định, khớp cách CALVIN's literature luôn báo cáo theo cohort cố định — xem `docs/REPRODUCING_TABLES_II_VI.md`).
5. **Dựng pattern debug/logging** (radius sweep, `min_clearance` percentile stats, GIF visualization, output dir có `results.json`/`run.log`/`gifs/`) — pattern này **thuần kỹ thuật, không phụ thuộc CALVIN**, áp dụng thẳng.
6. **Tune radius** bằng `violation_rate`/`success_rate` (baseline unshielded) + `min_clearance` percentiles — tránh floor/ceiling effect (mục 1 của `PARAMETERS_REFERENCE.md`).
7. **Wire shield thật vào closed loop** (đã làm cho Conf-Thresh + STL-Monitor — khuôn lặp lại y hệt, chỉ khác class/tham số):
   1. Viết shield class trong `shortstop/arm_shield.py` — 1 class/baseline, tái dùng `_admissible`/`select()` nếu cùng họ reachtube+STL (vd. `ArmSTLMonitorShield(ArmSTLShield)` chỉ zero hoá `w_bar`/`model_error`, giữ `epsilon` làm param tường minh — không hardcode giá trị "theo paper" nếu paper chính nó mâu thuẫn, xem `docs/PARAMETERS_REFERENCE.md`'s "epsilon (STL-Monitor)").
   2. Harness (`calvin_experiment.py::run_calvin_shielded_subtask`) **sửa 1 lần, dùng chung cho MỌI shield** — không sửa lại mỗi khi thêm baseline mới. Baseline mới chỉ cần tuân theo interface `select(joint_angles, candidates, scores) -> (chunk, info)`; 2 hành vi optional được harness tự dò bằng `hasattr`, không cần đổi signature: `.obstacles` (shield "obstacle-aware" cần biết obstacle trước quyết định đầu tiên) và `.recertify(joint_angles, remaining_chunk)` (shield "certify rẻ, không cần sample K mới" — được gọi mỗi bước thật để tách tần suất filter khỏi tần suất policy, xem `docs/PARAMETERS_REFERENCE.md`'s "tach tan suat filter khoi policy"). `ArmConfThreshShield` có cả 2 đều `False`/không có — vẫn chạy đúng như trước, không bị ảnh hưởng.
   3. Script runner riêng mỗi baseline (`run_calvin_shielded.py`, `run_calvin_stl_monitor.py`), copy nguyên khuôn `--tuning`/eval (mục 0) + **3 phase cố định**: (a) diagnostic — tắt filter hoàn toàn (giá trị "vô hiệu hoá" tuỳ chiều reject của baseline: `disagreement_threshold=inf` cho Conf-Thresh, `epsilon=-inf` cho STL-Monitor) để log percentile THẬT của đại lượng đang tune, dùng để revise list sweep; (b) sweep list đó trên **tuning cohort**, chọn `CHOSEN_*` theo tradeoff thật (không đoán); (c) confirm **đúng 1 lần** trên **eval cohort** — số này mới là số final để report.
   4. Ghi quyết định + số liệu vào `docs/PARAMETERS_REFERENCE.md` + memory ngay khi có — đừng để trong đầu/log rời rạc.

   **Bài học đắt nhất, đừng lặp lại**: list sweep ban đầu LUÔN nên coi là placeholder đoán, không phải giá trị tin được — Conf-Thresh's `[0.02, 0.05, 0.1]` ban đầu thấp hơn cả p10 thật (0.146), khiến mọi ngưỡng đều reject ~100% (sweep vô nghĩa, phải revise sau khi thấy percentile thật). **Luôn chạy diagnostic (phase a) trước, nhìn percentile thật, rồi mới tin/sửa list sweep** — đừng chạy thẳng phase (b) với placeholder chưa kiểm chứng.

---

## 2. File map — vai trò & khả năng tái dùng cho LIBERO

| Vai trò | File (CALVIN) | Tái dùng cho LIBERO? | Ghi chú |
|---|---|---|---|
| Robot geometry (sphere-chain, Panda FK) | `shortstop/robot_geometry.py` | ✅ dùng thẳng | Cùng robot Panda, không đổi gì. |
| Reach step (task-space→joint, reachtube) | `shortstop/arm_reach.py` | ✅ dùng thẳng | |
| Shield stages (Reach/STL/Repair/Conf-Thresh/STL-Monitor) | `shortstop/arm_shield.py` | ✅ dùng thẳng | Đã thiết kế sẵn để dùng chung cho cả 2 (Stage 7a/7b), xem docstring đầu file. `recertify()` (tách filter freq khỏi policy freq) cũng nằm sẵn trên `ArmReachOnlyShield`, kế thừa tự động xuống mọi shield họ reachtube. |
| Shielded rollout harness (Certify/Select/recertify loop) | `shortstop/calvin_experiment.py::run_calvin_shielded_subtask` | ⚠️ pattern tái dùng, code viết riêng | Giống dòng "Unshielded rollout harness" dưới — obs keys/action format khác LIBERO nên viết lại, nhưng COPY NGUYÊN cấu trúc vòng lặp (obstacle-then-select cho shield obstacle-aware, execute+recertify mỗi hàng, gripper-fallback fix) — đây là những bug/thiết kế thật đã trải qua, không phải suy đoán. |
| Shielded runner script (mỗi baseline 1 file) | `scripts/run_calvin_shielded.py` (Conf-Thresh), `scripts/run_calvin_stl_monitor.py` (STL-Monitor) | ❌ viết riêng (load model/env khác) | **Copy nguyên khuôn 3-phase** (diagnostic → sweep → final, mục 1.7 ở trên) + cohort split (mục 0) — đây là phần tốn công nghĩ nhất, không phải phần CALVIN-specific. |
| Obstacle injection (privileged, geometric) | `shortstop/calvin_obstacle.py` | ✅ **thực ra đã generic** — chỉ gọi `arm_reach`/`env.Obstacle`/`robot_geometry`, không có gì CALVIN-specific | Dùng thẳng được; nếu muốn rõ ràng hơn có thể đổi tên thành `arm_obstacle.py` khi làm LIBERO (không bắt buộc). |
| Obstacle video visualization | `shortstop/calvin_obstacle_viz.py` + `shortstop/camera_projection.py` | ⚠️ **đổi rồi, không còn generic như trước** — bản cũ (matplotlib, vẽ sơ đồ sphere-chain trừu tượng) đã bỏ, bản mới composite obstacle lên đúng ảnh camera thật (`rgb_static`) bằng `camera.viewMatrix`/`projectionMatrix`/`fov` của calvin_env's `StaticCamera` | `camera_projection.py` (thuần toán chiếu 3D→2D, không import pybullet) tái dùng được nếu LIBERO/openpi cũng expose OpenGL view/projection matrix tương tự; `calvin_obstacle_viz.py`'s `save_sequence_video` thì cần biết tên field ảnh raw (`rgb_static`) + object camera kiểu calvin_env — audit lại theo camera API thật của LIBERO trước khi dùng thẳng. |
| Policy client (I/O đúng với model thật) | `shortstop/mdt_policy_client.py` | ❌ viết riêng | LIBERO đã có sẵn `shortstop/pi_policy_client.py` — vẫn cần audit lại theo đúng cách đã làm ở bước 1 (đọc code thật của openpi, xác nhận forward/propose contract, xem có bị giống lỗi "MDTAgent vs MDTVAgent" hay "thiếu lang_text" đã gặp không). |
| Unshielded rollout harness | `shortstop/calvin_experiment.py` | ❌ viết riêng (obs keys/action format/success-checker của LIBERO khác CALVIN) | **Nhưng copy nguyên pattern bên trong**: `record_trajectory` opt-in, `min_clearance` tracking, reseed-per-sequence (`SEQUENCE_SEED_BASE + idx` dùng chung cho with/without), obstacle lấy từ chunk sắp thực thi (không gọi `propose()` phụ) — đây là những bug thật đã sửa, đừng lặp lại. |
| Fixed-cohort metrics | `shortstop/calvin_metrics.py` | ❌ viết riêng theo convention thật của LIBERO's eval protocol | Cần research trước: LIBERO có early-stop-on-failure giống CALVIN không, hay mỗi task độc lập (không chained subtask) — nếu độc lập thì có thể không cần "fixed-cohort" phức tạp như CALVIN, đơn giản hơn. |
| Runner script (sweep + logging + entrypoint) | `scripts/run_calvin_unshielded.py` | ❌ viết riêng (load model/env khác) | **Copy nguyên pattern logging**: `RUN_OUTPUT_DIR` theo timestamp, `results.json` structured, `run.log` text, `gifs/` cùng dir, hàm `_log()` tee ra cả 2 nơi — để khi test trên máy khác vẫn gửi lại đúng 1 thư mục zip được. |
| Config/patch (debug flag, defaults, paths) | `patches/mdt_policy_shortstop.patch` | ❌ N/A | LIBERO dùng openpi, không liên quan `mdt_policy`; nhưng nguyên tắc "mọi fix nằm trong `patches/`, không sửa trực tiếp submodule upstream" áp dụng như nhau. |
| Tests | `tests/test_calvin_*.py` | ❌ viết riêng | Theo đúng mẫu: `FakeEnv`/`FakePolicy`/`FakeTaskOracle` mock lại interface thật, test control-flow (không cần simulator thật) — xem `tests/test_calvin_experiment.py`. |
| Docs tham số | `docs/PARAMETERS_REFERENCE.md` | ✅ áp dụng ngay | Mục 1 (khái niệm & cách tune) hoàn toàn generic; mục 7 ("Arm/CALVIN shield stages") áp dụng thẳng cho LIBERO vì cùng dùng `arm_shield.py`. Chỉ cần thêm mục tra cứu riêng cho policy sampling config của openpi/pi0.5 (giống mục 9 hiện đang là của MDT). |

---

## 3. "File để tune" — cái gì cần gửi lại sau khi chạy trên máy khác

Từ `scripts/run_calvin_unshielded.py` (và bản LIBERO tương lai nên theo đúng convention này):

| File | Vai trò | Mức độ cần gửi |
|---|---|---|
| `outputs/.../run_<timestamp>/results.json` | Số liệu có cấu trúc (violation_rate/success_rate/clearance_stats/gif_paths mỗi radius) | **Chính** — chỉ cần file này để tune. |
| `outputs/.../run_<timestamp>/run.log` | Bản sao text đầy đủ | Phụ — backup, đọc khi cần trace lại thứ tự log. |
| `outputs/.../run_<timestamp>/gifs/*.gif` | Minh hoạ trực quan tay chạm/né obstacle | Gửi kèm nếu muốn xem trực tiếp, không bắt buộc để tune số. |

Zip cả thư mục `run_<timestamp>/` là đủ, không cần chọn lọc từng file.
