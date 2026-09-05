# Stage 7a: P-R-C-S + baselines cho Panda arm (LIBERO) — design notes

Không chạy thật (chưa có LIBERO/openpi/GPU trong môi trường này) — toàn bộ
verify bằng unit test + synthetic data. Code: `shortstop/robot_geometry.py`,
`shortstop/arm_reach.py`, `shortstop/arm_shield.py`,
`shortstop/pi_policy_client.py`, cộng parametrize nhỏ trong
`shortstop/baselines.py` (`action_dim`). Test:
`tests/test_robot_geometry.py`, `tests/test_arm_reach.py`,
`tests/test_arm_shield.py`, `tests/test_pi_policy_client.py`,
`tests/test_arm_pipeline_integration.py`, `tests/test_baselines_arm_dim.py`
— tất cả pass, không có test 2D nào bị ảnh hưởng (57/57 tổng).

## 1. Input/output của checkpoint $\pi_{0.5}$ (đã research, xem `docs/LIBERO_SETUP.md`)

Observation gửi lên server: 2 ảnh RGB 224×224 (camera chính + wrist), state
7D (vị trí gripper 3D + axis-angle 3D + gripper 1D), prompt (câu lệnh ngôn
ngữ). Action trả về: **chunk** 7D/step (delta pose 6D + gripper 1D), chạy
theo kiểu **replan** (chỉ thực thi vài step đầu rồi hỏi lại) — đúng khớp
pattern "propose chunk → thực thi 1 phần → certify lại" mà
`shortstop/experiment.py:run_episode()` đã làm cho 2D.

## 2. Test set: dùng public của LIBERO, không tự build

**Có sẵn, không cần tự tạo scenario.** LIBERO có protocol eval chuẩn dùng
chung cho mọi paper (kể cả số $\pi_{0.5}$ công bố): mỗi task suite
(`libero_spatial`/`libero_object`/`libero_goal`/`libero_10`) có 10 task,
mỗi task có initial state **cố định** lấy qua
`task_suite.get_task_init_states(task_id)` — đúng vai trò `make_scenario()`
của 2D. Protocol chuẩn: 10 seed/task × 10 task = 100 rollout/suite.

**Nhưng "start + goal" và "obstacle" không tương ứng 1-1 với 2D** — quan
trọng để không hiểu lầm:

| | 2D (`Reach-Avoid-2D`) | LIBERO |
|---|---|---|
| Start | tọa độ cố định | initial state cố định (`get_task_init_states`) — **có** |
| Goal | 1 tọa độ | **predicate symbolic** trong file BDDL (VD "object A trong region B"), check bằng code chứ không bằng khoảng cách |
| Obstacle | 3 hình tròn tường minh | **không có khái niệm này trong benchmark** — LIBERO chỉ đo task xong chưa, không đo "có đụng object khác không" |

$\Rightarrow$ Phần "an toàn" (né object khác) là khái niệm **tự thêm vào cho
vai trò shield**, không phải thứ LIBERO official đo hay đảm bảo — object
khác trong scene (từ danh sách object của BDDL) được dùng làm sphere-
obstacle cho $X_u$, lấy vị trí qua privileged sim state, không phải qua
LIBERO's success-checker.

**Không dùng ảnh default/tĩnh được** — quan trọng, khác 2D: quan sát của
policy là **ảnh render thật từ MuJoCo**, cập nhật mỗi step. Dùng ảnh tĩnh
sẽ làm policy "không thấy" hậu quả hành động vừa rồi, phá vỡ closed loop
hoàn toàn.

## 3. Full pipeline closed-loop (khác 2D ở đâu)

```
env = LIBERO task (robosuite/MuJoCo), init từ get_task_init_states()
obs = env.reset()   # dict: ảnh camera chính + wrist (render), proprioception thật

loop mỗi replan cycle:
    # Nhánh POLICY (không privileged, giống robot thật ngoài đời):
    element = {ảnh chính, ảnh wrist, state 7D, prompt ngôn ngữ}
    candidates = [pi_client.infer(element)["actions"] for _ in range(K)]
        -> K chunk (H, 7): 6D delta pose + gripper

    # Nhánh SHIELD (privileged, đọc trực tiếp state thật của simulator --
    # y hệt cách shield 2D biết Obstacle thật dù policy không biết):
    joint_angles_thật = env.sim.data...          # Reach: sphere-chain FK
    vị_trí_object_khác_thật = env.sim.data...    # Certify: định nghĩa X_u

    tube = arm_reach.propagate_arm_tube(joint_angles_thật, candidate, ...)
    final_chunk = ArmRepairShield.select(...)     # Reach -> Certify -> Select/Repair

    for a in final_chunk[:replan_steps]:          # chỉ chạy vài step đầu, giống 2D
        obs, reward, done, info = env.step(a)
        # BÊN TRONG env.step(): robosuite tự convert "delta pose 6D + gripper"
        # -> lệnh khớp thật (Operational Space Controller), simulate, RENDER
        # lại ảnh mới -> obs mới. info["success"] = check BDDL predicate.
```

Điểm khác 2D quan trọng nhất: "action" không phải lệnh motor trực tiếp —
robosuite tự chuyển "delta pose 6D" thành lệnh khớp qua 1 controller có sẵn
(OSC), cùng interface robot thật sẽ dùng — không phải thứ mình tự viết.

## 4. Vì sao chọn chuỗi sphere thay vì point+circle (đã research literature)

2D prototype dùng "point + circle" thuần vì đơn giản; **không phải cách
literature thật làm cho tay máy**. Đã tìm và trích dẫn cụ thể:

- **ARMTD** (Autonomous Reachability-based Manipulator Trajectory Design) —
  mô hình hóa **toàn bộ cánh tay** thành chuỗi sphere dọc từng link
  ("Spherical Forward Occupancy"), không chỉ 1 điểm gripper — vì elbow/
  forearm cũng có thể đụng vật cản dù gripper né được.
- Zonotope trong literature này thường dùng cho **joint-space/parameter
  uncertainty**, không phải để vẽ hình vật cản trực tiếp.
- Sphere/capsule được ưa chuộng vì lý do giống hệt circle ở 2D: khoảng
  cách closed-form, rẻ, real-time được — nhưng cho **nhiều điểm dọc cánh
  tay**, không phải 1 điểm.

**Cập nhật 2026-09-05 -- đoạn dưới đây đã lỗi thời trên 2 điểm**: (1) API
đã đổi -- không còn `SPHERE_RADII`/4-sphere, đã thay bằng `LINK_RADIUS`
(8 giá trị) + `capsule_segments()` phủ **toàn bộ** 8 link, không phải 1
subset 4 điểm; (2) claim "đã verify khớp URDF thật" cho DH table
**không chính xác** -- cái đã làm chỉ là so tay vài hằng số DH (d/a) với
`<origin>` của URDF, không phải so sánh FK/pose thật. Xem
`docs/PARAMETERS_REFERENCE.md` mục 8 (đã sửa) và
`scripts/verify_robot_geometry_against_pybullet.py` (verify thật, so
`panda_frames()` trực tiếp với `p.getLinkState()` của sim CALVIN -- viết
xong, chưa chạy, cần WSL2) cho hiện trạng đúng.

Implementation ở đây: forward-kinematics thật của Panda (`robot_geometry.py`,
DH table modified-DH). Bán kính link (`LINK_RADIUS`, đo cho cả 8 link,
không chỉ 4) **đã đo từ mesh collision thật** (không còn placeholder) --
phương pháp: max bán kính cắt ngang vuông góc trục dài nhất (SVD) của
mesh link tương ứng, làm tròn lên cm. Xem chi tiết + số liệu thô trong
comment ngay trên `LINK_RADIUS` (`robot_geometry.py`) và
`docs/PARAMETERS_REFERENCE.md` mục 8. Coverage hình học đã khá hơn: mỗi
link có capsule riêng phủ hết chiều dài (không chỉ elbow/forearm/wrist/
gripper như 4-sphere cũ) -- vẫn đơn giản hơn ARMTD thật ở chỗ reachtube
(`propagate_arm_tube`) trước đây chỉ inflate từng điểm-frame riêng lẻ,
không model đúng capsule giữa 2 frame; đã fix 2026-09-05 (link-box
bounding cả 2 đầu, xem `shortstop/arm_reach.py`'s module docstring).

## 5. Vì sao gộp CE-search + Repair thành 1 hàng ablation

Đúng theo yêu cầu — 2D prototype tự nó đã chứng minh Stage 2 (STLShield) và
Stage 3 (CEShield) ra **kết quả giống hệt nhau** (CEShield chỉ thêm chẩn
đoán, không đổi accept/reject) — xem docstring `shortstop/shield.py:
CEShield`. Lặp lại 1 hàng ablation không đổi số cho arm case chỉ tốn công
mà không cho thêm thông tin. `shortstop/arm_shield.py` chỉ có 3 class:
`ArmReachOnlyShield` (Stage 1) → `ArmSTLShield` (Stage 2) →
`ArmRepairShield` (Stage 3+4 gộp — counterexample search vẫn chạy bên
trong, chỉ không phải 1 hàng so sánh riêng).

## 6. Câu hỏi thiết kế còn mở (chưa giải quyết, cần quyết định trước khi chạy thật)

- **$g(a)$ (task-progress score, Eq. 5)**: 2D có sẵn vì có 1 điểm goal cố
  định; LIBERO là task ngôn ngữ, không có tọa độ đích duy nhất. Lead mới từ
  mục 2: BDDL predicate có thể dùng làm proxy (VD predicate "A trong B" ->
  `-dist(A, B)` làm $g(a)$ tạm, giống cách 2D dùng `-dist(final, goal)`),
  hoặc đơn giản hơn: ưu tiên candidate cần repair ít nhất. Chưa code, chưa
  quyết định cái nào dùng.
- **Định nghĩa $X_u$ cụ thể**: hướng đã rõ hơn (mục 2: object khác trong
  BDDL's object list, trừ object đang thao tác, lấy vị trí qua privileged
  sim state) — **chưa code**, vì cần biết chính xác key trong `obs` dict
  của robosuite cho vị trí object, chỉ xem được khi cài LIBERO thật.
  **Quyết định thiết kế đã chốt (giống CALVIN, xem
  `STAGE7B_CALVIN_PIPELINE_DESIGN.md` mục "Quyết định thiết kế cho $X_u$"
  cho lý do đầy đủ)**: obstacle giữ **thuần privileged/hình học**, không
  spawn object thật vào scene MuJoCo — vì $\pi_{0.5}$ nhận input qua ảnh
  render thật (`observation/image`/`wrist_image`), spawn object thật sẽ
  lộ ra trong ảnh, gây vision-domain-shift confound (policy đổi hành vi vì
  lý do thị giác lạ, không phải vì lý do an toàn). Obstacle chỉ là (center,
  radius) trong code eval/shield, check qua sphere-chain — camera/ảnh render
  không đổi gì so với LIBERO gốc. Bản "obstacle hiện thật trong ảnh" để dành
  làm hướng mở rộng sau, không phải thí nghiệm chính.
- **Reach step không phải reachtube chứng minh được (sound)**: dùng Jacobian
  pseudo-inverse để suy joint delta từ task-space action — chỉ đúng tại 1
  điểm tuyến tính hóa, giống hệt cảnh báo `MPCFilterShield` đã ghi cho 2D.
  Chỉ mô hình vị trí (position), bỏ qua rotation/gripper hoàn toàn.
  **Cập nhật 2026-09-05**: không chứng minh hình thức được, nhưng giờ có
  thể *calibrate* thay vì đoán — `scripts/calibrate_arm_model_error.py`
  đo residual thật (`step_prediction_residual`) trên rollout thật, chưa
  chạy (cần WSL2). `model_error` vẫn hardcode 0.02 cho tới khi có số liệu
  đó.

## 7. Stress-test có ý nghĩa thật không? (đã research, câu trả lời: có)

Tìm được nhiều paper cùng subfield (shield/safety-filter cho robot) đang
làm đúng kiểu "stress test" mà Report 1/2 nhắc tới:

- **Adversarial Stress Testing of SPARK Humanoid Safety Filters** — tên gọi
  đúng y "stress testing" cho safety filter; SPARK là "a modular benchmark
  for safe humanoid control... across robot configurations, tasks,
  obstacles, policies, and safety modules" — xác nhận đây là mối quan tâm
  nghiêm túc, có hẳn benchmark riêng.
- **RoboGate** — dùng bootstrap confidence interval + logistic regression
  trên ngưỡng thất bại, validate trên 30,000 thí nghiệm Isaac Sim.
- **Shield-Loco** — so sánh trực tiếp 2 độ dài horizon (đúng "horizon
  sweep" đã làm ở `scripts/run_horizon_sweep.py`).
- **Policy Library CBF**, **Interval POMDP Shielding** — cùng pattern
  "quét tham số/điều kiện, chạy nhiều Monte Carlo rollout, báo cáo thống
  kê" cho runtime safety filter.

$\Rightarrow$ horizon sweep + bootstrap CI + quét điều kiện đối nghịch
(obstacle đặt xấu, disturbance lớn hơn train) là thực hành chuẩn, có nhiều
paper cùng ngành làm y hệt — không phải chỉ mình Report 1/2 tự nghĩ ra.
Lưu ý: đây là 1 track **tách riêng** khỏi eval chuẩn (mục 2) — eval chuẩn
dùng `get_task_init_states()` nguyên bản; stress-test mới là lúc thêm
object/clutter/disturbance ngoài kịch bản BDDL gốc để test robustness của
shield, không dùng để so sánh success rate với số $\pi_{0.5}$ công bố.

## Sources

- [ARMTD / Reachability-based Trajectory Design](https://www.researchgate.net/publication/344476553_Reachability-based_Trajectory_Design)
- [Safe Planning for Articulated Robots Using Reachability-based Obstacle Avoidance With Spheres](https://arxiv.org/html/2402.08857v1)
- [Adversarial Stress Testing of SPARK Humanoid Safety Filters](https://arxiv.org/html/2605.19009)
- [Shield-Loco](https://arxiv.org/pdf/2606.07193)
- [Policy Library CBF](https://arxiv.org/html/2605.16588)
- [Interval POMDP Shielding for Imperfect-Perception Agents](https://arxiv.org/pdf/2604.20728)
- [openpi README](https://github.com/Physical-Intelligence/openpi/blob/main/README.md), [examples/libero](https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/main.py)
- [LIBERO benchmark (NeurIPS 2023 paper)](https://papers.neurips.cc/paper_files/paper/2023/file/8c3c666820ea055a77726d66fc7d447f-Paper-Datasets_and_Benchmarks.pdf)
- [LIBERO · Hugging Face (eval protocol notes)](https://huggingface.co/docs/lerobot/en/libero)
