# Stage 7a: P-R-C-S + baselines cho Panda arm (LIBERO) — design notes

Không chạy thật (chưa có LIBERO/openpi/GPU trong môi trường này) — toàn bộ
verify bằng unit test + synthetic data. Code: `shortstop/robot_geometry.py`,
`shortstop/arm_reach.py`, `shortstop/arm_shield.py`,
`shortstop/pi_policy_client.py`, cộng parametrize nhỏ trong
`shortstop/baselines.py` (`action_dim`). Test:
`tests/test_robot_geometry.py`, `tests/test_arm_reach.py`,
`tests/test_arm_shield.py`, `tests/test_pi_policy_client.py`,
`tests/test_arm_pipeline_integration.py`, `tests/test_baselines_arm_dim.py`
— 21 test mới, tất cả pass, không có test 2D nào bị ảnh hưởng (52/52 tổng).

## 1. Input/output của checkpoint $\pi_{0.5}$ (đã research, xem `docs/LIBERO_SETUP.md`)

Observation gửi lên server: 2 ảnh RGB 224×224 (camera chính + wrist), state
7D (vị trí gripper 3D + axis-angle 3D + gripper 1D), prompt (câu lệnh ngôn
ngữ). Action trả về: **chunk** 7D/step (delta pose 6D + gripper 1D), chạy
theo kiểu **replan** (chỉ thực thi vài step đầu rồi hỏi lại) — đúng khớp
pattern "propose chunk → thực thi 1 phần → certify lại" mà
`shortstop/experiment.py:run_episode()` đã làm cho 2D.

## 2. Vì sao chọn chuỗi sphere thay vì point+circle (đã research literature)

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

Implementation ở đây: 4 sphere (elbow/forearm/wrist/gripper) theo forward-
kinematics thật của Panda (`robot_geometry.py`, DH table modified-DH,
verify một phần qua search — **cần đối chiếu lại với `franka_description`
URDF thật trước khi tin dùng cho robot/sim thật**). Bán kính sphere (0.06–
0.08m) là placeholder thô, chưa fit theo mesh collision thật.

## 3. Vì sao gộp CE-search + Repair thành 1 hàng ablation

Đúng theo yêu cầu — 2D prototype tự nó đã chứng minh Stage 2 (STLShield) và
Stage 3 (CEShield) ra **kết quả giống hệt nhau** (CEShield chỉ thêm chẩn
đoán, không đổi accept/reject) — xem docstring `shortstop/shield.py:
CEShield`. Lặp lại 1 hàng ablation không đổi số cho arm case chỉ tốn công
mà không cho thêm thông tin. `shortstop/arm_shield.py` chỉ có 3 class:
`ArmReachOnlyShield` (Stage 1) → `ArmSTLShield` (Stage 2) →
`ArmRepairShield` (Stage 3+4 gộp — counterexample search vẫn chạy bên
trong, chỉ không phải 1 hàng so sánh riêng).

## 4. Câu hỏi thiết kế còn mở (chưa giải quyết, cần quyết định trước khi chạy thật)

- **$g(a)$ (task-progress score, Eq. 5)**: 2D có sẵn vì có 1 điểm goal cố
  định; LIBERO là task ngôn ngữ, không có tọa độ đích duy nhất. Hiện
  `select()` nhận `scores` từ bên ngoài (caller cung cấp) thay vì tự tính —
  cần quyết định dùng gì (ví dụ: ưu tiên candidate cần repair ít nhất, hay
  policy server có expose value/likelihood nào không).
- **Định nghĩa $X_u$ (unsafe set) cho LIBERO**: đã đề xuất hướng (mặt bàn =
  half-space, object khác trên bàn = sphere bao quanh, giống cách 2D shield
  có quyền truy cập privileged vào vị trí obstacle thật dù policy không
  thấy) — **chưa code**, vì cần biết chính xác key trong observation của
  robosuite (sim state thật, không phải ảnh) cho vị trí object — việc này
  cần làm khi có LIBERO cài thật để xem đúng cấu trúc `obs` dict.
- **Reach step không phải reachtube chứng minh được (sound)**: dùng Jacobian
  pseudo-inverse để suy joint delta từ task-space action — chỉ đúng tại 1
  điểm tuyến tính hóa, giống hệt cảnh báo `MPCFilterShield` đã ghi cho 2D.
  Chỉ mô hình vị trí (position), bỏ qua rotation/gripper hoàn toàn.

## 5. Stress-test có ý nghĩa thật không? (đã research, câu trả lời: có)

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

## Sources

- [ARMTD / Reachability-based Trajectory Design](https://www.researchgate.net/publication/344476553_Reachability-based_Trajectory_Design)
- [Safe Planning for Articulated Robots Using Reachability-based Obstacle Avoidance With Spheres](https://arxiv.org/html/2402.08857v1)
- [Adversarial Stress Testing of SPARK Humanoid Safety Filters](https://arxiv.org/html/2605.19009)
- [Shield-Loco](https://arxiv.org/pdf/2606.07193)
- [Policy Library CBF](https://arxiv.org/html/2605.16588)
- [Interval POMDP Shielding for Imperfect-Perception Agents](https://arxiv.org/pdf/2604.20728)
- [openpi README](https://github.com/Physical-Intelligence/openpi/blob/main/README.md), [examples/libero](https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/main.py)
