# Stage 7c: P-R-C-S + baselines cho Panda arm (RLBench) — design notes

Không chạy thật (chưa có RLBench/Mini Diffuser/GPU trong môi trường này) —
verify bằng unit test + synthetic data, giống Stage 7a. Code:
`shortstop/keypose_reach.py`, `shortstop/keypose_shield.py`,
`shortstop/mini_diffuser_client.py`. Test:
`tests/test_keypose_reach.py`, `tests/test_keypose_shield.py`,
`tests/test_mini_diffuser_client.py`,
`tests/test_keypose_pipeline_integration.py` — tất cả pass, 67/67 tổng
(bao gồm mọi test 2D + Stage 7a).

**Branch `rlbench` tạo từ `libero`** (không phải từ `2d-prototype` trực
tiếp) — vì RLBench dùng **cùng robot Franka Panda 7-DOF** như LIBERO, nên
`shortstop/robot_geometry.py` (DH table, forward-kinematics, 4-sphere
chain) tái dùng được **thẳng, không sửa gì**.

## 1. Khác biệt lớn nhất với Stage 7a: keypose, không phải dense chunk

Đây là phát hiện quan trọng nhất khi research — **khác với điều ghi trong
Report 2** (lúc đó nghĩ Mini Diffuser là "generative chunk policy đúng
kiểu"). Thực tế, theo convention PerAct mà toàn bộ hệ sinh thái RLBench
(PerAct, RVT, Mini Diffuser) dùng chung:

- Demo được cắt thành **keyframe** — thời điểm joint velocity gần 0 VÀ
  gripper đổi trạng thái đột ngột (VD ngay trước/sau khi grasp).
- Policy predict **1 keypose**/lần: 8D = 3D translation + 4D quaternion +
  1D gripper — **không phải chunk $(H, 7)$** như $\pi_{0.5}$.
- **RLBench tự có motion planner (sampling-based)** di chuyển tay từ pose
  hiện tại tới keypose được predict — policy không kiểm soát đường đi
  giữa 2 điểm, chỉ chọn điểm đến.

| | Stage 7a (LIBERO/$\pi_{0.5}$) | Stage 7c (RLBench/Mini Diffuser) |
|---|---|---|
| Output/lần gọi | chunk $(H,7)$ dense | 1 keypose 8D |
| Ai di chuyển tay giữa các điểm | policy tự quyết (chạy `replan_steps` đầu của chunk) | motion planner có sẵn của RLBench, không phải policy |
| Robot | Franka Panda | Franka Panda (giống) |
| Camera | 2 ảnh (main+wrist) | 4 ảnh RGB-D (front/left-shoulder/right-shoulder/wrist) |

$\Rightarrow$ Không tái dùng được `shortstop/arm_reach.py`/`arm_shield.py`
(thiết kế cho dense chunk) — phải viết `keypose_reach.py`/
`keypose_shield.py` riêng, dù dùng lại `robot_geometry.py`.

## 2. Reach step cho keypose — xấp xỉ, không chứng minh được (sound)

`propagate_keypose_tube()`: giải IK số (damped least squares, chỉ vị trí,
bỏ orientation — cùng giản lược như Stage 7a) từ pose hiện tại ra joint
config ứng với keypose target, rồi **nội suy tuyến tính trong joint-space**
giữa 2 config, lấy vài điểm mẫu dọc đường nội suy làm "tube" xấp xỉ.

**Đây chỉ là xấp xỉ đường đi thật** — motion planner sampling-based của
RLBench không có nghĩa vụ đi theo đường thẳng joint-space này. Coi đây là
tiền-kiểm nhanh/thô trên target + xấp xỉ đường đi, không phải bound chứng
minh được.

## 3. Câu hỏi mở quan trọng nhất: certify thêm cái gì mà motion planner chưa làm?

RLBench's motion planner (sampling-based, kiểu OMPL) **đã tự làm collision
avoidance** với scene tĩnh khi di chuyển tới keypose — đây là câu hỏi thiết
kế genuine, chưa trả lời:

- Có thể planner chỉ biết obstacle nó được cho biết trước (không phải mọi
  object động/mới), nên certify lại vẫn có giá trị.
- Certify target TRƯỚC KHI gọi planner (rẻ) có thể lọc bỏ candidate rõ
  ràng tệ (target ở trong vật khác) mà không cần tốn compute chạy motion
  planning đầy đủ.
- Nhưng **không nên tuyên bố** ShortStop ở đây thay thế/bổ khuyết cho
  planner's guarantee — cần đọc kỹ RLBench's motion planner code
  (`stepjam/RLBench`) để biết chính xác nó certify cái gì, trước khi biết
  ShortStop thêm giá trị gì thật.

## 4. Test set: public của RLBench (giống LIBERO, xem `docs/RLBENCH_SETUP.md`)

Cấu trúc Task > Variation > Episode, 18 task × 249 variation, 1 eval seed
cố định trên toàn benchmark — **dùng thẳng**, không tự build. Y hệt LIBERO
(`docs/STAGE7A_ARM_PIPELINE_DESIGN.md` mục 2): **không có khái niệm
"obstacle" tường minh** trong benchmark — object khác trong scene (privileged
sim state, không phải ảnh policy thấy) mới là nguồn cho $X_u$, tự thêm vào
cho vai trò shield, không phải RLBench official đo.

## 5. Câu hỏi mở khác (kế thừa từ Stage 7a, chưa giải quyết)

- $g(a)$: RLBench task cũng ngôn ngữ-điều kiện, không có 1 tọa độ đích —
  cùng vấn đề như LIBERO.
- $X_u$ cụ thể: cần biết đúng key trong observation của RLBench/PyRep cho
  vị trí object khác — chỉ xem được khi cài RLBench thật.
- Interface Mini Diffuser thật (client/server hay in-process?) — chưa xác
  nhận, xem `docs/RLBENCH_SETUP.md`.

## Sources

- [RLBench paper](https://arxiv.org/pdf/1909.12271), [stepjam/RLBench](https://github.com/stepjam/RLBench)
- [PerAct (keyframe extraction convention)](https://peract.github.io/)
- [Mini Diffuser (arXiv 2505.09430)](https://arxiv.org/pdf/2505.09430), [utomm/mini-diffuse-actor](https://github.com/utomm/mini-diffuse-actor)
- [hqfang/rlbench-18-tasks dataset](https://huggingface.co/datasets/hqfang/rlbench-18-tasks)
