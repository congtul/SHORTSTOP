# Stage 7b: P-R-C-S cho CALVIN (MDT policy) — design notes

Không chạy thật (chưa có CALVIN/mdt_policy/GPU phù hợp trong môi trường
này) — toàn bộ verify bằng unit test + synthetic data, giống Stage 7a. Code
mới: `shortstop/mdt_policy_client.py`. Code tái dùng **không sửa gì** từ
Stage 7a: `shortstop/robot_geometry.py`, `shortstop/arm_reach.py`,
`shortstop/arm_shield.py` — xem mục 1 vì sao tái dùng được thẳng. Test:
`tests/test_mdt_policy_client.py`, `tests/test_calvin_pipeline_integration.py`
— tất cả pass, không ảnh hưởng test 2D/Stage 7a nào (72/72 tổng).

## 1. Vì sao arm_reach.py/arm_shield.py tái dùng được thẳng, không cần viết lại

Đã research kỹ I/O của MDT (đọc trực tiếp `mdt/models/mdt_agent.py`) và
action space của CALVIN, không đoán:

- **Robot**: CALVIN dùng **Franka Panda 7-DOF** — giống hệt LIBERO. Sphere-
  chain geometry (`robot_geometry.py`, DH table) không đổi 1 dòng.
- **Action space**: CALVIN's "relative cartesian displacement" = 3D delta
  vị trí + 3D delta orientation (Euler) + 1D gripper = 7D/step — **cùng quy
  ước cột 0:3 = delta vị trí task-space** như LIBERO's 6D delta pose + 1D
  gripper (pi0.5). `arm_reach.py`'s `propagate_arm_tube` chỉ đọc cột 0:3,
  bỏ qua rotation/gripper — nên khác biệt Euler-vs-axis-angle ở cột 3:6
  không ảnh hưởng gì.
- **Action shape**: MDT sample `(batch, act_window_size=10, 7)` mỗi lần
  diffusion — cùng dạng "chunk (H, action_dim)" mà `arm_reach.py`/
  `arm_shield.py` đã viết chung cho mọi policy kiểu này (không hardcode
  H hay tên field nào riêng của pi0.5).

$\Rightarrow$ Sự khác biệt thật giữa LIBERO và CALVIN nằm ở **policy client
lấy chunk như thế nào** (mục 2), không nằm ở hình học/toán Reach-Certify.

## 2. Khác biệt thật với LIBERO: MDT tự làm "propose+chấp hành" bên trong nó

Đọc trực tiếp `MDTAgent.step()`:

```python
if self.rollout_step_counter % self.multistep == 0:
    pred_action_seq = self(obs, goal)   # sample 1 chunk (1, 10, 7) mới
    self.pred_action_seq = pred_action_seq
# ... index vào pred_action_seq theo rollout_step_counter, trả 1 action
```

Khác pi0.5 (server chỉ trả `infer()`, **client** (LIBERO eval script) tự
quyết định chấp hành bao nhiêu step trước khi hỏi lại — đúng chỗ
`shortstop/experiment.py:run_episode()` chèn vào được) — MDT's `step()`
**tự cache và tự chấp hành chunk bên trong nó**, không method public nào
trả nguyên chunk thô + để caller tự quyết định chấp hành. Method thực sự
sinh chunk là `self(obs, goal)` (`__call__`/`.forward()`, qua
`denoise_actions`) — nhưng **chưa xác nhận được đây có phải API ổn định/
công khai đúng nghĩa hay là chi tiết implementation nội bộ** (không có
checkout thật để chạy `dir(model)`/đọc docstring chính thức xác nhận).

$\Rightarrow$ Để chèn ShortStop, **không gọi `step()`** (nó đã tự chấp hành
1 phần chunk không qua shield) — phải gọi thẳng vào sub-call sinh chunk
(khả năng `model(obs, goal)`), rồi **tự viết lại vòng lặp chấp hành-prefix-
rồi-certify-lại bên ngoài**, thay hoàn toàn cho vòng lặp `multistep` nội bộ
của MDT — `shortstop/mdt_policy_client.py`'s `MDTPolicyClient.propose()`
đã viết theo hướng này (gọi `self._model(observation, goal)` trực tiếp,
không gọi `.step()`), nhưng **cần verify lại đúng tên method khi có
checkout thật** trước khi tin dùng.

## 3. Test set: dùng public protocol của CALVIN, không tự build

Giống LIBERO ở nguyên tắc (dùng eval chuẩn có sẵn, không tự bịa scenario),
nhưng khác ở hình dạng:

| | LIBERO | CALVIN |
|---|---|---|
| Eval unit | 10 task × 10 seed độc lập/suite | **1000 sequence, mỗi sequence 5 lệnh nối tiếp** (long-horizon) |
| Start | `get_task_init_states(task_id)` | robot reset về vị trí neutral trước mỗi sequence |
| Goal | BDDL predicate symbolic | lệnh ngôn ngữ kế tiếp trong sequence (`multistep_sequences`) |
| Obstacle | không có khái niệm, tự thêm (object khác trong scene) | tương tự — CALVIN không đo "có đụng object khác không", $X_u$ vẫn phải tự định nghĩa qua privileged sim state (PyBullet) |
| Metric official | success rate/suite | Avg. Len. (số subtask liên tiếp hoàn thành, tối đa 5) |

**$X_u$ cụ thể cho CALVIN — chưa code, cùng tình trạng mở như LIBERO**: cần
biết đúng key lấy vị trí object/table trong obs dict của PyBullet env
(`calvin_env`), chỉ xem được khi cài CALVIN thật.

## 4. Câu hỏi thiết kế còn mở

- **Tên method public đúng để lấy chunk thô từ MDT** (mục 2) — quan trọng
  nhất, chặn việc viết `MDTPolicyClient.propose()` đúng thật. Cần checkout
  thật để xác nhận (`dir(model)`, đọc code `denoise_actions` đầy đủ).
- **$g(a)$ (task-progress score)**: giống LIBERO, không có tọa độ đích duy
  nhất (ngôn ngữ). CALVIN's Avg.Len. protocol gợi ý 1 lead khác LIBERO:
  vì mỗi sequence có 5 subtask nối tiếp, có thể dùng "khoảng cách tới hoàn
  thành subtask hiện tại" nếu CALVIN's success-checker expose được — chưa
  code, chưa quyết định.
- **$X_u$ cụ thể** (mục 3) — chưa code, cần key thật của `calvin_env`.

## 5. Vì sao vẫn gộp CE-search + Repair (giống Stage 7a)

Lý do giữ nguyên như `docs/STAGE7A_ARM_PIPELINE_DESIGN.md` mục 5 — không
lặp lại ở đây; `shortstop/arm_shield.py` đã dùng chung cho cả LIBERO và
CALVIN, không có bản riêng cho CALVIN.

## Sources

- [MDT (Multimodal Diffusion Transformer) policy repo](https://github.com/intuitive-robots/mdt_policy) — `mdt/models/mdt_agent.py` (`MDTAgent.step`/`.denoise_actions`), `mdt/evaluation/mdt_evaluate.py`
- [CALVIN benchmark repo](https://github.com/mees/calvin) — action/observation space docs
- [CALVIN: A Benchmark for Language-Conditioned Policy Learning for Long-Horizon Robot Manipulation Tasks (RA-L 2022)](https://www.researchgate.net/publication/361082731_CALVIN_A_Benchmark_for_Language-Conditioned_Policy_Learning_for_Long-Horizon_Robot_Manipulation_Tasks)
- `docs/STAGE7A_ARM_PIPELINE_DESIGN.md` (sphere-chain rationale, CE-search+Repair merge rationale — dùng chung, không lặp lại)
