# CALVIN + MDT setup checklist (Stage 7b, sau khi có WSL2)

Các bước làm ngay sau khi clone repo này (branch dành cho CALVIN) trong
WSL2 Ubuntu. Xem `docs/STAGE7_ENV_SETUP.md` cho bối cảnh/lý do chọn CALVIN
làm benchmark đầu tiên trong bộ 3 (khác LIBERO/DROID — **chạy local hoàn
toàn được, không cần thuê GPU cloud**).

## 0. Prereq

- WSL2 + Ubuntu 22.04, CUDA + GPU driver NVIDIA hoạt động trong WSL2
  (`nvidia-smi` chạy được từ trong WSL2). **Không cần Docker/thuê server**
  cho CALVIN — model MDT chỉ 40–122M tham số, chạy tự tin trên RTX 3050 6GB.

## 1. Clone repo này

```bash
git clone <url-repo> SHORTSTOP
cd SHORTSTOP
git checkout calvin
```

## 2. Cài CALVIN (simulator PyBullet + task suite)

```bash
git clone --recurse-submodules https://github.com/mees/calvin
cd calvin
sh install.sh   # script cài sẵn của repo -- cài PyBullet + deps + dataset tools
```

**Dataset demo đầy đủ (~500GB, dùng để train) KHÔNG bắt buộc** nếu chỉ chạy
eval bằng checkpoint có sẵn (MDT ở bước 3) — chỉ cần khi muốn tự train/
fine-tune. Eval chuẩn (mục 4) chỉ cần simulator + checkpoint, không cần tải
dataset.

## 3. Cài MDT + lấy checkpoint

```bash
git clone https://github.com/intuitive-robots/mdt_policy
cd mdt_policy
conda env create -f environment.yaml   # hoặc pip install -e ., tuỳ version repo hiện hành
conda activate mdt_env
```

Checkpoint public đã pretrain trên CALVIN (ABCD→D hoặc ABC→D split, xem
README hiện hành của repo để lấy đúng link) — **không dùng checkpoint của
GR-1/RoboFlamingo** (chưa verify được output chunk thật hay chỉ 1 action/
lần) và **không dùng checkpoint "3D Diffuser Actor"** dù nó cũng benchmark
trên CALVIN — đó là model **keypose**-based (giống RLBench/PerAct), không
phải dense-chunk, sai khung P-R-C-S đang dùng.

## 4. Chạy eval chuẩn để xác nhận setup đúng

```bash
python mdt/evaluation/mdt_evaluate.py \
    --checkpoint <path/to/checkpoint> \
    --calvin_dataset_path <path/to/calvin/dataset>  # chỉ cần env assets, không cần data train
```

Eval chuẩn CALVIN (official protocol, dùng chung cho mọi paper so sánh):
**1000 multi-step sequence, mỗi sequence 5 lệnh ngôn ngữ nối tiếp** (long-
horizon), đo "Avg. Len." (số subtask liên tiếp hoàn thành, tối đa 5) — vai
trò tương đương `get_task_init_states()` của LIBERO, nhưng đây là *chuỗi*
task nối tiếp, không phải 10 seed/task độc lập.

## ⚠️ GPU memory — khác hẳn LIBERO/DROID

MDT chỉ **40–122M tham số** (tuỳ variant `mdt_agent.py` config) — nhỏ hơn
$\pi_0$/$\pi_{0.5}$ (VLA-scale) rất nhiều bậc. RTX 3050 6GB chạy inference
(và khả năng cả fine-tune nhẹ) **local được, không cần thuê cloud GPU** —
khác hẳn LIBERO ($\pi_{0.5}$) và DROID ($\pi_0$/$\pi_0$-FAST), cả 2 đều cần
>8GB VRAM theo doc chính thức của `openpi`. Xem `docs/STAGE7_ENV_SETUP.md`
mục 1/1b.

## Double-check khi thực hiện

Link/checkpoint/API (`mdt/evaluation/mdt_evaluate.py`, `MDTAgent.step`) có
thể đổi theo thời gian — luôn đối chiếu lại README + code hiện hành của
`mees/calvin` và `intuitive-robots/mdt_policy` trước khi chạy.

## Sau khi chạy được MDT + CALVIN — khác LIBERO ở đâu (quan trọng)

**MDT's `step(obs, goal)` tự làm luôn việc "propose chunk -> chấp hành 1
phần -> hỏi lại" bên trong nó** (đếm `rollout_step_counter % multistep`,
cache `pred_action_seq`), khác với pi0.5's server chỉ trả `infer()` và để
*client* (LIBERO eval script) tự quyết định chấp hành bao nhiêu step. Nghĩa
là **không thể chèn ShortStop ngay tại `step()`** như đã làm với LIBERO —
phải gọi vào đúng sub-call sinh ra chunk thô (khả năng là
`model(obs, goal)`/`.forward()`, theo code path của `denoise_actions` —
**chưa xác nhận được tên method public chính xác vì chưa có checkout thật
chạy được, xem `shortstop/mdt_policy_client.py`'s docstring**), rồi tự viết
lại vòng lặp "chấp hành `replan_steps` action đầu -> certify lại" bên
ngoài, giống `shortstop/experiment.py:run_episode()` đã làm cho 2D, **thay
thế** vòng lặp `multistep` nội bộ của MDT, không dùng chung với nó.

Sau khi xác nhận đúng method, việc còn lại là viết `propose()` gọi model đó
(đã có sẵn `shortstop/mdt_policy_client.py`) — **Reach/Certify/Repair dùng
lại nguyên `shortstop/arm_reach.py`/`shortstop/arm_shield.py` của Stage 7a,
không cần viết lại**, vì CALVIN's action space (7D: 3D delta pos + 3D delta
Euler + gripper) và robot (Franka Panda) giống LIBERO — xem
`docs/STAGE7B_CALVIN_PIPELINE_DESIGN.md`.
