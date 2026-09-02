# CALVIN + MDT setup checklist (Stage 7b, sau khi có WSL2)

Các bước làm ngay sau khi clone repo này (branch `calvin`) trong WSL2
Ubuntu. Xem `docs/STAGE7_ENV_SETUP.md` cho bối cảnh/lý do chọn CALVIN làm
benchmark đầu tiên trong bộ 3 (khác LIBERO/DROID — **chạy local hoàn toàn
được, không cần thuê GPU cloud**).

**Đã verify chạy được thật trên WSL2 Ubuntu 22.04 + RTX 3050 6GB** (checkpoint
`mdtv-1-abcd`, dataset debug) — quy trình dưới đây là bản đã sửa lại từ
chính log lỗi thật gặp lúc setup, không phải suy đoán.

## 0. Prereq

- WSL2 + Ubuntu 22.04, CUDA + GPU driver NVIDIA hoạt động trong WSL2
  (`nvidia-smi` chạy được từ trong WSL2). **Không cần Docker/thuê server**
  cho CALVIN — model MDT chỉ ~22.5M tham số (`Number of parameters` in log
  thật khi load checkpoint `mdtv-1-abcd`), chạy tự tin trên RTX 3050 6GB.
- **Chỉ cần 1 conda environment** cho toàn bộ pipeline (xem mục "Vì sao chỉ
  1 environment" bên dưới) — không cần môi trường riêng cho "CALVIN" và
  "MDT".

## 1. Clone repo này + submodule

```bash
git clone --recurse-submodules <url-repo> SHORTSTOP
cd SHORTSTOP
git checkout calvin
git submodule update --init --recursive   # kéo mdt_policy + mdt_policy's own calvin_env/tacto
```

Repo chỉ còn **1 submodule** (`mdt_policy` — trỏ tới
`intuitive-robots/mdt_policy`). `mdt_policy` tự vendor `calvin_env`
(simulator PyBullet) làm submodule con của chính nó — **không cần clone
riêng `mees/calvin`** (xem giải thích ở cuối file).

## 2. Apply SHORTSTOP's patch cho `mdt_policy`

`mdt_policy` (submodule, upstream) không được sửa trực tiếp/commit — mọi
thay đổi cần cho pipeline chạy được nằm trong `patches/mdt_policy_shortstop.patch`,
apply bằng:

```bash
bash scripts/apply_mdt_patch.sh
```

Patch này sửa (không đụng research logic của MDT):
- `conf/mdt_evaluate.yaml`: path/config machine-specific của tác giả gốc →
  đổi thành path tương đối trong repo này, `num_sequences: 1`/`debug: True`
  cho smoke test (đổi lại `num_sequences: 1000`/`debug: False` khi chạy eval
  thật).
- `mdt/evaluation/mdt_evaluate.py`: fix bug `evaluate_policy()` chỉ được gọi
  bên trong `if log_wandb:` (nên khi `log_wandb: False`, eval không chạy
  luôn) — patch tách `evaluate_policy()` ra khỏi điều kiện đó, W&B chỉ còn
  là optional logging. Cũng comment `join_vis_lang()` (gọi `cv2.imshow()`)
  vì WSL headless không có Qt/X11 (`Could not load the Qt platform plugin
  "xcb"`), giữ nguyên phần log/GIF recording.
- `mdt_policy/calvin_env/.gitmodules` (submodule con): đổi URL `git@github.com:...`
  → `https://github.com/...` (tránh cần SSH key để clone).

## 3. Setup environment + cài đặt (1 environment duy nhất)

```bash
conda create -n mdt_env python=3.8
conda activate mdt_env

cd mdt_policy/calvin_env/tacto
pip install -e .
cd ..
pip install -e .          # calvin_env (simulator PyBullet), torch-agnostic
cd ..

pip install setuptools==57.5.0    # BẮT BUỘC trước khi build pyhash (xem lỗi #1 dưới)
cd pyhash-0.9.3
python setup.py build
python setup.py install
cd ..

pip install -r requirements.txt   # torch==2.0.1, transformers, pytorch-lightning, ...
pip install torchvision==0.15.2   # requirements.txt không pin, dễ resolve sai nếu bỏ qua
pip install networkx==2.8.8       # override transitive dep cũ (xem lỗi #3 dưới)
```

Sau đó set `LD_LIBRARY_PATH` **riêng cho environment này** (không đặt global
`.bashrc`, tránh ảnh hưởng environment khác) — cần cho `libcudnn`/`libcuda`
trong WSL2:

```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
cat > $CONDA_PREFIX/etc/conda/activate.d/cuda_libs.sh <<'EOF'
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$CONDA_PREFIX/lib/python3.8/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
EOF
conda deactivate && conda activate mdt_env   # để activate.d hook chạy
```

Toàn bộ quy trình trên đã đóng gói sẵn thành 1 script:
`scripts/setup_calvin_env.sh` (idempotent, không hardcode path máy cụ thể
nào — tự suy ra từ vị trí repo).

### Các lỗi đã gặp thật lúc setup (và fix)

1. **`pyhash` fail: `error in pyhash setup command: use_2to3 is invalid`** —
   `setuptools` mới bỏ hỗ trợ `use_2to3`. Fix: `pip install setuptools==57.5.0`
   **trước** khi build `pyhash-0.9.3` (thứ tự quan trọng — cài sau vẫn lỗi
   vì `pyhash` build 1 lần lúc `python setup.py build`).
2. **`MulticoreTSNE` cần cmake, máy không có / version PyPI không khớp** —
   **không còn gặp phải nữa** sau khi bỏ submodule `calvin` top-level:
   `MulticoreTSNE` chỉ được `calvin_models`'s visualization utilities dùng
   (`calvin_agent/visualization/tsne_plot.py`), **không nằm trong code path
   của `mdt_evaluate.py`** — đã verify bằng cách chạy thật eval mà không
   cần cài `calvin_models`/cmake gì cả.
3. **`networkx`/`np.int`**: `networkx==2.2` (kéo transitive qua `gym`/
   `urdfpy`) dùng `np.int`, đã bị xoá ở NumPy hiện tại. Fix: pin tường minh
   `networkx==2.8.8` (không tự nhiên được resolver chọn nếu không ép).
4. **CUDA**: `Could not load library libcudnn_cnn_infer.so.8` /
   `libcuda.so: cannot open shared object file` trong WSL2 — do
   `libcuda.so` thật nằm ở `/usr/lib/wsl/lib` (không phải path CUDA chuẩn),
   `cudnn` nằm trong site-packages của `nvidia-cudnn-cu*` (pip package, đi
   kèm torch). Fix: `LD_LIBRARY_PATH` set qua `activate.d` hook (mục 3
   trên), không set global.
5. **`cv2.imshow()`/`cv2.waitKey()` crash trên WSL headless** (`debug: True`)
   — patch đã comment dòng `join_vis_lang()` gọi GUI, giữ log/GIF text.

## 4. Chạy smoke test

```bash
python scripts/smoke_test_calvin.py
```

Check: Python/torch/torchvision version, CUDA available + tên GPU, import
`calvin_env`/`mdt`/`tacto`/`shortstop` được không, checkpoint path tồn tại.
Fail sớm với message rõ ràng, không chạy tới bước load model/eval nếu thiếu
gì.

## 5. Dataset — chỉ cần bản debug cho smoke test

**Không cần full dataset (166GB cho D→D, 656GB cho ABCD→D)** để verify
setup hay chạy eval bằng checkpoint có sẵn. Dùng bản debug:

```bash
cd mdt_policy/dataset
sh download_data.sh debug
```

`dataset_path` trong `conf/mdt_evaluate.yaml` (đã patch trỏ vào
`mdt_policy/dataset/calvin_debug_dataset`) chỉ thực sự đọc `lang_annotations/
embeddings.npy` + file thống kê normalization từ đó — không đọc dữ liệu
train đầy đủ (xem `docs/STAGE7B_CALVIN_PIPELINE_DESIGN.md` nếu cần chi tiết
code path). `evaluation/multistep_sequences.py` (1000-sequence eval
protocol) hoàn toàn tách biệt, không phụ thuộc dataset đã tải.

## 6. Checkpoint

Checkpoint public: repo cung cấp đúng **6 checkpoint** (`mdtv-{1,2,3}-abcd`,
`mdtv-{1,2,3}-d`), qua [Google Drive](https://drive.google.com/drive/folders/13EDBcdYyOV7FsF9Z7Eb0YN8aMTrtsAsi)
— thực ra chỉ là **2 split × 3 seed**, không phải 6 model khác kiến trúc.

| | Split | Ý nghĩa | Avg. Len. (tối đa 5) |
|---|---|---|---|
| `mdtv-{1,2,3}-abcd` | **ABCD → D** | train trên data gộp cả 4 layout A+B+C+D, eval trên layout D (D **đã có** trong training data — không phải "chưa thấy") | 4.57–4.66 |
| `mdtv-{1,2,3}-d` | **D → D** | train **chỉ** trên data của layout D, eval trên layout D | 3.67–3.74 |

Số `1/2/3` (seed 142/242/42) chỉ là 3 lần train lại để paper báo cáo mean/
variance — không khác ý nghĩa, chọn 1 trong nhóm là được. **Không có
checkpoint public cho split ABC→D** (train A,B,C, eval D chưa thấy — split
"zero-shot environment mới" hay được nhắc trong literature CALVIN).

**Dùng `mdtv-1-abcd`** (seed 142, avg len 4.66 cao nhất, checkpoint file
tên `avg_seq_len=4.64.ckpt` — số trong tên file là kết quả reproduce cục bộ
của tác giả, không phải hằng số cố định) — đã verify chạy được thật, cho
kết quả 4/5 subtask thành công trên 1 sequence smoke test:
`open_drawer → turn_on_lightbulb → move_slider_left → rotate_red_block_left
→ lift_red_block_table` (subtask cuối fail — bình thường, không phải lỗi
setup).

**A/B/C/D là gì — dễ hiểu nhầm là "CALVIN có 4 loại robot/task khác nhau",
KHÔNG phải vậy**: CALVIN ship sẵn **4 bản mô phỏng (layout) khác nhau** của
cùng 1 kiểu bàn thao tác — cùng robot (Panda), cùng bộ cơ cấu (cửa trượt,
ngăn kéo, nút đèn, switch, block màu), cùng vị trí bàn/camera tĩnh, nhưng
**texture bàn + vị trí đặt các cơ cấu đó khác nhau** giữa A/B/C/D.

**Không dùng checkpoint của GR-1/RoboFlamingo** (chưa verify được output
chunk thật hay chỉ 1 action/lần) và **không dùng checkpoint "3D Diffuser
Actor"** dù nó cũng benchmark trên CALVIN — đó là model **keypose**-based
(giống RLBench/PerAct), không phải dense-chunk, sai khung P-R-C-S đang
dùng.

## 7. Chạy eval chuẩn (đầy đủ, không phải smoke test)

Sửa `mdt_policy/conf/mdt_evaluate.yaml` (sau khi patch): `num_sequences: 1000`,
`debug: False`, rồi:

```bash
cd mdt_policy
python mdt/evaluation/mdt_evaluate.py
```

Eval chuẩn CALVIN (official protocol, dùng chung cho mọi paper so sánh):
**1000 multi-step sequence, mỗi sequence 5 lệnh ngôn ngữ nối tiếp**
(long-horizon), đo "Avg. Len." (số subtask liên tiếp hoàn thành, tối đa 5).

## Vì sao chỉ cần 1 environment (không cần submodule `mees/calvin` riêng)

Setup ban đầu (bản cũ của doc này) dạy clone thêm `mees/calvin` (top-level,
cho `calvin_models`) làm 1 conda env riêng (`torch==1.13.1`, khác
`mdt_env`'s `torch==2.0.1`) — **đây là bước dư thừa, đã verify bỏ được**:

- README chính thức của `mdt_policy` chỉ dạy đúng 1 environment, không có
  bước clone `mees/calvin` nào.
- `mdt_policy` tự vendor `calvin_env` (submodule con, PyBullet simulator —
  **không có `torch` trong dependency** của nó) — đủ cho cả `env.reset()`/
  `env.step()` và task-success checker (`calvin_env.envs.tasks.Tasks`).
- Grep toàn bộ `mdt_policy/mdt/` và `shortstop/`: **0 chỗ import
  `calvin_models`**. Cái duy nhất cần `calvin_models` là chính bộ
  visualization/tsne-plot của `calvin_models`'s own baseline (HULC), không
  liên quan tới eval loop của MDT.
- **Đã verify bằng chạy thật**: eval 1 sequence chạy trọn (4/5 subtask
  success) chỉ trong `mdt_env`, dùng `mdt_policy/calvin_env` (log xác nhận
  `Using calvin_env with commit 797142c...` — đúng commit của bản vendor
  trong `mdt_policy`, không phải bản top-level `calvin` cũ).

## Sau khi chạy được MDT + CALVIN — khác LIBERO ở đâu (quan trọng)

**MDT's `step(obs, goal)` tự làm luôn việc "propose chunk -> chấp hành 1
phần -> hỏi lại" bên trong nó** (đếm `rollout_step_counter % multistep`,
cache `pred_action_seq`), khác với pi0.5's server chỉ trả `infer()` và để
*client* (LIBERO eval script) tự quyết định chấp hành bao nhiêu step. Nghĩa
là **không thể chèn ShortStop ngay tại `step()`** như đã làm với LIBERO —
phải gọi vào đúng sub-call sinh ra chunk thô. **Đã xác nhận (đọc code thật
từ checkout local `mdt_policy/mdt/models/mdtv_agent.py`, lớp `MDTVAgent` —
đúng lớp đứng sau 6 checkpoint `mdtv-*` public, không phải `MDTAgent` cũ
hơn)**: `model(obs, goal)`/`.forward()` gọi `denoise_actions()` rồi trả
thẳng `act_seq`, không cache/index gì thêm — đúng là chunk thô cần dùng.
Cũng đã xác nhận K lần gọi ra K chunk **thật sự khác nhau** (fresh
`torch.randn` mỗi lần trong `denoise_actions`, không seed lại). ShortStop
tự viết lại vòng lặp "chấp hành `replan_steps` action đầu -> certify lại"
bên ngoài, giống `shortstop/experiment.py:run_episode()` đã làm cho 2D,
**thay thế** vòng lặp `multistep` nội bộ của MDT, không dùng chung với nó.

Việc còn lại là viết `propose()` gọi model đó (đã có sẵn, đúng lớp/method,
trong `shortstop/mdt_policy_client.py`) — **Reach/Certify/Repair dùng
lại nguyên `shortstop/arm_reach.py`/`shortstop/arm_shield.py` của Stage 7a,
không cần viết lại**, vì CALVIN's action space (7D: 3D delta pos + 3D delta
Euler + gripper) và robot (Franka Panda) giống LIBERO — xem
`docs/STAGE7B_CALVIN_PIPELINE_DESIGN.md`.
