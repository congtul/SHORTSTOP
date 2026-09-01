# LIBERO + $\pi_{0.5}$ setup checklist (Stage 7a, sau khi có WSL2)

Các bước làm ngay sau khi clone repo này (branch dành cho LIBERO) trong
WSL2 Ubuntu. Xem `docs/STAGE7_ENV_SETUP.md` cho bối cảnh/lý do chọn LIBERO
làm benchmark đầu tiên.

## 0. Prereq

- WSL2 + Ubuntu 22.04 đã cài, có CUDA, GPU driver NVIDIA hoạt động trong
  WSL2 (`nvidia-smi` chạy được từ trong WSL2).
- Nếu định dùng Cách B (Docker, mục 4) — cần thêm:
  - Docker Engine cài **thẳng trong Ubuntu WSL2** (không phải Docker Desktop
    cho Windows): `curl -fsSL https://get.docker.com | sh`.
  - **NVIDIA Container Toolkit** — bắt buộc để container thấy được GPU qua
    WSL2 (khác với việc chỉ `nvidia-smi` chạy được ở prereq trên — đó là
    GPU cho tiến trình Linux thường, còn đây là GPU cho tiến trình *bên
    trong container*, cần cấu hình runtime riêng). Cài theo hướng dẫn hiện
    hành của NVIDIA cho Docker + WSL2, xác nhận bằng
    `docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi` chạy ra
    đúng thông tin GPU.

## 1. Clone repo này

```bash
git clone <url-repo> SHORTSTOP
cd SHORTSTOP
git checkout -b libero <base-branch>   # base-branch: 2d-prototype hoặc master, tuỳ bạn
```

## 2. Cài LIBERO (simulator + task suite)

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO
cd LIBERO
pip install -e .
```

**Dataset demo (`download_libero_datasets.py`) KHÔNG bắt buộc** nếu chỉ chạy
eval bằng checkpoint có sẵn ($\pi_{0.5}$ ở bước 3) — chỉ cần khi muốn tự
fine-tune hoặc xem qua data thật. Nếu vẫn muốn tải:

```bash
python benchmark_scripts/download_libero_datasets.py --datasets libero_spatial
# đổi --datasets: libero_spatial | libero_object | libero_goal | libero_100
# hoặc bỏ --datasets để tải cả 4 suite
```

## 3. Cài openpi + lấy checkpoint $\pi_{0.5}$, chạy eval LIBERO

**2 cách thay thế nhau — chọn 1, không phải làm cả 2**: Cách A tự cài mọi
thứ trực tiếp trên WSL2 (đơn giản để hiểu, dễ debug từng bước); Cách B dùng
Docker (openpi khuyến nghị, đóng gói sẵn, dễ chuyển máy sau này — xem mục
"Chuyển sang server/máy mạnh hơn" bên dưới).

### Cách A — Thủ công (không Docker)

```bash
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi
cd openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

Checkpoint đúng cho LIBERO là **`pi05_libero`** — **tự động tải** từ
`gs://openpi-assets/checkpoints/pi05_libero` về `~/.cache/openpi` ngay khi
chạy lệnh serve bên dưới, không cần tải tay:

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_libero \
    --policy.dir=checkpoints/pi05_libero/my_experiment/20000
```

Server lắng nghe port 8000, chờ observation gửi tới — cần tự viết/chạy thêm
script phía LIBERO để gửi observation/nhận action (xem `examples/libero/`
trong repo `openpi` để lấy code mẫu).

### Cách B — Docker (openpi khuyến nghị)

Kiến trúc: `docker compose` dựng **2 container riêng biệt, nói chuyện qua
mạng** — không phải 1 chương trình chạy trong Docker:

```
┌─────────────────────┐   observation (ảnh+state) →  ┌──────────────────────┐
│  Container 1:        │                               │  Container 2:         │
│  Policy Server       │   ← action (7 số/chunk)       │  LIBERO Simulation    │
│  (chạy π0.5 model)   │                               │  (MuJoCo/robosuite)   │
└─────────────────────┘                               └──────────────────────┘
```

Container 1 chỉ chứa deps của `openpi` (JAX/PyTorch, model), Container 2 chỉ
chứa deps của LIBERO (MuJoCo/robosuite) — tách ra để 2 bộ dependency không
xung đột version. Chạy cả 2 bằng đúng 1 lệnh (từ **trong terminal WSL2**,
không phải PowerShell/cmd của Windows):

```bash
SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build
```

Đổi checkpoint/task suite qua biến môi trường, không cần sửa file:

```bash
export SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi05_libero --policy.dir ./my_custom_checkpoint"
export CLIENT_ARGS="--args.task-suite-name libero_10"
```

Gặp lỗi tương thích GPU rendering thì thêm `MUJOCO_GL=glx` trước lệnh. Kết
quả (success rate từng task suite) in thẳng ra stdout khi episode chạy.

## ⚠️ Cảnh báo GPU memory

Theo doc openpi: **inference cần >8GB VRAM** (ví dụ họ dùng RTX 4090),
**fine-tune cần >70GB** (A100/H100). Máy cá nhân dùng RTX 3050 **6GB** —
**thấp hơn mức tối thiểu họ nêu ngay cả cho inference** (không chỉ
fine-tune). $\pi_{0.5}$ là VLA model cỡ lớn (vision+language+action), khác
hẳn model 108K tham số của Stage 6a.

Gợi ý: cứ thử chạy trước (8GB là số mặc định họ nêu, thực tế có thể vừa tuỳ
batch size/precision) — nếu OOM, cần thuê cloud GPU instance Ubuntu
(AWS/GCP/Lambda Labs...) cho LIBERO, không chỉ dành riêng cho ManiSkill2 như
dự tính ban đầu.

## Double-check khi thực hiện

Link/checkpoint/tên config (`pi05_libero`) có thể đổi theo thời gian — luôn
đối chiếu lại README hiện hành của `Lifelong-Robot-Learning/LIBERO` và
`Physical-Intelligence/openpi` trước khi chạy.

## Sau khi chạy được $\pi_{0.5}$ + LIBERO

Việc còn lại là viết lớp `propose()` mới gọi HTTP tới policy server (thay vì
gọi model local như `DiffusionChunkPolicy` ở Stage 6a), rồi mới tính tới
Reach/Certify cho robot 7-DOF thật thay vì point-mass 2D. Nếu đi Cách B
(Docker), chỗ chèn logic P-R-C-S tự nhiên nhất là **Container 2 (LIBERO
eval)** — đó là nơi nhận candidate chunk từ server *trước khi* thực thi,
đúng chỗ shield cần đứng vào giữa. Code shield của `shortstop/` chỉ cần
`numpy`/`scipy`/`cvxpy` — nhẹ, không xung đột với deps sẵn có của container
đó, nên chỉ cần thêm `pip install`/`COPY` code vào Dockerfile của Container
2, không cần dựng container thứ 3 hay đổi kiến trúc.

## Chuyển sang server/máy mạnh hơn sau này (nếu đi Cách B)

Đây chính là lý do Docker được xếp "reproducibility cao nhất" ở
`docs/STAGE7_ENV_SETUP.md`: mọi thứ đóng gói trong image (Python, LIBERO,
openpi, ShortStop, đúng version) đi theo y nguyên khi chuyển máy — không
cần cài lại. Thứ **duy nhất luôn phải làm lại trên máy mới** (dùng Docker
hay không cũng vậy) là **driver GPU + NVIDIA Container Toolkit** ở tầng OS —
nhưng đa số nhà cung cấp cloud GPU (Lambda Labs, RunPod, AWS Deep Learning
AMI...) đã cài sẵn 2 thứ này trong image mặc định, nên bước này trên server
thuê thường **nhanh hơn** tự làm trên WSL2. Sau đó chỉ cần xác nhận
`docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi` chạy đúng, rồi
`docker compose up` (cùng file `compose.yml` đã có ShortStop) chạy y hệt như
trên WSL2 — không phải setup lại từ đầu.
