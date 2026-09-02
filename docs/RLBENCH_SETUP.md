# RLBench + Mini Diffuser setup checklist (Stage 7c, sau khi có WSL2)

Xem `docs/STAGE7_ENV_SETUP.md` cho bối cảnh chung, `docs/LIBERO_SETUP.md`
cho checklist tương tự của LIBERO (Stage 7a) — file này theo đúng cấu trúc
đó. Khác LIBERO: RLBench dùng CoppeliaSim/PyRep (không phải MuJoCo), và
Mini Diffuser predict **keypose** (không phải dense action chunk) — xem
`docs/STAGE7C_ARM_PIPELINE_DESIGN.md` mục 1 cho chi tiết khác biệt này.

## 0. Prereq

Giống `docs/LIBERO_SETUP.md` mục 0: WSL2 + Ubuntu 22.04, CUDA/GPU driver
hoạt động trong WSL2, cộng Docker Engine + NVIDIA Container Toolkit nếu đi
theo hướng container hóa.

## 1. Cài CoppeliaSim + PyRep

```bash
# tải CoppeliaSim (Edu version) từ trang chính thức Coppelia Robotics,
# rồi set biến môi trường (thêm vào ~/.bashrc):
export COPPELIASIM_ROOT=<path/to/CoppeliaSim>
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
source ~/.bashrc

git clone https://github.com/stepjam/PyRep.git pyrep && cd pyrep
pip install -r requirements.txt
pip install .
```

Test PyRep cài đúng:
```bash
python pyrep/examples/example_youbot_navigation.py
```

## 2. Cài RLBench

```bash
git clone https://github.com/stepjam/RLBench.git rlbench && cd rlbench
pip install -e .
```

## 3. Lấy demo data (nếu cần) + checkpoint Mini Diffuser

Demo data chính thức cho 18-task benchmark (PerAct's chuẩn, Mini Diffuser
dùng chung): HuggingFace dataset `hqfang/rlbench-18-tasks`. Giống LIBERO —
**không bắt buộc** nếu chỉ chạy eval bằng checkpoint có sẵn.

```bash
git clone https://github.com/utomm/mini-diffuse-actor
```

**Chưa xác nhận được link checkpoint cụ thể** (README repo không ghi rõ
địa chỉ HuggingFace/Google Drive) — vào trực tiếp repo
`utomm/mini-diffuse-actor` xác nhận trước khi tải, repo tự nhận có kèm
"checkpoints, training logs, and test logs".

## ⚠️ Interface chưa xác nhận — khác LIBERO ở điểm này

LIBERO/openpi có tài liệu I/O rõ ràng (`examples/libero/main.py`,
`examples/libero/README.md`). Mini Diffuser's repo **không có tài liệu
tương đương** cho cách serve/gọi model — nhìn giống 1 codebase train+eval
chạy in-process với RLBench, không phải client/server tách rời như openpi.
`shortstop/mini_diffuser_client.py:MiniDiffuserClient` (real client) vì vậy
đang để `NotImplementedError` với ghi chú rõ — **cần đọc trực tiếp code
repo đó** (không phải chỉ README) để biết cách gọi model đúng, trước khi
viết client thật.

## Double-check khi thực hiện

Link repo/checkpoint/dataset có thể đổi theo thời gian — luôn đối chiếu lại
`stepjam/RLBench`, `stepjam/PyRep`, `utomm/mini-diffuse-actor` trước khi
setup.

## Test set: dùng public của RLBench, không tự build

Giống LIBERO (`docs/STAGE7A_ARM_PIPELINE_DESIGN.md` mục 2) — RLBench có
cấu trúc Task > Variation > Episode với seed eval cố định, dùng chung bởi
mọi paper trên benchmark này (PerAct, RVT, Mini Diffuser): 18 task, 249
variation, 1 eval seed cố định trên toàn benchmark. Dùng thẳng protocol
này, không tự tạo scenario riêng — xem
`docs/STAGE7C_ARM_PIPELINE_DESIGN.md` mục 3 cho chi tiết + điểm khác biệt
so với LIBERO/2D (không có khái niệm "obstacle" tường minh, giống LIBERO).
