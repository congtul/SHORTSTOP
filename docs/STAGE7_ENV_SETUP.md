# Stage 7a–7c: thứ tự setup 3 benchmark thật + ghi chú

Ghi chú thực hành cho khi bắt đầu Stage 7a/7b/7c, dựa trên kết luận đã bàn
khi đánh giá checkpoint public + hạ tầng cho máy cá nhân (RTX 3050 6GB,
Windows 11 Home, ~5GB RAM rảnh). Xem thêm `report/ShortStop_Report_2.tex`
(Phần III) cho bản trình bày đầy đủ (bản cũ, trước khi đổi bộ 3 benchmark
dưới đây — chưa cập nhật).

**Bộ 3 benchmark đã đổi**: ban đầu là LIBERO/ManiSkill2/RLBench (giống
paper gốc). Sau khi kiểm tra kỹ, **thay ManiSkill2 và RLBench** bằng
**DROID** và **CALVIN** — lý do chi tiết ở mục 2 và
`docs/STAGE7C_ARM_PIPELINE_DESIGN.md`/`docs/RLBENCH_SETUP.md` (giữ nguyên
làm tài liệu tham khảo, không xóa, dù không còn nằm trong kế hoạch chính).

## 1. Thứ tự đề xuất — cập nhật sau khi kiểm tra VRAM thật

**Phát hiện quan trọng nhất** (đổi hẳn thứ tự cũ): $\pi_{0.5}$ (LIBERO) và
$\pi_0$/$\pi_0$-FAST (DROID) là **cùng 1 họ model/backbone** của `openpi` —
tài liệu chính thức ghi rõ **inference cần >8GB VRAM** (ví dụ RTX 4090) cho
**cả hai**, không phải riêng LIBERO. Máy cá nhân (RTX 3050 **6GB**) khả
năng cao **không đủ** cho cả 2 cái này. MDT (CALVIN) thì khác hẳn — chỉ
**40–122M tham số**, nhỏ hơn $\pi_0$ rất nhiều bậc, chạy 6GB VRAM tự tin
được.

1. **CALVIN trước tiên** — **chạy local hoàn toàn được, không cần thuê
   server**. Model nhẹ (MDT), simulator PyBullet nhẹ, phản hồi nhanh nhất
   khi dev/debug shield logic.
2. **LIBERO và DROID cùng nhóm, làm sau** — cả 2 đều cần >8GB VRAM (họ
   $\pi_0$), máy cá nhân nhiều khả năng không đủ cho riêng bước
   **inference thật**. Xem mục 1b cho hướng Docker + thuê server cụ thể.
   Trong nhóm này làm LIBERO trước DROID (sim-eval của LIBERO turnkey hơn,
   DROID's hạ tầng sim-eval còn mới — xem mục 2).

**Vì sao đổi thứ tự so với bản trước**: bản trước ưu tiên theo "độ trưởng
thành simulator", giả định ngầm là LIBERO/DROID chạy local được vì đã có
checkpoint sẵn — **sai**, vì chưa kiểm tra VRAM thật lúc đó. Giờ ưu tiên
theo **cái nào chạy được ngay trên máy hiện có, không cần thuê gì** trước.

## 1b. LIBERO + DROID: hướng Docker + thuê server (do cùng vượt 8GB VRAM)

Không cần thuê server ngay từ đầu — chỉ khi tới bước chạy inference thật:

1. **Viết + test code shield (Reach/Certify/Repair) local, không cần GPU**
   — phần này thuần Python/numpy (đã làm suốt Stage 6a/7a/7c bằng cách
   này: mock policy client, synthetic data, verify structure trước).
2. **Build Docker image local** (trong WSL2) — chứa đúng deps (`openpi` +
   LIBERO hoặc DROID) — build được dù GPU không đủ mạnh để CHẠY, vì build
   chỉ cài đặt, không phải inference.
3. **Chỉ khi cần chạy thật** (cần >8GB VRAM) → thuê GPU cloud (Lambda
   Labs/RunPod/AWS Deep Learning AMI — thường có sẵn Docker + NVIDIA
   Container Toolkit), chạy **đúng image đã build local**, không setup lại
   từ đầu (đúng lý do Docker được xếp "reproducibility cao nhất").
4. **Thuê theo giờ, không thuê liên tục** — chỉ trả tiền lúc thật sự chạy
   (lấy số liệu), tắt máy ngay sau đó; phần code/debug (bước 1) không tốn
   tiền vì làm local.

## 2. Từng benchmark

### LIBERO
- **Checkpoint**: có — $\pi_{0.5}$ (Physical Intelligence, flow-matching),
  fine-tune trên LIBERO-Spatial/Object/Goal/10, host qua repo `openpi`.
- **Chunk xác nhận**: có (`examples/libero/main.py` — Actions shape
  `(batch, action_horizon, action_dim)`).
- **VRAM**: tài liệu openpi ghi rõ **inference cần >8GB** (ví dụ RTX 4090)
  — khả năng cao **vượt** RTX 3050 6GB của máy cá nhân. Xem mục 1b.
- Env: robosuite + MuJoCo, offscreen renderer (EGL/OSMesa).

### CALVIN (thay ManiSkill2)
- **Checkpoint**: **MDT** (Multimodal Diffusion Transformer, RSS 2024,
  `github.com/intuitive-robots/mdt_policy`, có pretrained weights) —
  **không dùng GR-1/RoboFlamingo**: chưa verify được chắc 2 cái đó có
  output chunk thật hay chỉ 1 action/lần (tài liệu tìm được mâu thuẫn).
- **Chunk xác nhận**: có — "10 denoising step mỗi 10 rollout step để tạo 1
  action chunk", diffusion-based, cùng họ với $\pi_{0.5}$ và model Stage
  6a của mình.
- **Uy tín**: RSS 2024 (top venue), đang giữ SOTA trên CALVIN benchmark,
  hệ sinh thái theo sau dày (RoboFlamingo, GR-1, 3D Diffuser Actor, Seer,
  Dita... đều so sánh trên CALVIN).
- **VRAM**: model chỉ **40–122M tham số** (tùy variant) — nhỏ hơn $\pi_0$
  rất nhiều bậc, **chạy được local trên 6GB VRAM**, không cần thuê server.
- Env: **PyBullet** — nhẹ hơn nhiều so với ManiSkill2's SAPIEN GPU-parallel,
  hợp máy 6GB VRAM hơn.
- **Cẩn thận riêng cho CALVIN**: baseline "3D Diffuser Actor" trên CALVIN
  lại predict **keypose** (360 keypose/task) — giống RLBench, không phải
  dense chunk. Chỉ dùng MDT, không lấy nhầm checkpoint 3D Diffuser Actor.

### DROID (thay RLBench)
- **Checkpoint**: $\pi_0$/$\pi_0$-FAST DROID, cùng host qua `openpi` như
  LIBERO — tái dùng gần nguyên `shortstop/pi_policy_client.py`, chỉ đổi
  tên config (`pi0_droid` thay `pi05_libero`).
- **Chunk xác nhận**: có (cùng cơ chế Actions `(batch, action_horizon,
  action_dim)` như LIBERO).
- **VRAM**: **cùng vấn đề với LIBERO** — $\pi_0$/$\pi_0$-FAST là cùng họ
  backbone (khác data fine-tune, không phải kiến trúc nhỏ hơn dù tên có
  "FAST" — FAST chỉ là cách tokenize action nhanh hơn) — inference cần
  >8GB VRAM. Xem mục 1b.
- **Uy tín**: RSS 2024, 1 phần của Open X-Embodiment (corpus pretrain
  cross-embodiment dùng nhiều nhất hiện nay — OpenVLA, Octo đều dùng).
  Lưu ý trung thực: OpenVLA thử trộn DROID 10% nhưng action accuracy thấp,
  phải bỏ khỏi 1/3 cuối training — DROID khó/đa dạng, không phải kém uy
  tín, chỉ là thử thách hơn để fit.
- **Env — điểm yếu thật của candidate này**: DROID gốc là dataset
  real-world, **không có simulator chính thức turnkey** như LIBERO/CALVIN.
  Có hạ tầng sim-eval đang phát triển (`droid-dataset/droid_policy_learning`,
  `arhanjain/sim-evals`, "DROIDSim", "RoboArena") nhưng chưa trưởng thành/
  chuẩn hóa bằng 2 cái trên — cần đọc kỹ trước khi tin tưởng dùng để so
  sánh số liệu.

## 3. Cần double-check khi setup thực tế

Link/checkpoint/license có thể đã đổi theo thời gian — luôn vào đúng repo
(`openpi`, `mees/calvin`, `intuitive-robots/mdt_policy`,
`droid-dataset/droid_policy_learning`) xác nhận lại trước khi setup: format
checkpoint, version dependency, và điều khoản sử dụng (đặc biệt $\pi_0$
host qua S3 của Physical Intelligence).

## 4. Hạ tầng Linux (áp dụng chung)

Cả 3 đều bọc quanh physics/rendering engine, official install nhắm Ubuntu.
Train diffusion/flow policy cũng cần CUDA thật. LIBERO (MuJoCo) và CALVIN
(PyBullet) đều là simulator CPU-friendly hơn ManiSkill2's SAPIEN
GPU-parallel (đã bỏ khỏi kế hoạch) — ít lo về driver Vulkan hơn.

**4 hướng đã so sánh**:

| Hướng | GPU/CUDA training | Công sức setup | Reproducibility |
|---|---|---|---|
| 1. Dual Boot | Tốt nhất (native) | Cao | Trung bình |
| 2. VM chia RAM (VirtualBox/VMware) | Kém/không có | Thấp | Trung bình |
| 3. Docker (Desktop, WSL2 backend) | Tốt | Trung bình | Cao nhất |
| 4. WSL2 trực tiếp | Tốt | Thấp nhất | Trung bình |

**Đề xuất theo giai đoạn** (máy cá nhân, cập nhật theo thứ tự mục 1/1b):

1. **WSL2 trước tiên** để thử nhanh — Ubuntu 22.04, có CUDA, không cần
   reboot. Đủ cho **CALVIN** (nhẹ, local hoàn toàn, không cần server) và
   cho bước viết/test code (không cần GPU) của LIBERO/DROID.
2. Nếu WSL2 đủ ổn cho CALVIN — dừng ở đây cho benchmark này, không cần
   dual-boot/thuê gì.
3. Với LIBERO/DROID (cần >8GB VRAM): **không cần dual-boot** để giải quyết
   VRAM — dual-boot chỉ đổi driver/hiệu năng native, không tăng VRAM vật
   lý của card 6GB. Thuê cloud GPU (mục 1b) là hướng đúng cho riêng vấn
   đề này.
4. Docker dùng để đóng gói môi trường reproducible **sau khi** đã có Linux
   — không phải giải pháp thay cho "có Linux", nhưng **là** cách chuyển
   từ local (WSL2) sang server thuê mà không setup lại (mục 1b).

**Bỏ qua** VM chia RAM kiểu VirtualBox/VMware cho mục đích train GPU — chỉ
hợp để đọc code/test phần không cần GPU.
