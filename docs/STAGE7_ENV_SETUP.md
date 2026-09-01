# Stage 7a–7c: thứ tự setup 3 benchmark thật + ghi chú

Ghi chú thực hành cho khi bắt đầu Stage 7a/7b/7c (LIBERO/ManiSkill2/RLBench),
dựa trên kết luận đã bàn khi đánh giá checkpoint public + hạ tầng cho máy cá
nhân (RTX 3050 6GB, Windows 11 Home, ~5GB RAM rảnh). Xem thêm
`report/ShortStop_Report_2.tex` (Phần III) cho bản trình bày đầy đủ.

## 1. Thứ tự đề xuất

1. **LIBERO trước tiên** — có checkpoint public đúng kiểu generative chunk
   policy ($\pi_0$), chỉ cần inference, không phải train lại.
2. **RLBench thứ hai** — không có checkpoint đúng kiểu sẵn qua PerAct/RVT,
   nhưng có "Mini Diffuser" (nhẹ, có checkpoint public) khớp đúng khung.
3. **ManiSkill2/3 sau cùng, hoãn tới khi có cloud GPU** — chưa có checkpoint
   sẵn, cần tự train (1–3 ngày A100), không hợp lý train trên GPU 6GB laptop.

**Vì sao thứ tự này**: ưu tiên theo (a) có checkpoint public đúng kiểu
generative-chunk-policy hay không (đỡ phải train), và (b) mức GPU/compute
máy cá nhân đáp ứng được tới đâu — không theo thứ tự khó/dễ của bản thân
benchmark.

## 2. Từng benchmark

### LIBERO
- **Checkpoint**: có — $\pi_0$ (Physical Intelligence, flow-matching), đã
  fine-tune trên LIBERO-Spatial/Object/Goal/10, host qua repo `openpi`.
- **Vì sao khớp**: $\pi_0$ đúng kiểu flow-matching sinh action chunk — khớp
  thẳng giả định $\pi_\theta$ mà ShortStop shield cần, không phải train lại.
- Env: robosuite + MuJoCo, offscreen renderer (EGL/OSMesa).

### ManiSkill2/3
- **Checkpoint**: chỉ có code + demo data (4M+ frame), **chưa** có checkpoint
  sẵn để tải. Official repo có script train BC/Diffusion Policy/ACT.
- Hướng khả thi: tự train Diffusion Policy bằng script + demo chính thức
  (1–3 ngày trên 1 A100), hoặc thử adapt checkpoint ngoài (Octo, RT-X,
  RDT-1B) nếu action space khớp.
- Env: SAPIEN, thế mạnh chính là GPU-parallel simulation qua Vulkan + driver
  NVIDIA Linux — không có tương đương ổn định trên Windows.

### RLBench
- **Checkpoint**: có, nhưng **sai kiểu policy** — PerAct (18 task) và
  RVT/RVT-2 (20 task) có checkpoint chính thức, nhưng cả hai là transformer
  dự đoán 1 keyframe action, không phải generative chunk policy.
- **Thay bằng**: "Mini Diffuser" (arXiv:2505.09430,
  github.com/utomm/mini-diffuse-actor) — diffusion policy multitask cho
  RLBench-18, nhẹ, có checkpoint public, đúng khung generative-chunk-policy.
- Env: CoppeliaSim qua PyRep — có build Windows nhưng toàn bộ
  tooling/community giả định Ubuntu (đặc biệt cho training headless).

## 3. Cần double-check khi setup thực tế

Link/checkpoint/license có thể đã đổi theo thời gian — luôn vào đúng repo
(`openpi`, ManiSkill, `peract/peract`, `NVlabs/RVT`,
`utomm/mini-diffuse-actor`) xác nhận lại trước khi setup: format checkpoint,
version dependency, và điều khoản sử dụng (đặc biệt $\pi_0$ host qua S3 của
Physical Intelligence).

## 4. Hạ tầng Linux (áp dụng chung cho cả 3 benchmark)

Cả 3 đều bọc quanh physics/rendering engine biên dịch sẵn, official install
đều nhắm Ubuntu. Train diffusion/flow policy cũng cần CUDA thật.

**4 hướng đã so sánh**:

| Hướng | GPU/CUDA training | Công sức setup | Reproducibility |
|---|---|---|---|
| 1. Dual Boot | Tốt nhất (native) | Cao | Trung bình |
| 2. VM chia RAM (VirtualBox/VMware) | Kém/không có | Thấp | Trung bình |
| 3. Docker (Desktop, WSL2 backend) | Tốt | Trung bình | Cao nhất |
| 4. WSL2 trực tiếp | Tốt | Thấp nhất | Trung bình |

**Đề xuất theo giai đoạn** (máy cá nhân):

1. **WSL2 trước tiên** để thử nhanh — Ubuntu 22.04, có CUDA, không cần
   reboot. Test riêng từng benchmark: LIBERO/RLBench nhiều khả năng cài
   được ngay; ManiSkill2 (GPU-parallel qua Vulkan) cần kiểm tra kỹ trong
   WSL2 trước khi tin tưởng.
2. Nếu WSL2 đủ ổn (đặc biệt cho LIBERO $\pi_0$) — có thể dừng ở đây, không
   cần dual-boot.
3. Nếu vướng driver/rendering (nhiều khả năng nhất ở ManiSkill2's Vulkan),
   hoặc muốn hiệu năng full cho train dài hạn: dual boot sang Ubuntu.
4. Nếu máy cá nhân không đủ GPU mạnh cho training thật (đặc biệt
   ManiSkill2): thuê cloud GPU instance Ubuntu (AWS/GCP/Lambda Labs...),
   trả phí theo giờ.
5. Docker dùng để đóng gói môi trường reproducible **sau khi** đã có Linux
   (WSL2, dual-boot, hoặc cloud instance) — không phải giải pháp thay cho
   "có Linux".

**Bỏ qua** VM chia RAM kiểu VirtualBox/VMware cho mục đích train GPU — chỉ
hợp để đọc code/test phần không cần GPU.
