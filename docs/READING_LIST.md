# Reading List — các paper được đề xuất trong Phan_Tich_Sau.tex / Report_1.tex

Ghi chú: đây là danh sách tra cứu, không phải bibliography chính thức — tên/venue lấy từ
`ShortStop_Phan_Tich_Sau.tex` (mục Tài liệu tham khảo) và đã verify tồn tại thật cho 6 bài
ưu tiên ở mục A/B (xem hội thoại trước). Các bài còn lại (mục C, nền tảng) lấy từ chính
reference list gốc của paper ShortStop hoặc bibliography tự tổng hợp, chưa fact-check
từng cái — coi là điểm khởi đầu để tự tìm, không phải đã xác nhận 100%.

## A. 3 baseline thực tế (dùng trong Table II của ShortStop) — đọc trước tiên

1. **MPC-Filter** — K. P. Wabersich, M. N. Zeilinger, *"A Predictive Safety Filter for
   Learning-Based Control of Constrained Nonlinear Dynamical Systems,"* Automatica,
   vol. 129, art. 109597, 2021.
   → Vì sao đọc: nguồn baseline `MPC-Filter`; gần nhất với nền tảng MPC/QP bạn đã có.

2. **CBF-Shield (gốc)** — A. D. Ames, X. Xu, J. W. Grizzle, P. Tabuada,
   *"Control Barrier Function Based Quadratic Programs for Safety Critical Systems,"*
   IEEE Trans. Automatic Control, 62(8):3861–3876, 2017.
   → Vì sao đọc: nguồn baseline `CBF-Shield`; định nghĩa CBF-QP gốc.

3. **CBF-Shield (mở rộng sang RL)** — R. Cheng, G. Orosz, R. M. Murray, J. W. Burdick,
   *"End-to-End Safe Reinforcement Learning through Barrier Functions for
   Safety-Critical Continuous Control Tasks,"* AAAI 2019.
   → Vì sao đọc: bổ sung cho #2, ghép CBF-QP vào policy RL.

4. **STL-Monitor** — J. V. Deshmukh, A. Donzé, S. Ghosh, X. Jin, G. Juniwal, S. A. Seshia,
   *"Robust Online Monitoring of Signal Temporal Logic,"* Formal Methods in System
   Design, 2017.
   → Vì sao đọc: nguồn baseline `STL-Monitor`, và nguồn công thức robustness đệ quy
   (Eq. 2) mà ShortStop tái sử dụng.

## B. 3 bài bối cảnh rộng hơn (không benchmark trực tiếp, nhưng cùng hướng đề tài)

5. **SafeDiffuser** — W. Xiao, T.-H. Wang, C. Gan, D. Rus, *"SafeDiffuser: Safe Planning
   with Diffusion Probabilistic Models,"* arXiv:2306.00148 (2023); ICLR 2025.
   https://arxiv.org/abs/2306.00148
   → Nhúng CBF vào chính quá trình denoising — sửa generator, khác cách tiếp cận
   post-hoc của ShortStop.

6. **MPM-STL** — X. Yu, W. Dong, X. Yin, S. Li, *"Model Predictive Monitoring of
   Dynamical Systems for Signal Temporal Logic Specifications,"* arXiv:2209.12493;
   Automatica, 2024.
   https://arxiv.org/abs/2209.12493
   → Tiền lệ gần nhất của ý tưởng "STL trên reachable set thay vì 1 trajectory".

7. **Path-Consistent Filtering** — R. Römer, J. Balletshofer, J. Thumm, M. Pavone,
   A. P. Schoellig, M. Althoff, *"From Demonstrations to Safe Deployment:
   Path-Consistent Safety Filtering for Diffusion Policies,"* arXiv:2511.06385 (2025);
   chấp nhận ICRA 2026.
   https://arxiv.org/abs/2511.06385
   → Công trình đồng thời, cùng mục tiêu "an toàn hậu kiểm cho diffusion policy".

## C. Nền tảng (đọc sau, theo 5 khối kiến thức — không phải baseline, không benchmark)

### Khối 1 — Generative robot policy
- C. Chi et al., *"Diffusion Policy: Visuomotor Policy Learning via Action Diffusion,"* RSS 2023.
- M. Janner, Y. Du, J. B. Tenenbaum, S. Levine, *"Planning with Diffusion for Flexible Behavior Synthesis,"* ICML 2022.
- A. Ajay et al., *"Is Conditional Generative Modeling all you need for Decision-Making?"* ICLR 2023.
- Y. Lipman et al., *"Flow Matching for Generative Modeling,"* ICLR 2023.
- T. Z. Zhao, V. Kumar, S. Levine, C. Finn, *"Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware,"* RSS 2023. (ALOHA — động lực cho action chunking)
- R. Ivanov et al., *"Verisig: Verifying Safety Properties of Hybrid Systems with Neural Network Controllers,"* HSCC 2019.
- H.-D. Tran et al., *"NNV: The Neural Network Verification Tool,"* CAV 2020.
- C. Huang et al., *"ReachNN: Reachability Analysis of Neural-Network Controlled Systems,"* EMSOFT 2019.
- M. Everett, G. Habibi, C. Sun, J. P. How, *"Reachability Analysis of Neural Feedback Loops,"* IEEE Access, 2021.
  (4 bài trên: lý do neural-network reachability không scale tới denoiser 100M+ tham số)

### Khối 2 — Reachability analysis (set propagation)
- M. Althoff, *"An Introduction to CORA 2015,"* ARCH Workshop 2015.
- X. Chen, E. Ábrahám, S. Sankaranarayanan, *"Flow*: An Analyzer for Non-Linear Hybrid Systems,"* CAV 2013.

### Khối 3 — Signal Temporal Logic (STL) & robustness
- O. Maler, D. Nickovic, *"Monitoring Temporal Properties of Continuous Signals,"* FORMATS/FTRTFT 2004.
- A. Donzé, O. Maler, *"Robust Satisfaction of Temporal Logic over Real-Valued Signals,"* FORMATS 2010.
- G. E. Fainekos, G. J. Pappas, *"Robustness of Temporal Logic Specifications for Continuous-Time Signals,"* Theoretical Computer Science, 2009.
- (STL-Monitor gốc — Deshmukh et al. 2017 — đã liệt kê ở mục A.4)

### Khối 4 — CEGIS/CEGAR & falsification
- A. Solar-Lezama et al., *"Combinatorial Sketching for Finite Programs,"* ASPLOS 2006.
- E. Clarke, O. Grumberg, S. Jha, Y. Lu, H. Veith, *"Counterexample-Guided Abstraction Refinement,"* CAV 2000.
- Y. Annpureddy et al., *"S-TaLiRo: A Tool for Temporal Logic Falsification for Hybrid Systems,"* TACAS 2011.
- A. Donzé, *"Breach, a Toolbox for Verification and Parameter Synthesis of Hybrid Systems,"* CAV 2010.
- T. Dreossi et al., *"VerifAI: A Toolkit for the Formal Design and Analysis of AI-based Systems,"* CAV 2019.

### Khối 5 — Predictive safety filters, MPC, CBF (nền tảng gần nhất với kinh nghiệm cá nhân)
- O. Bastani, *"Safe Reinforcement Learning with Nonlinear Dynamics via Model Predictive Shielding,"* ACC 2021.
- A. D. Ames et al., *"Control Barrier Functions: Theory and Applications,"* ECC 2019.
- D. Q. Mayne, J. B. Rawlings, C. V. Rao, P. O. M. Scokaert, *"Constrained Model Predictive Control: Stability and Optimality,"* Automatica, 2000.
- (MPC-Filter gốc — Wabersich & Zeilinger 2021 — và CBF-Shield gốc — Ames 2017 / Cheng 2019 — đã liệt kê ở mục A)

### Khác (nhắc tới trong Sec. II.B của ShortStop, không phải baseline)
- T. Power, R. Soltani-Zarrin, S. Iba, D. Berenson, *"Sampling Constrained Trajectories Using Composable Diffusion Models,"* IROS Workshop 2023.

---

## Thứ tự đọc gợi ý (general → specific, hiểu toàn cảnh trước khi vào chi tiết)

Không đọc tuần tự A→B→C ở trên — thứ tự dưới đây hiệu quả hơn cho người đã có nền
MPC/QP nhưng mới với generative/reachability/STL/CEGIS:

1. **Giai đoạn 0 — Bức tranh toàn cảnh trước (không phải paper ngoài)**: đọc
   `ShortStop_Phan_Tich_Sau.tex` §2 ("Bức tranh toàn cảnh") + Sec.~I của chính paper
   ShortStop. Nắm *câu hỏi* (vì sao cần shield) trước khi nắm *câu trả lời* (shield làm
   bằng gì) — đọc kỹ thuật trước khi có câu hỏi này sẽ trôi tuột.

2. **Giai đoạn 1 — Skim (không đọc sâu) Khối 1**: chỉ abstract của Diffusion Policy
   (Chi et al.) và Flow Matching (Lipman et al.) — đủ hiểu "action chunk" và "sampler
   sinh xác suất" là gì, không cần hiểu toán DDPM/ODE.

3. **Giai đoạn 2 — 3 baseline gốc (mục A), đọc kỹ — chỗ đáng đầu tư nhất**:
   MPC-Filter → CBF-Shield → STL-Monitor, đúng thứ tự này (gần nền tảng cũ nhất trước,
   xa nhất để cuối). Mục tiêu: tự thấy giới hạn thật của từng bài gốc (CBF cần $h(x)$
   thiết kế tay, STL-Monitor chỉ là monitor bị động...) *trước khi* đọc ShortStop giải
   thích lại — để lúc đọc, nó "click" vì bạn đã mang sẵn câu hỏi cụ thể trong đầu.

4. **Giai đoạn 3 — Quay lại đọc Sec.~II + III của chính ShortStop**: giờ đọc Related
   Work/Method sẽ nhanh và "đã" hơn nhiều vì có 3 baseline gốc trong tay để đối chiếu
   trực tiếp từng câu.

5. **Giai đoạn 4 — 3 bài bối cảnh (mục B), đọc sau cùng, không đọc trước**:
   SafeDiffuser/MPM-STL/Path-Consistent chỉ có ý nghĩa khi đã hiểu rõ P-R-C-S của
   ShortStop — đọc sớm sẽ chỉ thấy 3 cách tiếp cận rời rạc, không có khung để so sánh.

6. **Nhóm C (nền tảng)**: không đọc trước — chỉ tra cứu just-in-time khi code
   Phase 1–3 gặp khái niệm cụ thể chưa rõ, đúng tinh thần "ramp-up" ở
   `ShortStop_Research_Plan.tex` §5.

Đánh đổi: giai đoạn 2 tốn thời gian nhất (đọc kỹ thay vì lướt), nhưng đổi lại giai đoạn 3
(đọc lại chính ShortStop) nhanh hơn hẳn và nhớ lâu hơn vì có sẵn khung so sánh cụ thể.
