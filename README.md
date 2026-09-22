# Reimplementation: Coupling Experts and Routers in Mixture-of-Experts via an Auxiliary Loss (ICLR 2026)

Dự án tái hiện đầy đủ bài báo ICLR 2026 và thiết lập hệ thống **đối chứng công bằng (Fair Benchmark)** giữa 4 kiến trúc Mixture-of-Experts (MoE) hàng đầu hiện nay, tối ưu hóa cho **Kaggle (2x GPU NVIDIA T4, DistributedDataParallel - DDP)** và theo dõi tiến trình qua **Weights & Biases (WandB)**.

---

## 1. Bốn Kiến Trúc Đối Chứng Hoàn Toàn Công Bằng (Fair Benchmark)

Tất cả 4 mô hình đều được chuẩn hóa cùng: **110.29 Triệu tham số (Active ~35.4M)**, 8 layers, $d=512, D=256$, context length 512, tối ưu AdamW $(\beta_1=0.9, \beta_2=0.95, \text{wd}=0.1)$, Cosine LR ($4\text{e-}4 \to 4\text{e-}5$), stream cùng tập dữ liệu `dolma-v1.5-sample`:

| Model | Kiến trúc Routing & Coupling | Tham số & Tải | Đặc trưng toán học |
| :--- | :--- | :--- | :--- |
| **1. MoE + ERC Loss** *(Ours, ICLR 2026)* | Linear Router $R$ + SwiGLU + ERC Loss | 16 experts, Top-2 | Áp dụng hàm auxiliary loss $O(n^2)$ với Bounded Noise Perturbation (Eq. 3 & 4) |
| **2. Vanilla MoE** *(Switch Transformer)* | Linear Router $R$ + SwiGLU | 16 experts, Top-2 | Router softmax tiêu chuẩn, chỉ dùng Load Balancing Loss $\mathcal{L}_{\text{LB}}$ |
| **3. AoE** *(Lv et al., ICML 2025)* | Không Router, phân rã $W_g \to W_{\text{down}} W_{\text{up}}$ | 16 experts, Top-2 ($r=171$) | Tự định tuyến qua chuẩn kích hoạt $\|x \hat{W}_{\text{down}}\|_2$ và Softmax Top-K (Algorithm 2 & Eq. 6) |
| **4. DeepSeek-MoE** *(Dai et al., 2024)* | 1 Shared Expert + 15 Routed Experts | 1 Shared + 1 Routed (Top-2) | Tách biệt tri thức chung (Shared Expert luôn kích hoạt) và tri thức chuyên biệt |

---

## 2. Cấu trúc thư mục chuẩn hóa (Clean Architecture)

```text
coupling_e_a_r/
├── utils/                           # Các hàm loss, optimizer và streaming dataset
│   ├── erc_loss.py                  # Thuật toán ERC Loss (Bounded noise perturbation, Eq. 3 & 4, Specialization ratio)
│   ├── optimizer.py                 # AdamW optimizer (wd=0.1 không decay 1D norm/bias) & Cosine scheduler (min_lr=4e-5)
│   ├── dataset.py                   # Zero-disk, fault-tolerant Dolma v1.5 HTTP streaming dataset
│   └── __init__.py                  # Exported functions & classes
├── models/                          # 4 mô hình MoE đối chứng tách biệt hoàn toàn
│   ├── config.py                    # Cấu hình chuẩn hóa (8 layers, d=512, D=256, 16 experts, 110M params)
│   ├── moe_erc.py                   # Model 1: OLMoE monkey-patched với ERC Auxiliary Loss
│   ├── moe_vanilla.py               # Model 2: Vanilla MoE (Switch Transformer baseline)
│   ├── aoe.py                       # Model 3: Autonomy-of-Experts (Lv et al., 2025, r=171, Algorithm 2)
│   ├── deepseek_moe.py              # Model 4: DeepSeek-MoE (1 Shared Expert + 15 Routed Experts, Top-1)
│   └── __init__.py                  # Exported model builders
├── entrypoint/                      # Điểm thực thi huấn luyện trên Kaggle & Notebook
│   ├── train.py                     # Script DDP 2x T4 hỗ trợ cả 4 mô hình (--model_type {erc, vanilla, aoe, deepseek}) + WandB
│   └── kaggle_train.ipynb           # Jupyter Notebook chạy trực tiếp trên môi trường Kaggle
├── README.md                        # Tài liệu hướng dẫn chi tiết
├── 17. Coupling Experts and Routers in Mixture-of-Experts via an Auxiliary Loss -26.pdf # Paper gốc ICLR 2026
└── 2647_Autonomy_of_Experts_Model.pdf # Paper gốc AoE ICML 2025
```

---

## 3. Cách chạy 4 mô hình trên Kaggle (2x T4 GPU)

Chỉ cần mở file [kaggle_train.ipynb](file:///d:/researcher_engineer/paper_moe/coupling_experts_and_routers/coupling_e_a_r/entrypoint/kaggle_train.ipynb) trên Kaggle, đăng nhập WandB và chạy các cell tương ứng:

```bash
# Model 1: MoE + ERC Loss (ICLR 2026)
torchrun --nproc_per_node=2 entrypoint/train.py \
    --model_type erc \
    --dolma_num_shards 30 \
    --seq_len 512 \
    --batch_size 8 \
    --grad_accum_steps 4 \
    --lr 4e-4 \
    --min_lr 4e-5 \
    --weight_decay 0.1 \
    --beta1 0.9 \
    --beta2 0.95 \
    --alpha 1.0 \
    --erc_weight 1.0 \
    --max_steps 4000 \
    --save_every 500 \
    --use_wandb \
    --wandb_project moe-erc-iclr2026 \
    --wandb_run_name erc_moe_110m \
    --output_dir /kaggle/working/checkpoints_erc

# Model 2: Vanilla MoE (Switch Transformer)
torchrun --nproc_per_node=2 entrypoint/train.py \
    --model_type vanilla \
    --use_wandb \
    --wandb_project moe-erc-iclr2026 \
    --wandb_run_name vanilla_moe_110m \
    --output_dir /kaggle/working/checkpoints_vanilla

# Model 3: AoE (Autonomy-of-Experts)
torchrun --nproc_per_node=2 entrypoint/train.py \
    --model_type aoe \
    --use_wandb \
    --wandb_project moe-erc-iclr2026 \
    --wandb_run_name aoe_110m \
    --output_dir /kaggle/working/checkpoints_aoe

# Model 4: DeepSeek-MoE
torchrun --nproc_per_node=2 entrypoint/train.py \
    --model_type deepseek \
    --use_wandb \
    --wandb_project moe-erc-iclr2026 \
    --wandb_run_name deepseek_moe_110m \
    --output_dir /kaggle/working/checkpoints_deepseek
```

---

## 4. Theo dõi trực tiếp qua Weights & Biases (WandB)

Toàn bộ quá trình huấn luyện tự động ghi nhận các metric thời gian thực lên Dashboard của WandB:
- `train/total_loss`: Cross-entropy task loss.
- `train/erc_loss`: ERC auxiliary loss khớp router và expert ($O(n^2)$).
- `train/learning_rate`: Cosine schedule decaying từ $4\times 10^{-4} \to 4\times 10^{-5}$.
- `train/specialization_ratio`: Tỷ lệ chuyên môn hóa đường chéo $\frac{\text{diag}(M)}{\text{off-diag}(M)}$.
- `train/throughput_tokens_per_sec`: Tốc độ xử lý token/giây trên 2 GPU T4.
