# Reimplementation: Coupling Experts and Routers in Mixture-of-Experts via an Auxiliary Loss (ICLR 2026)

Dự án tái hiện đầy đủ bài báo ICLR 2026 và thiết lập hệ sinh thái **đối chứng công bằng (Fair Benchmark)** giữa 4 kiến trúc Mixture-of-Experts (MoE) hàng đầu, tối ưu hóa đặc biệt cho **Kaggle (2x GPU NVIDIA T4, DistributedDataParallel - DDP)**, đồng bộ trực tiếp với **Weights & Biases (WandB)** và hỗ trợ **Chained Save Version** liên tục vượt qua giới hạn 12 tiếng.

---

## 1. Bốn Kiến Trúc Đối Chứng Hoàn Toàn Công Bằng (Fair Benchmark)

Tất cả 4 mô hình đều được chuẩn hóa nghiêm ngặt trên cùng một mặt bằng: **110.29 Triệu tham số ($P_{\text{total}}$)**, **~35.4 Triệu tham số kích hoạt / token ($P_{\text{active}}$)**, 8 layers, $d=512, D=256$, context length 512, tối ưu AdamW $(\beta_1=0.9, \beta_2=0.95, \text{wd}=0.1)$, Cosine LR ($4\text{e-}4 \to 4\text{e-}5$), stream cùng tập dữ liệu `dolma-v1.5-sample`:

| Model | Kiến trúc Routing & Coupling | Tham số & Sparsity | Đặc trưng toán học | Paper tham chiếu |
| :--- | :--- | :--- | :--- | :--- |
| **1. MoE + ERC Loss** *(Ours, ICLR 2026)* | Linear Router $R$ + SwiGLU + ERC Loss | 16 experts, Top-2 (12.5%) | Auxiliary loss $O(n^2)$ với Bounded Noise Perturbation $\epsilon_i \le \frac{\min_{j \neq i} \|R_i - R_j\|}{2\|R_i\|}$ (Eq. 3 & 4) | `paper/Coupling Experts...-26.pdf` |
| **2. Vanilla MoE** *(Switch Transformer)* | Linear Router $R$ + SwiGLU | 16 experts, Top-2 (12.5%) | Router Softmax tiêu chuẩn với Load Balancing Loss $\alpha N \sum f_i P_i$ ($\alpha=0.01$) | `paper/Switch Transformers...pdf` |
| **3. AoE** *(Lv et al., ICML 2025)* | Không Router, phân rã $W_g \to W_{\text{down}} W_{\text{up}}$ | 16 experts, Top-2 ($r=171$) | Tự định tuyến qua chuẩn kích hoạt $\|x \hat{W}_{\text{down}}\|_2$ và Softmax Top-K (Algorithm 2 & Eq. 6) | `paper/2647_Autonomy_of_Experts...pdf` |
| **4. DeepSeek-MoE** *(Dai et al., 2024)* | 1 Shared Expert + 15 Routed Experts | 1 Shared + 1 Routed (Top-2) | Tách biệt tri thức chung (Shared Expert luôn kích hoạt) và tri thức chuyên biệt (Eq. 12) | `paper/DeepSeekMoE.pdf` |

---

## 2. Cấu trúc thư mục chuẩn hóa (Clean Architecture)

```text
coupling_e_a_r/
├── utils/                           # Các hàm loss, optimizer và streaming dataset
│   ├── erc_loss.py                  # Thuật toán ERC Loss (Bounded noise perturbation, Eq. 3 & 4, Specialization ratio)
│   ├── optimizer.py                 # AdamW optimizer (wd=0.1 không decay 1D norm/bias) & Cosine scheduler (min_lr=4e-5)
│   ├── dataset.py                   # Zero-disk, fault-tolerant Dolma v1.5 HTTP streaming dataset (Gzip direct stream)
│   └── __init__.py                  # Exported functions & classes
├── models/                          # 4 mô hình MoE đối chứng tách biệt hoàn toàn
│   ├── config.py                    # Cấu hình chuẩn hóa (8 layers, d=512, D=256, 16 experts, 110.29M params)
│   ├── moe_erc.py                   # Model 1: OLMoE monkey-patched với ERC Auxiliary Loss
│   ├── moe_vanilla.py               # Model 2: Vanilla MoE (Switch Transformer baseline)
│   ├── aoe.py                       # Model 3: Autonomy-of-Experts (Lv et al., 2025, r=171, Algorithm 2)
│   ├── deepseek_moe.py              # Model 4: DeepSeek-MoE (1 Shared Expert + 15 Routed Experts, Top-1)
│   └── __init__.py                  # Exported model builders
├── entrypoint/                      # Điểm thực thi huấn luyện trên Kaggle & Notebook
│   ├── train.py                     # Unified DDP training script 2x T4 hỗ trợ 4 mô hình + WandB + Auto-Resume
│   └── kaggle_train.ipynb           # Jupyter Notebook hoàn chỉnh với 7 cells tự động chạy trên Kaggle
├── paper/                           # 4 bài báo gốc tham chiếu toán học
│   ├── Coupling Experts and Routers in Mixture-of-Experts via an Auxiliary Loss -26.pdf
│   ├── 2647_Autonomy_of_Experts_Model.pdf
│   ├── DeepSeekMoE.pdf
│   └── Switch Transformers Scaling to Trillion Parameter Models.pdf
├── .gitignore                       # Bộ lọc chuẩn (loại bỏ checkpoints, logs, caches, venv)
└── README.md                        # Tài liệu hướng dẫn chi tiết
```

---

## 3. Cơ chế Checkpoint & Bảo hiểm khi Chạy trên Kaggle (Save Version)

Script [train.py](file:///d:/researcher_engineer/paper_moe/coupling_experts_and_routers/coupling_e_a_r/entrypoint/train.py) được trang bị cơ chế **3 tầng bảo hiểm**:

1. **Lưu định kỳ tiết kiệm đĩa (`checkpoint_latest.pt`):**
   * Cứ mỗi 500 steps, script ghi đè file `checkpoint_latest.pt` (chứa toàn bộ `model`, `optimizer`, `scheduler`, `scaler`, `step`, và `history`).
   * Mỗi model chỉ chiếm ~1.5GB (tổng 4 model ~6GB), hoàn toàn an toàn trong giới hạn **20GB** của Kaggle.
2. **Bảo hiểm ngắt container (`SIGTERM / SIGINT` Handler):**
   * Nếu phiên chạy chạm ngưỡng 12 tiếng của Kaggle hoặc người dùng bấm cancel, script lập tức bắt tín hiệu khẩn cấp, **lưu ngay step hiện tại vào `checkpoint_latest.pt`** trước khi container tắt.
3. **Lưu hoàn chỉnh sau khi hoàn tất (`Final Artifacts`):**
   * `checkpoint_final.pt`: Checkpoint đầy đủ cho resume.
   * `model_final.pt`: File trọng số PyTorch thuần (`state_dict()`) phục vụ inference.
   * `save_pretrained(output_dir)`: File cấu hình Hugging Face safetensors + `tokenizer` để load bằng `transformers`.
   * `training_history.json`: Lưu trữ loss, ERC loss, learning rate theo từng step.

---

## 4. Hướng dẫn Chạy trên Kaggle (2x NVIDIA T4)

### Bước 1: Thiết lập Kaggle Secrets (Để chạy ngầm không treo)
1. Trên giao diện Kaggle Notebook: Vào menu **`Add-ons`** $\to$ **`Secrets`**.
2. Thêm secret:
   * **Label:** `WANDB_API_KEY`
   * **Value:** Dán API key từ [https://wandb.ai/authorize](https://wandb.ai/authorize)
3. Tích chọn đính kèm secret vào notebook.

### Bước 2: Bật tài nguyên
* **Accelerator:** `GPU T4 x2`
* **Internet:** `Internet on`

### Bước 3: Huấn luyện 4 mô hình
Mở file [kaggle_train.ipynb](file:///d:/researcher_engineer/paper_moe/coupling_experts_and_routers/coupling_e_a_r/entrypoint/kaggle_train.ipynb) hoặc chạy trực tiếp lệnh:

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

### Bước 4: Chạy vượt giới hạn 12 tiếng qua Chained Versions (Nếu cần train dài hơn)
1. Khi **Version 1** chạy xong, mở notebook ở chế độ Edit.
2. Tại panel bên phải mục **Input** $\to$ bấm **Add Input** $\to$ chọn **Your Work** $\to$ chọn **Version 1**.
3. Thêm cờ `--resume` trỏ vào checkpoint của Version 1:
   ```bash
   --resume /kaggle/input/<tên-notebook>/checkpoints_erc/checkpoint_latest.pt --max_steps 8000
   ```
4. Bấm **Save Version (Version 2)**. Quá trình train sẽ tự động load tiếp từ step 4.001 đến 8.000!

---

## 5. Ước lượng Thời gian & Báo cáo Trực quan

* **Tốc độ trung bình:** ~20.000 tokens/s trên 2 GPU T4.
* **Thời gian mỗi model (4.000 steps ~ 131M tokens):** ~1.8 đến 2.4 giờ.
* **Tổng thời gian cả 4 model:** **~ 7.5 đến 8.5 giờ** (nằm trọn vẹn trong 1 lần Save Version 12 giờ của Kaggle).
* **Theo dõi trực tiếp:** Mở dashboard tại `https://wandb.ai/<user>/moe-erc-iclr2026` để theo dõi các đường cong loss thời gian thực:
  * `train/total_loss`: Cross-entropy task loss.
  * `train/erc_loss`: ERC auxiliary loss ($O(n^2)$).
  * `train/specialization_ratio`: Mức độ phân tách chuyên môn hóa giữa Expert và Router ($\frac{\text{diag}(M)}{\text{off-diag}(M)}$).
  * `train/throughput_tokens_per_sec`: Tốc độ xử lý token/giây trên 2 GPU T4.
* **Biểu đồ tổng kết:** Sau khi cả 4 mô hình chạy xong, Cell 7 trong [kaggle_train.ipynb](file:///d:/researcher_engineer/paper_moe/coupling_experts_and_routers/coupling_e_a_r/entrypoint/kaggle_train.ipynb) tự động vẽ và xuất file ảnh chất lượng cao `/kaggle/working/fair_comparison_4_models.png`.
