# Reimplementation & Benchmark: Coupling Experts and Routers in MoE (ICLR 2026)

Dự án tái hiện và thiết lập benchmark đối chứng cho phương pháp **Expert-Router Coupling (ERC) Loss** từ bài báo ICLR 2026:
> **Coupling Experts and Routers in Mixture-of-Experts via an Auxiliary Loss**  
> *Ang Lv, Jin Ma, Yiyuan Ma, Siyuan Qiao (ICLR 2026)*

Kho mã nguồn cung cấp cài đặt hoàn chỉnh bằng **PyTorch** và **Hugging Face Transformers**, so sánh trực tiếp 4 kiến trúc Mixture-of-Experts (MoE) dưới cùng một cấu hình tham số, tập dữ liệu huấn luyện và bộ tối ưu hóa chuẩn mực.

---

## 1. Tổng quan & Phương pháp

### Vấn đề: Hiện tượng Decoupling trong MoE
Trong các mô hình MoE truyền thống (như Switch Transformer), Router (bộ định tuyến) và Experts (các mạng nơ-ron chuyên gia) được huấn luyện độc lập qua hàm mất mát chung. Điều này dẫn đến hiện tượng **mất liên kết (Expert-Router Decoupling)**: Router có thể phân bổ token tới một expert mà bản thân expert đó không có phản ứng kích hoạt mạnh nhất với loại token đó, làm suy giảm hiệu quả chuyên môn hóa.

### Giải pháp: ERC Auxiliary Loss
Bài báo đề xuất hàm mất mát phụ trợ **Expert-Router Coupling (ERC) Loss**:
* Sử dụng proxy token $\tilde{R}_i = R_i \odot \delta_i$ với nhiễu đồng nhất có chặn $\epsilon_i \le \frac{\min_{j \neq i} \|R_i - R_j\|}{2\|R_i\|}$ (Eq. 4).
* Xây dựng ma trận kích hoạt $M \in \mathbb{R}^{n \times n}$ đo lường phản ứng của Expert $j$ đối với đại diện của Router $i$:
  $$M[i, j] = \|\tilde{R}_i W_g^j\|_2$$
* Ép ma trận $M$ có đường chéo trội qua hàm mất mát $O(n^2)$ (Eq. 3), buộc Expert $i$ phản ứng mạnh nhất với Router $i$.
* **Ưu điểm thực tế:** Không làm tăng chi phí tính toán khi suy luận (Zero-inference-overhead), độ phức tạp lúc train chỉ chiếm $< 0.5\%$ FLOPs.

---

## 2. Thiết lập Benchmark Đối chứng (Standardized MoE Benchmark)

Để đảm bảo tính khách quan và khoa học, cả 4 mô hình đều được chuẩn hóa chính xác trên cùng một quy mô tài nguyên:

* **Tổng số tham số ($P_{\text{total}}$):** 110.29 triệu tham số.
* **Tham số kích hoạt mỗi token ($P_{\text{active}}$):** ~35.4 triệu tham số (Top-2 experts, tương đương độ thưa 12.5%).
* **Cấu trúc mạng:** 8 Decoder layers, $d=512$, $D=256$, 8 attention heads, context length 512.
* **Tập dữ liệu:** Streaming trực tiếp từ `dolma-v1.5-sample` (AllenAI) qua HTTP/Gzip, không ghi đĩa tạm.
* **Bộ tối ưu:** AdamW $(\beta_1=0.9, \beta_2=0.95)$, weight decay 0.1 (không decay 1D norm/bias).
* **Lịch trình học (LR):** Cosine scheduler giảm từ $4\times 10^{-4}$ về $4\times 10^{-5}$ (warmup 5% tổng steps).
* **Cân bằng tải:** Load Balancing Loss với hệ số $\alpha = 0.01$ đồng nhất cho các mô hình.

| Mô hình | Cơ chế Định tuyến & Ghép nối | Đặc trưng kiến trúc | Paper gốc |
| :--- | :--- | :--- | :--- |
| **1. ERC MoE** | Linear Router $R$ + SwiGLU + ERC Loss | Khớp Router-Expert qua hàm phạt $O(n^2)$ (Eq. 3 & 4) | Lv et al. (ICLR 2026) |
| **2. Vanilla MoE** | Linear Router $R$ + SwiGLU | Switch MoE tiêu chuẩn với Softmax Top-K Router | Fedus et al. (JMLR 2022) |
| **3. AoE** | Không dùng Router, phân rã $W_g \to W_{\text{down}} W_{\text{up}}$ | Tự định tuyến qua chuẩn kích hoạt $\|x \hat{W}_{\text{down}}\|_2$ ($r=171$) | Lv et al. (ICML 2025) |
| **4. DeepSeek-MoE** | 1 Shared Expert + 15 Routed Experts | Tách biệt tri thức chung (Shared) và chuyên biệt (Top-1 Routed) | Dai et al. (2024) |

---

## 3. Cấu trúc Thư mục

```text
coupling_e_a_r/
├── utils/                           # Các tiện ích bổ trợ
│   ├── erc_loss.py                  # Cài đặt ERC loss, bounded perturbation, specialization metric
│   ├── optimizer.py                 # AdamW optimizer cấu hình 2D/1D decay & Cosine scheduler
│   ├── dataset.py                   # ShardedStreamingDataset đọc trực tiếp Dolma v1.5 qua HTTP
│   └── __init__.py
├── models/                          # Cài đặt 4 kiến trúc MoE
│   ├── config.py                    # Cấu hình chung 110M / 8 layers
│   ├── moe_erc.py                   # Model 1: OLMoE kết hợp ERC Loss hook
│   ├── moe_vanilla.py               # Model 2: Vanilla Switch MoE baseline
│   ├── aoe.py                       # Model 3: Autonomy-of-Experts layer (Algorithm 2)
│   ├── deepseek_moe.py              # Model 4: DeepSeek-MoE block (Shared + Routed)
│   └── __init__.py
├── entrypoint/                      # Điểm thực thi và notebook
│   ├── train.py                     # Script huấn luyện phân tán (DDP), checkpointing, WandB
│   └── kaggle_train.ipynb           # Notebook chạy trọn gói trên Kaggle
├── paper/                           # 4 bài báo khoa học tham chiếu
│   ├── Coupling Experts and Routers in Mixture-of-Experts via an Auxiliary Loss -26.pdf
│   ├── 2647_Autonomy_of_Experts_Model.pdf
│   ├── DeepSeekMoE.pdf
│   └── Switch Transformers Scaling to Trillion Parameter Models.pdf
├── .gitignore
└── README.md
```

---

## 4. Hướng dẫn Huấn luyện

### 4.1. Cài đặt môi trường
Yêu cầu Python >= 3.10 và môi trường PyTorch hỗ trợ CUDA. Cài đặt toàn bộ các gói phụ thuộc cần thiết qua [requirements.txt](file:///d:/researcher_engineer/paper_moe/coupling_experts_and_routers/coupling_e_a_r/requirements.txt):
```bash
pip install -r requirements.txt
```

Hệ thống thư viện được phân bổ rõ ràng theo các nhóm chức năng:
* **Deep Learning Core:** `torch>=2.1.0` (hỗ trợ phân tán DDP và huấn luyện hỗn hợp FP16 qua `torch.amp`).
* **Hugging Face Ecosystem:** `transformers>=4.40.0`, `tokenizers`, `datasets`, `accelerate`, `safetensors`, `huggingface-hub` (mô hình nền causal LM và quản lý trọng số).
* **MoE & Tensor Operations:** `einops>=0.7.0`, `numpy`, `scipy` (thao tác ma trận kích hoạt và tensor định tuyến).
* **Data Streaming & Networking:** `pyarrow`, `fsspec`, `requests`, `urllib3` (hỗ trợ stream dữ liệu trực tiếp từ các shard Dolma v1.5 qua HTTP/Gzip).
* **Tracking & Visualization:** `wandb>=0.16.0`, `matplotlib>=3.8.0`, `tqdm` (ghi nhận metric và vẽ đồ thị đối chứng).

### 4.2. Huấn luyện bằng dòng lệnh (CLI / DDP)
Script `entrypoint/train.py` hỗ trợ huấn luyện phân tán qua `torchrun` hoặc đơn GPU.

```bash
# Huấn luyện mô hình ERC MoE trên 2 GPU:
torchrun --nproc_per_node=2 entrypoint/train.py \
    --model_type erc \
    --dolma_num_shards 30 \
    --seq_len 512 \
    --batch_size 8 \
    --grad_accum_steps 4 \
    --lr 4e-4 \
    --min_lr 4e-5 \
    --weight_decay 0.1 \
    --alpha 1.0 \
    --erc_weight 1.0 \
    --max_steps 4000 \
    --save_every 500 \
    --output_dir ./checkpoints_erc \
    --use_wandb \
    --wandb_project moe-erc-iclr2026 \
    --wandb_run_name erc_moe_110m
```

Để chuyển sang các mô hình đối chứng khác, chỉ cần thay đổi đối số `--model_type`:
* `--model_type vanilla`: Chạy baseline Vanilla Switch MoE.
* `--model_type aoe`: Chạy baseline Autonomy-of-Experts.
* `--model_type deepseek`: Chạy baseline DeepSeek-MoE.

### 4.3. Quản lý Checkpoint & Khôi phục huấn luyện (`--resume`)
* **Lưu định kỳ:** Cứ mỗi `save_every` steps (mặc định 500), checkpoint toàn diện gồm trọng số, optimizer, scheduler và scaler được lưu vào `checkpoint_latest.pt`.
* **Lưu hoàn tất:** Khi kết thúc `max_steps`, mô hình được xuất ra cả dạng trọng số PyTorch (`model_final.pt`) và định dạng Hugging Face (`save_pretrained`).
* **Tiếp tục huấn luyện:** Nếu tiến trình bị gián đoạn hoặc bạn muốn train thêm steps, truyền đường dẫn checkpoint vào `--resume`:
  ```bash
  torchrun --nproc_per_node=2 entrypoint/train.py \
      --model_type erc \
      --resume ./checkpoints_erc/checkpoint_latest.pt \
      --max_steps 8000 \
      --output_dir ./checkpoints_erc
  ```

---

## 5. Chạy trên Kaggle (2x NVIDIA T4)

Notebook [entrypoint/kaggle_train.ipynb](file:///d:/researcher_engineer/paper_moe/coupling_experts_and_routers/coupling_e_a_r/entrypoint/kaggle_train.ipynb) đã được thiết kế sẵn để thực thi trực tiếp trên Kaggle GPU:

1. **Thiết lập Kaggle Notebook:**
   * **Accelerator:** Chọn `GPU T4 x2`.
   * **Internet:** Bật `Internet on`.
   * **WandB Key:** (Tùy chọn) Thêm secret `WANDB_API_KEY` trong menu `Add-ons > Secrets` để đăng nhập tự động.
2. **Kéo mã nguồn:**
   ```python
   !git clone https://github.com/minhduong05/reimplement_coupling_ex_ro.git /kaggle/working/repo
   %cd /kaggle/working/repo
   ```
3. **Thực thi:**
   Chạy tuần tự các cells trong notebook hoặc chọn **Save Version (Save & Run All)** để kernel chạy ngầm trên máy ảo của Kaggle.

---

## 6. Theo dõi Thực nghiệm qua Weights & Biases (WandB)

Khi bật `--use_wandb`, các thông số huấn luyện được đồng bộ thời gian thực:
* `train/total_loss`: Cross-entropy task loss (+ auxiliary losses).
* `train/erc_loss`: Giá trị hàm phạt ERC loss theo step.
* `train/specialization_ratio`: Tỷ lệ chuyên môn hóa $\frac{\text{diag}(M)}{\text{off-diag}(M)}$ đo mức độ phân bổ chính xác giữa Router và Expert.
* `train/throughput_tokens_per_sec`: Tốc độ xử lý token thực tế trên phần cứng.
* `train/learning_rate`: Đường cong học theo lịch trình Cosine decay.

Sau khi hoàn thành cả 4 lượt chạy, Cell cuối cùng trong notebook sẽ tổng hợp các file lịch sử huấn luyện và xuất biểu đồ so sánh `fair_comparison_4_models.png`.

---

## 7. Tài liệu Tham khảo

1. **ERC Loss:** Lv, A., Ma, J., Ma, Y., & Qiao, S. (2026). *Coupling Experts and Routers in Mixture-of-Experts via an Auxiliary Loss*. ICLR 2026.
2. **AoE:** Lv, A., et al. (2025). *Autonomy-of-Experts Models*. ICML 2025 / arXiv:2407.02568.
3. **DeepSeek-MoE:** Dai, D., et al. (2024). *DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models*. arXiv:2401.06066.
4. **Switch Transformer:** Fedus, W., Zoph, B., & Shazeer, N. (2022). *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*. JMLR.
