# Benchmarking Expert-Router Coupling (ERC) in MoE

Kho mã nguồn tinh gọn tái hiện và so sánh thực nghiệm 4 kiến trúc Mixture-of-Experts (MoE) dựa trên bài báo:
> **Coupling Experts and Routers in Mixture-of-Experts via an Auxiliary Loss** (*ICLR 2026*)  
> *Ang Lv, Jin Ma, Yiyuan Ma, Siyuan Qiao*

---

## 1. Vấn đề & Ý tưởng cốt lõi

* **Vấn đề (Decoupling):** Trong MoE truyền thống (Switch Transformer), Router phân bổ token mà không biết liệu Expert nhận token đó có phản ứng kích hoạt mạnh nhất hay không.
* **Giải pháp (ERC Loss):** Thêm một hàm mất mát phụ trợ $\mathcal{L}_{\text{ERC}}$ trong lúc huấn luyện để ép ma trận kích hoạt giữa Router vectors và Expert weights có đường chéo trội.
* **Mục tiêu so sánh:** Đánh giá tính hiệu quả giữa **ERC Loss** (khớp Router-Expert với zero-inference overhead) và **AoE - Autonomy-of-Experts** (loại bỏ Router, định tuyến trực tiếp bằng phân rã ma trận).

---

## 2. 4 Kiến trúc trong Benchmark

Cả 4 mô hình đều được chuẩn hóa cùng cấu hình: **110M tham số tổng**, ~35.4M active parameters, 8 layers ($d=512, D=256$), dữ liệu streaming từ Dolma v1.5.

| Mô hình | Cơ chế định tuyến | Đặc trưng kiến trúc | Paper gốc |
| :--- | :--- | :--- | :--- |
| **1. ERC MoE** | Router Tuyến tính + ERC Loss | Khớp Router-Expert qua hàm phạt $\mathcal{L}_{\text{ERC}}$ | Lv et al. (ICLR 2026) |
| **2. Vanilla MoE** | Softmax Top-K Router | Switch MoE chuẩn mực, không có ERC Loss | Fedus et al. (JMLR 2022) |
| **3. AoE** | Tự định tuyến (Routerless) | Tính chuẩn kích hoạt $\|x \hat{W}_{\text{down}}\|_2$ | Lv et al. (ICML 2025) |
| **4. DeepSeek-MoE** | Shared + Routed Experts | 1 Shared Expert + 15 Fine-grained Routed Experts | Dai et al. (2024) |

---

## 3. Kết quả Thực nghiệm

Dưới đây là kết quả benchmark thực tế từ lượt chạy đối chứng trên cùng tài nguyên phần cứng (xuất từ WandB):

| Model | Wikitext PPL $\downarrow$ | AVG Acc (%) | COPA | SocialIQa | C-QA | Throughput (tok/s) $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **VANILLA** | **748.03** | 10.79 | 50.00 | 35.82 | 22.03 | **55.35** |
| **DEEPSEEK** | 789.91 | 10.59 | 49.00 | 35.21 | 21.70 | 12.12 |
| **ERC** | 800.60 | 10.21 | 45.00 | 35.36 | 21.79 | **49.90** |
| **AOE** | 851.07 | 10.25 | 45.00 | 35.57 | 21.95 | 11.06 |

*(Các benchmark phức tạp khác như MMLU, HellaSwag, ARC-C đều ở mức 0 do quy mô early-train).*

### Nhận xét & Đánh giá nhanh:
1. **Thông lượng huấn luyện (Throughput):**
   * **ERC** đạt **$49.90\text{ tok/s}$**, giữ được $\approx 90\%$ tốc độ của **Vanilla MoE** ($55.35\text{ tok/s}$).
   * **AoE** ($11.06\text{ tok/s}$) và **DeepSeek** ($12.12\text{ tok/s}$) chậm hơn **~4.5 đến 5 lần**. Điều này xác nhận đúng kết luận bài báo: *AoE quá tốn kém chi phí tính toán khi scale, trong khi ERC duy trì throughput gần như Vanilla*.
2. **PPL & Khả năng học ở giai đoạn đầu:**
   * Vanilla MoE giảm loss nhanh nhất ở giai đoạn đầu do không phải chịu thêm hàm ràng buộc phụ nào.
   * ERC có hàm phạt chuyên môn hóa $\mathcal{L}_{\text{ERC}}$ đóng vai trò như một regularizer. Cần huấn luyện đủ bước (hàng triệu/tỷ tokens) để lợi ích chuyên môn hóa của ERC vượt qua Vanilla MoE như bài báo chứng minh.
   * So sánh trực tiếp giữa hai giải pháp chống decoupling: **ERC vượt trội hơn hẳn AoE cả về tốc độ lẫn Perplexity** (800.6 vs 851.07).

---

## 4. Cấu trúc Dự án

```text
├── models/             # Định nghĩa 4 kiến trúc: ERC, Vanilla, AoE, DeepSeek-MoE
│   ├── config.py       # Cấu hình chuẩn hóa (110M params)
│   ├── moe_erc.py      # OLMoE tích hợp ERC Loss hook
│   ├── moe_vanilla.py  # Switch MoE chuẩn
│   ├── aoe.py          # Autonomy-of-Experts
│   └── deepseek_moe.py # Shared + Routed Experts
├── utils/
│   ├── erc_loss.py     # Triển khai thuật toán ERC Loss & Bounded Perturbation
│   ├── dataset.py      # Streaming trực tiếp Dolma v1.5 qua HTTP
│   └── optimizer.py    # AdamW & Cosine LR Scheduler
├── entrypoint/
│   ├── train.py        # Script huấn luyện phân tán DDP / CLI
│   └── kaggle_train.ipynb # Notebook chạy trực tiếp trên Kaggle GPU (T4 x2)
└── requirements.txt
```

---

## 5. Hướng dẫn Chạy Nhanh

### Cài đặt
```bash
pip install -r requirements.txt
```

### Huấn luyện (CLI / DDP)
Chạy huấn luyện một trong các mô hình bằng lệnh:
```bash
torchrun --nproc_per_node=2 entrypoint/train.py \
    --model_type erc \
    --batch_size 8 \
    --grad_accum_steps 4 \
    --lr 4e-4 \
    --max_steps 4000 \
    --output_dir ./checkpoints_erc \
    --use_wandb
```
> Thay `--model_type` bằng `erc`, `vanilla`, `aoe`, hoặc `deepseek`.

### Chạy trên Kaggle
Mở và thực thi trực tiếp [entrypoint/kaggle_train.ipynb](file:///d:/researcher_engineer/paper_moe/coupling_experts_and_routers/coupling_e_a_r/entrypoint/kaggle_train.ipynb) trên môi trường Kaggle GPU (2x T4).

---

## 6. Tham khảo
* **ERC Loss:** *Coupling Experts and Routers in Mixture-of-Experts via an Auxiliary Loss* (ICLR 2026).
* **AoE:** *Autonomy-of-Experts Models* (ICML 2025).
* **DeepSeekMoE:** *DeepSeekMoE: Towards Ultimate Expert Specialization in MoE* (Dai et al., 2024).
* **Switch Transformers:** *Scaling to Trillion Parameter Models with Simple and Efficient Sparsity* (Fedus et al., 2022).
