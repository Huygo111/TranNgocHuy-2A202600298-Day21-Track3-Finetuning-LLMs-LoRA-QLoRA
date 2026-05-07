# Lab 21 - Tổng Quan Và Hướng Dẫn Từng Bước

## 1. Lab này cần làm gì?

Đây là bài lab về **fine-tuning LLM bằng LoRA/QLoRA**. Nhiệm vụ chính là:

1. Chuẩn bị một dataset instruction-tuning theo định dạng Alpaca.
2. Fine-tune cùng một base model bằng **3 cấu hình rank LoRA**:
   - `r=8`
   - `r=16`
   - `r=64`
3. So sánh 3 rank theo 4 nhóm chỉ số:
   - thời gian train
   - mức dùng VRAM
   - eval loss / perplexity
   - chất lượng output thực tế
4. Lưu adapter checkpoint và viết `REPORT.md` tổng kết kết quả.

Nói ngắn gọn: bài lab không chỉ yêu cầu "train được model", mà còn yêu cầu **làm một thí nghiệm so sánh rank** và rút ra kết luận rank nào hợp lý nhất.

## 2. Mục tiêu học tập

Sau khi xong lab, bạn cần thể hiện được rằng mình:

- Biết chuẩn bị dataset Alpaca format dùng cho SFT.
- Biết load model 4-bit để train theo kiểu QLoRA.
- Biết cấu hình LoRA với `r`, `alpha`, `target_modules`.
- Biết train bằng `TRL SFTTrainer`.
- Biết đánh giá model bằng perplexity và qualitative examples.
- Biết phân tích trade-off giữa `r=8`, `r=16`, `r=64`.

## 3. Notebook và tài liệu cần dùng

Repo này có 3 file quan trọng:

- `README.md`
  - Mô tả tổng quan Day 21.
  - Xác nhận phần hands-on của lab là rank experiment với `r=8,16,64`.
- `Lab21_Rubric_and_Format.md`
  - Mô tả rubric, deliverables, cấu trúc `REPORT.md`, cách nộp bài.
- `notebooks/Lab21_LoRA_Finetuning_T4.ipynb`
  - Notebook thao tác chính cho Google Colab Free GPU T4.
  - Dùng model `unsloth/Qwen2.5-3B-bnb-4bit`.
  - Mặc định lấy 200 mẫu dataset để chạy vừa T4.

## 4. Bài nộp cần có gì?

Theo rubric, bạn cần nộp ít nhất các thành phần sau:

- `REPORT.md`
- `notebook.ipynb` đã clear outputs
- kết quả experiment, ít nhất gồm:
  - adapter checkpoint
  - `rank_experiment_summary.csv`
  - `qualitative_comparison.csv`
  - `loss_curve.png`

Repo rubric đưa ra 3 cách nộp:

1. Option A - ZIP gọn nhẹ, nộp adapter tốt nhất + bảng kết quả.
2. Option B - Đẩy adapter lên HuggingFace Hub, nộp link.
3. Option C - Code/report only, ưu tiên tính reproducible.

Nếu muốn an toàn và dễ nộp nhất, chọn **Option A**.

## 5. Bạn cần làm thực tế theo thứ tự nào?

## Bước 1 - Chuẩn bị môi trường

- Mở notebook `notebooks/Lab21_LoRA_Finetuning_T4.ipynb` trên Google Colab.
- Bật `Runtime > Change runtime type > GPU`.
- Chạy các cell setup:
  - `nvidia-smi`
  - kiểm tra `torch.cuda.is_available()`
  - cài thư viện: Unsloth, TRL, PEFT, bitsandbytes, datasets, pandas, matplotlib

Bạn cần xác nhận:

- GPU đang có sẵn
- VRAM được nhận
- môi trường cài thư viện thành công

## Bước 2 - Chuẩn bị dataset

Lab yêu cầu dataset theo Alpaca format:

```json
{
  "instruction": "...",
  "input": "...",
  "output": "..."
}
```

Yêu cầu dataset:

- Khoảng **100-500 samples**
- Nên được làm sạch
- Chia **90% train / 10% eval**
- Dùng `seed=42`

Notebook cho 2 cách:

1. Dùng dataset có sẵn:
   - `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated`
2. Dùng dataset tự tạo:
   - uncomment phần custom data và đưa dữ liệu của bạn vào

Notebook đã hỗ trợ:

- auto-detect tên cột `instruction/input/output`
- format lại prompt huấn luyện
- tính độ dài token

## Bước 3 - Phân tích token length và chọn `max_seq_length`

Bạn không nên đặt `max_seq_length` theo cảm tính.

Cần làm:

1. Tokenize dataset.
2. Tính phân vị `p95` của độ dài token.
3. Chọn `max_seq_length` dựa trên `p95`.
4. Với profile T4, notebook đang giới hạn `MAX_SEQ_CAP = 1024`.

Ý nghĩa:

- Nếu đặt quá lớn, dễ OOM.
- Nếu đặt quá nhỏ, cắt mất nội dung mẫu train.

## Bước 4 - Load base model theo QLoRA

Notebook T4 đang dùng:

- `MODEL_NAME = "unsloth/Qwen2.5-3B-bnb-4bit"`

Đây là model 4-bit, phù hợp T4 16GB.

Bạn cần load model bằng Unsloth `FastLanguageModel` với:

- `load_in_4bit=True`
- `max_seq_length=MAX_SEQ_LENGTH`
- gradient checkpointing bật

Mục tiêu ở bước này là có một base model sẵn sàng để wrap LoRA.

## Bước 5 - Cấu hình LoRA baseline

Baseline của lab là:

```python
r = 16
lora_alpha = 32
target_modules = ["q_proj", "v_proj"]
lora_dropout = 0
```

Notebook xác nhận:

- baseline bắt đầu với `r=16`
- target modules theo lab spec là `q_proj` và `v_proj`
- gradient checkpointing được dùng để giảm VRAM

Bạn cần hiểu:

- `r` càng cao thì trainable params càng nhiều
- `alpha` được scale theo rank
- baseline sẽ là mốc để so sánh với `r=8` và `r=64`

## Bước 6 - Train baseline `r=16`

Hyperparameters chính theo rubric/notebook:

- `epochs = 3`
- `learning_rate = 2e-4`
- `lr_scheduler_type = cosine`
- `warmup_ratio = 0.10`
- effective batch size = 8 thông qua gradient accumulation
- T4 profile dùng train batch = 1
- `packing=False`
- eval during training tắt để tránh OOM trên T4

Bạn cần thu được:

- adapter checkpoint cho `r=16`
- training loss
- training time
- peak VRAM
- eval loss / perplexity sau train

## Bước 7 - Vẽ loss curve và quan sát overfitting

Sau khi train baseline:

- Vẽ `loss_curve.png`
- Nếu có `eval_loss` thì đối chiếu train loss và eval loss
- Ghi nhận xem có dấu hiệu overfitting hay không

Phần này bắt buộc phải được nhận xét trong `REPORT.md`.

## Bước 8 - Làm rank experiment

Đây là phần quan trọng nhất của lab.

Bạn phải train thêm 2 adapter:

1. `r=8`, `alpha=16`
2. `r=64`, `alpha=128`

Quy tắc khi so sánh:

- Cùng dataset
- Cùng base model
- Cùng hyperparameters train
- Chỉ thay `rank` và `alpha`

Cần ghi lại cho mỗi rank:

- training time
- peak VRAM
- trainable params
- eval loss
- eval perplexity

Notebook đã có hàm `train_one_rank(r, alpha)` để phục vụ bước này.

## Bước 9 - Đánh giá kết quả

Lab yêu cầu 2 kiểu đánh giá:

### 9.1. Quantitative

Tính cho cả:

- base model
- rank 8
- rank 16
- rank 64

Metric chính:

- `eval_loss`
- `perplexity = exp(eval_loss)`

### 9.2. Qualitative

Generate ít nhất **5 prompts test** để so sánh.

Notebook đã có sẵn danh sách prompt mẫu. Bạn cần so sánh:

- output của base model
- output của fine-tuned model, ưu tiên `r=16`

Nên nhận xét:

- output có đúng style/domain hơn không
- có rõ ràng hơn không
- có hallucination hay lặp lại không

## Bước 10 - Lưu kết quả

Theo notebook, bạn cần lưu:

- folder adapter `r8/`
- folder adapter `r16/`
- folder adapter `r64/`
- `rank_experiment_summary.csv`
- `qualitative_comparison.csv`
- `loss_curve.png`

Nếu nộp theo Option A, rubric cho phép chỉ nộp adapter của rank tốt nhất, nhưng bảng kết quả vẫn phải có đủ số liệu cho cả 3 rank.

## Bước 11 - Viết `REPORT.md`

Rubric bắt buộc `REPORT.md` phải có 6 phần:

1. **Setup**
   - base model
   - dataset
   - số mẫu train/eval
   - `max_seq_length`
   - GPU
   - training cost ước tính
2. **Rank Experiment Results**
   - bảng so sánh `r=8`, `r=16`, `r=64`, và base
3. **Loss Curve Analysis**
   - có/không overfitting, vì sao
4. **Qualitative Comparison**
   - ít nhất 5 ví dụ before/after
5. **Conclusion về Rank Trade-off**
   - tối thiểu 100 từ
   - kết luận rank nào cho ROI tốt nhất
6. **What I Learned**
   - 2-3 ý reflection cá nhân

Đây là phần chiếm nhiều điểm, nên không được viết qua loa.

## 6. Tiêu chí chấm điểm bạn cần lưu ý

Rubric chấm trên 100 điểm, trong đó:

- 40 điểm: code chạy end-to-end
- 25 điểm: thiết kế experiment và phân tích
- 15 điểm: evaluation quality
- 20 điểm: report quality

Điểm sẽ giảm mạnh nếu:

- không train đủ 3 rank
- không tính được perplexity
- qualitative examples quá sơ sài
- report thiếu 1 trong 6 mục bắt buộc

## 7. Cách làm tối ưu để hoàn thành bài này

Nếu mục tiêu là làm đúng và đủ bài, quy trình đề nghị:

1. Chạy notebook T4 gốc trước, dùng dataset mặc định 200 samples.
2. Xác nhận baseline `r=16` train được.
3. Chạy tiếp `r=8` và `r=64`.
4. Xuất CSV và loss curve.
5. Làm qualitative comparison với 5-10 prompts.
6. Viết `REPORT.md` dựa trên bảng kết quả.
7. Clear outputs của notebook trước khi nộp.

Nếu còn dư thời gian, mới cần:

- đổi dataset theo domain yêu thích
- push adapter lên HuggingFace Hub
- làm stretch goals

## 8. Các lỗi thường gặp

Repo đã nhắc sẵn những lỗi này:

- `KeyError: 'instruction'`
  - do tên cột khác, notebook đã có auto-detect
- lỗi do version `TRL` / `Transformers`
  - notebook đã thêm patch tương thích
- `packing=True` gây lỗi mismatch
  - notebook đang dùng `packing=False`
- OOM khi evaluate
  - notebook có `safe_evaluate()` fallback

Nếu bạn bị thiếu VRAM:

1. giảm `max_seq_length`
2. giữ batch size = 1
3. bật gradient checkpointing
4. clear cache giữa các rank training

## 9. Kết luận ngắn

Lab này về bản chất là một bài thực hành QLoRA đầy đủ:

- chuẩn bị dataset
- train 3 LoRA adapters
- đo metrics
- so sánh rank trade-off
- viết report kỹ thuật

Nếu làm đúng yêu cầu, sản phẩm cuối cùng của bạn sẽ là:

- 3 checkpoint adapter
- bảng tổng hợp metrics
- qualitative comparison
- loss curve
- `REPORT.md` có kết luận rank nào nên dùng và vì sao
