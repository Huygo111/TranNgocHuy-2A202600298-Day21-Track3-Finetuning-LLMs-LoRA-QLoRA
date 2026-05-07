# Lab 21 - Evaluation Report

**Học viên**: Trần Ngọc Huy - 2A202600298  
**Ngày nộp**: 2026-05-07  
**Submission option**: B (Hugging Face Hub) + stretch goal `target ALL layers`

## 1. Setup

- **Base model**: `unsloth/Qwen2.5-3B-bnb-4bit`
- **Dataset**: `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated`
- **Số mẫu**: 200 samples
- **Train / Eval split**: 180 train / 20 eval
- **max_seq_length**: 1024
- **GPU**: Tesla T4, 15.6 GB VRAM
- **Frameworks chính**: Unsloth + TRL SFTTrainer + PEFT + bitsandbytes
- **HF Hub link**: https://huggingface.co/huygo111/qwen2.5-3b-vi-lab21-r16

**Training cost ước tính**:
- Core experiment (`r=8,16,64`): 14.05 phút, khoảng **$0.08** tại mức `$0.35/giờ`
- Nếu tính cả bonus `target ALL layers`: 20.80 phút, khoảng **$0.12**

## 2. Rank Experiment Results

| Rank | Alpha | Trainable Params | Train Time (min) | Peak VRAM (GB) | Eval Loss | Perplexity |
|------|-------|------------------|------------------|----------------|-----------|------------|
| Base | 0 | 0 | 0.00 | - | 1.8840 | 6.5798 |
| 8 | 16 | 1,843,200 | 4.54 | 7.22 | 1.5577 | 4.7479 |
| 16 | 32 | 3,686,400 | 4.76 | 6.62 | 1.5161 | 4.5544 |
| 64 | 128 | 14,745,600 | 4.74 | 8.00 | 1.4768 | 4.3790 |

**Nhận xét nhanh**:
- Fine-tuning giúp perplexity giảm rõ rệt từ **6.58** ở base xuống còn **4.75 / 4.55 / 4.38**.
- `r=64` cho perplexity tốt nhất trong 3 rank chính.
- `r=16` cho trade-off khá tốt: ít tham số train hơn nhiều so với `r=64`, nhưng chất lượng không thua quá xa.
- Trên T4, mức VRAM của `r=64` tăng nhưng vẫn còn nằm trong ngưỡng chạy được.

## 3. Loss Curve Analysis

Notebook T4 profile tắt `eval_during_training` để tránh OOM, vì vậy loss curve chủ yếu phản ánh **train loss** của baseline `r=16`.

Từ log huấn luyện:
- Step 5: loss khoảng **1.6143**
- Step 25: loss khoảng **1.4791**
- Step 45: loss khoảng **1.3802**
- Step 65: loss khoảng **1.3942**

Nhìn chung, train loss giảm khá đều từ đầu đến cuối, chỉ dao động nhẹ ở giữa quá trình train. Không có dấu hiệu mất ổn định lớn như loss tăng vọt hoặc diverge. Vì không có eval loss theo từng bước trong lúc train nên chưa thể kết luận overfitting một cách mạnh tay, nhưng dựa trên:
- train loss giảm hợp lý
- eval perplexity sau train cải thiện rõ so với base

tôi đánh giá baseline `r=16` đang học được pattern của dataset mà chưa cho thấy dấu hiệu overfitting nghiêm trọng trên thiết lập 3 epochs này.

## 4. Qualitative Comparison (5 examples)

### Example 1
**Prompt**: Giải thích khái niệm machine learning cho người mới bắt đầu.  
**Base**: Giải thích đúng ý chính nhưng diễn đạt dài và hơi chung chung.  
**Fine-tuned (`r=16`)**: Câu trả lời gọn hơn, tập trung hơn vào định nghĩa và cơ chế học từ dữ liệu.  
**Nhận xét**: Fine-tuned rõ ý hơn và mang phong cách giải thích trực tiếp hơn.

### Example 2
**Prompt**: Viết đoạn code Python tính số Fibonacci thứ n.  
**Base**: Đưa ra lời giải hợp lệ nhưng phần giải thích và kiểm tra input chưa gọn.  
**Fine-tuned (`r=16`)**: Trả lời có cấu trúc hơn, code sạch và kiểm tra input hợp lý.  
**Nhận xét**: Fine-tuned cải thiện chất lượng trình bày và tính thực dụng của lời giải.

### Example 3
**Prompt**: Liệt kê 5 nguyên tắc thiết kế UI/UX.  
**Base**: Có ý đúng nhưng hơi lan man và nhiều câu mang tính mô tả chung.  
**Fine-tuned (`r=16`)**: Trả lời ngắn gọn hơn, dễ đọc hơn theo kiểu liệt kê.  
**Nhận xét**: Fine-tuned hợp với định dạng instruction-answer hơn.

### Example 4
**Prompt**: Tóm tắt sự khác biệt giữa LoRA và QLoRA.  
**Base**: Phân biệt tương đối đúng nhưng diễn đạt còn dài và chưa thật sắc nét.  
**Fine-tuned (`r=16`)**: Câu trả lời vẫn đúng hướng, có xu hướng dùng thuật ngữ chặt hơn.  
**Nhận xét**: Có cải thiện về độ tập trung, nhưng ví dụ này cũng cho thấy fine-tuned model đôi lúc dùng thuật ngữ hơi cứng.

### Example 5
**Prompt**: Phân biệt prompt engineering, RAG, và fine-tuning.  
**Base**: Trả lời đúng ý nhưng còn vòng vo.  
**Fine-tuned (`r=16`)**: Trình bày trực tiếp hơn, tách bạch ba khái niệm rõ hơn.  
**Nhận xét**: Fine-tuned tạo cảm giác “instruction-following” tốt hơn base model.

## 5. Conclusion về Rank Trade-off

Kết quả thí nghiệm cho thấy việc tăng rank LoRA giúp mô hình cải thiện chất lượng một cách nhất quán trên eval set, thể hiện qua perplexity giảm từ 4.75 ở `r=8` xuống 4.55 ở `r=16`, rồi 4.38 ở `r=64`. Điều này phù hợp với trực giác rằng rank cao hơn cho phép adapter biểu diễn cập nhật giàu năng lực hơn. Tuy nhiên, mức cải thiện không tăng tuyến tính theo số trainable parameters. Từ `r=16` lên `r=64`, số tham số train được tăng khoảng 4 lần, nhưng perplexity chỉ cải thiện thêm một mức tương đối nhỏ. Đây là dấu hiệu của **diminishing returns**.

Với dataset nhỏ 200 mẫu và hạ tầng T4, tôi cho rằng `r=16` là lựa chọn có **ROI tốt nhất**. Nó cho perplexity tốt hơn rõ so với `r=8`, trong khi chi phí bộ nhớ và thời gian vẫn rất dễ chấp nhận. `r=64` đạt chất lượng tốt nhất trong bài test này, nhưng mức lợi ích thêm chưa thật sự tương xứng với số tham số train lớn hơn nhiều. Nếu triển khai production trong điều kiện tài nguyên giới hạn hoặc cần train nhanh, tôi sẽ ưu tiên `r=16`. Nếu mục tiêu là tối đa hóa chất lượng trên cùng một dataset và phần cứng vẫn đủ VRAM, `r=64` là phương án mạnh hơn, nhưng lợi thế của nó nên được cân nhắc cùng chi phí thực tế.

## 6. What I Learned

- Fine-tuning bằng QLoRA trên T4 là hoàn toàn khả thi nếu kiểm soát tốt `max_seq_length`, batch size và eval strategy.
- Rank cao hơn không tự động đồng nghĩa với ROI tốt hơn; cần nhìn đồng thời vào perplexity, VRAM, thời gian train và số trainable parameters.
- Với notebook production-style, các chi tiết như `safe_evaluate()`, `packing=False`, gradient checkpointing và lưu checkpoint sớm rất quan trọng để tránh mất công train khi gặp lỗi OOM.

## 7. Bonus - Stretch Goal `target ALL layers`

Tôi có chạy thêm một cấu hình bonus với:

- `rank = 16`
- `alpha = 32`
- `target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`

Kết quả:

| Variant | Trainable Params | Train Time (min) | Peak VRAM (GB) | Eval Loss | Perplexity |
|---------|------------------|------------------|----------------|-----------|------------|
| Baseline `r=16` (`q_proj + v_proj`) | 3,686,400 | 4.76 | 6.62 | 1.5161 | 4.5544 |
| Bonus `all_layers_r16` | 29,933,568 | 6.75 | 12.70 | 1.4948 | 4.4586 |

**Nhận xét**:
- `target ALL layers` cải thiện perplexity từ **4.5544** xuống **4.4586**.
- Đổi lại, trainable params tăng rất mạnh và VRAM tăng từ **6.62 GB** lên **12.70 GB**.
- Trên T4 vẫn chạy được, nhưng đây là cấu hình sát trần hơn nhiều.

Kết luận của tôi là bonus này cho thấy mở rộng target modules thực sự có ích về chất lượng, nhưng hiệu quả chi phí không tốt bằng baseline `r=16`. Nó phù hợp hơn khi mục tiêu là ép thêm chất lượng từ model và phần cứng vẫn đủ dư địa.
