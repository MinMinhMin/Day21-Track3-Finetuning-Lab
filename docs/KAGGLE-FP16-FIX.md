# Sửa lỗi NaN khi train Qwen3.5 trên Kaggle T4×2

## Lỗi và bằng chứng

Log NB3 dừng ở step 5 với `loss=1.836e+07`, `grad_norm=nan`, `entropy=nan`.
Hai lỗi lệch GPU trước đó đã qua; lỗi hiện tại nằm trong backward FP16.

Đã tái hiện bằng Qwen3.5 nhỏ, PEFT và loss chunked của TRL, không cần tải model:

| Đường chạy trước bản sửa | Loss forward | Backward |
|---|---:|---|
| FP16, không AMP | 4.1608 | 34 tensor gradient không hữu hạn |
| FP16, AMP | 4.1610 | 34 tensor gradient không hữu hạn |
| FP32 | 4.1609 | hữu hạn |
| BF16 | 4.1589 | hữu hạn |

Anomaly detection chỉ ra `RsqrtBackward0` trong `l2norm(key)` của linear attention.
Transformers 5.15 chuẩn hóa Q/K trước khi chuyển sang FP32. Với vector FP16 nhỏ
hoặc bằng 0, đạo hàm của `rsqrt` tràn số; phép nhân tiếp theo có thể tạo `0 * inf`.
Loss forward hữu hạn không chứng minh backward hợp lệ.

### Lỗi dừng ở step 15 sau bản sửa đầu

Log tiếp theo đã qua preflight và train được: loss lần lượt 2.132 → 1.494 →
0.2227 → 0.04354, entropy ở step 15 là 0.05544. Run dừng vì callback từ chối
`grad_norm=nan`. Callback cũ đã coi cả gradient của update bị GradScaler bỏ qua
là lỗi không thể phục hồi.

Đã tái hiện trường hợp này bằng gradient FP16 thật: loss hữu hạn khoảng 0.04,
gradient NaN tại scale 128, Accelerate xác nhận bỏ qua optimizer update và trọng
số không thay đổi. Scale giảm về 64; update kế tiếp có gradient hữu hạn và cập
nhật trọng số thành công. Đây là cơ chế xử lý overflow của
[GradScaler](https://docs.pytorch.org/docs/2.14/notes/amp_examples.html).

Log Kaggle chưa in cờ skip hay scale sau update, nên chỉ loss thấp không đủ để
kết luận gradient NaN là an toàn. Callback mới chỉ cho tiếp tục khi **đủ cả ba**:

- Accelerate báo `optimizer_step_was_skipped=True` tại đúng step đó.
- GradScaler FP16 đang hoạt động và scale đã giảm so với trước update.
- Toàn bộ trọng số trainable vẫn hữu hạn.

Callback in cảnh báo có step và scale trước/sau. Loss, eval loss hoặc entropy
NaN vẫn dừng; gradient NaN không được xác nhận vẫn dừng; run không có update
thành công nào cũng không được lưu adapter.

Con số 18 triệu trong log không đủ để suy ra loss thực: mặc định Trainer thay loss
NaN bằng loss tích lũy trước đó. Với gradient accumulation, phép thay thế lặp lại
có thể khuếch đại giá trị đã tích lũy. Bản sửa tắt bộ lọc này để log phản ánh lỗi.

## Những thay đổi được áp dụng

1. Tính L2 normalization trong FP32, trước `rsqrt`, cho đường FP16 của Qwen3.5.
   Baseline, train và eval cùng dùng cách chuẩn hóa này. Trọng số nền vẫn 16-bit;
   QLoRA vẫn dùng NF4 4-bit.
2. Tắt autocast bên trong từng chunk loss, tính projection và softmax bằng FP32.
   Chỉ gọi `.float()` trước matmul chưa đủ vì AMP có thể chuyển matmul về FP16.
   Cách bảo vệ từng chunk cũng áp dụng khi checkpoint tính lại trong backward.
3. Giữ chuyển hidden states/labels sang GPU chứa `lm_head`, và wrapper cho partial
   forward của Qwen3.5. Đây là hai lỗi tương thích riêng đã gặp trên T4×2.
4. Khởi tạo GradScaler ở 128 thay vì 65536. Thử nghiệm trên cùng model nhỏ sau khi
   sửa normalization: scale 128 có gradient hữu hạn; scale 65536 vẫn gây overflow.
   Cơ chế tăng/giảm thang và bỏ qua update bị overflow của GradScaler vẫn hoạt động.
5. NB3 và mỗi run NB4 kiểm tra một batch thật bằng forward + scaled backward
   trước khi train. Kiểm tra không cập nhật trọng số, giữ trạng thái RNG và xóa
   gradient khi kết thúc. Không lưu adapter nếu loss/weights không hữu hạn hoặc
   gradient NaN không được xác nhận là update AMP đã bỏ qua an toàn.
6. Bật log step đầu tiên, tắt `logging_nan_inf_filter`, dùng checkpoint không reentrant.
7. Cố định Transformers 5.15.0, TRL 1.10.0, PEFT 0.20.0, Accelerate 1.14.0 và
   Tokenizers 0.22.2 — các phiên bản đã dùng trong kiểm tra tái hiện và kiểm tra sửa lỗi.

## Chạy lại trên Kaggle

1. Push các thay đổi lên fork, rồi import lại `colab/Lab21_RUN_ALL.ipynb` mới.
2. Khởi động session mới, bật Internet, chọn **GPU T4×2**. Chạy Setup để lấy commit
   và phiên bản thư viện mới. Tránh dùng kernel đã import thư viện trước khi cài lại.
3. Chạy Smoke. Kiểm tra gradient/loss giữa hai GPU chạy tự động khi có T4×2.
4. Chạy core với:

   ```python
   COMPUTE_TIER = "T4"
   EVAL_LIMIT = ""
   STAGES = "nb1 nb2 nb3 nb4 nb5"
   FORCE_RETRAIN = True
   ```

   Chạy lại cả NB2 vì đường chuẩn hóa đã thay đổi; không trộn baseline cũ với eval mới.
5. NB3/NB4 phải in `norm_dtype: fp32`, `projection_dtype: fp32`, scaler 128 và
   `forward/backward preflight` có `finite: True`, loss hữu hạn, gradient khác 0.
   Các run phải hoàn thành, có adapter và dòng kết quả riêng trong `results/runs.csv`.
   Nếu có cảnh báo `AMP overflow ... optimizer update skipped`, xem các cột
   `optimizer_steps_attempted`, `optimizer_updates`, `amp_skipped_steps`,
   `amp_final_scale`. `max_steps` là ngân sách step dự kiến; nếu số update thực
   tế khác nhau giữa các run, phải ghi rõ hạn chế đó khi so sánh trong report.
6. Dùng kết quả NB5 để viết `submission/REPORT.md`, rồi chạy gatekeeper trước khi
   nộp. Gatekeeper có thể báo report còn placeholder ngay sau run; cần điền report.
   Cell ZIP/upload vẫn dùng repo HF đã cấu hình và loại `.env`, `__pycache__`.

Nếu vẫn còn session của log step 15, dữ liệu NB1 và baseline NB2 đã được lưu.
Sau khi push bản sửa callback, chạy `git pull --ff-only` trong thư mục repo trên
Kaggle, rồi đổi `STAGES = "nb3 nb4 nb5"` trong cell Core và chạy lại với
`FORCE_RETRAIN=True`. Bản sửa callback không thay đổi forward/eval, nên không
cần tính lại baseline NB2 đã tạo bằng bản sửa FP16 đầu tiên.

## Giới hạn kiểm chứng

Đã kiểm tra tensor thật và ba vòng update LoRA trên Qwen3.5 nhỏ với FP16, AMP,
checkpointing, mask và loss TRL. Máy sửa mã không có CUDA: run đầy đủ model 4B,
QLoRA bitsandbytes và kiểm tra hai GPU cần thực hiện trên Kaggle. Kiểm tra nhỏ
không chứng minh chất lượng adapter hay điểm target/regression của run mới.

Trong report, ghi rõ FP16 weights + FP32 normalization/loss/adapters, GPU T4×2,
phiên bản thư viện và các số đo mới. Không dùng kết quả NaN cũ. Chất lượng tác vụ
vẫn phải kết luận theo eval; bản sửa số học không bảo đảm fine-tune thắng baseline.
