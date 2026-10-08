# Reflection — Lab 21

**Người thực hiện:** Vũ Đức Minh — **MSSV:** 2A202602895.

## 1. Kết quả đáng chú ý nhất

QLoRA đạt target 1,00, cao hơn `correct` 0,99, dù lab đưa ra một cảnh báo về QLoRA cho dòng model này. Tôi cần báo cáo số đã đo thay vì ép thí nghiệm khớp kỳ vọng. Đồng thời, `correct` cải thiện mạnh tác vụ ticket nhưng regression chỉ còn 0,3333 so với baseline 0,6911. Hai kết quả cùng tồn tại: một model có thể chuyên môn hóa tốt và vẫn không đáp ứng điều kiện triển khai đa nhiệm.

## 2. Tôi đã dành công sức ở đâu?

Quá trình làm lab gặp nhiều lỗi tương thích và số học trên Kaggle T4×2: forward của Qwen3.5 với TRL chunked loss, tensor ở hai GPU, normalization FP16, callback kiểm tra gradient và scale của preflight QLoRA. Điểm cần sửa trong cách làm là phân biệt lỗi thực với overflow mà AMP có thể xử lý an toàn, đồng thời kiểm tra số optimizer update thành công. Trong lần chạy đã lưu, NB4 mất 2654 giây, là giai đoạn dài nhất; toàn bộ NB1–NB5 khoảng 4543 giây theo log. Các số đó là thời gian pipeline, không phải thời lượng tất cả lần sửa lỗi trước đó.

## 3. Tôi rút ra điều gì về fine-tuning?

Loss thấp không bảo đảm năng lực tốt hơn trên tác vụ, và target cao không bảo đảm giữ được khả năng chung. `qlora` có train loss cao hơn một chút nhưng target cao hơn `correct`; `correct` lại thất bại ở cổng hồi quy. So attention-only với placement rộng phải khớp ngân sách tham số, nếu không tôi sẽ nhầm hiệu quả của ngân sách với hiệu quả của vị trí. Những kết luận này được giới hạn trong một seed và corpus nhỏ của lab.

## 4. AI assistant được sử dụng thế nào, và sai ở đâu?

Tôi sử dụng AI assistant để đọc hướng dẫn và log, chỉnh mã cho Kaggle, chẩn đoán lỗi FP16, kiểm tra artifact và hỗ trợ biên soạn report. Một callback cũ đã chặn cả gradient NaN thuộc update được AMP bỏ qua; sau đó preflight còn giả định scale 128 phù hợp cho cả QLoRA. Các giả định này phải được thay bằng bằng chứng forward/backward, cờ skip của Accelerate và kiểm tra trọng số. Việc có test trên CPU không cho phép tuyên bố một run đầy đủ trên T4×2 đã thành công; xác nhận cuối phải đến từ log và kết quả Kaggle.

## 5. Nếu làm cho khách hàng thực tế

Tôi bắt đầu bằng việc chốt đầu ra cần có, chi phí của từng dạng lỗi và corpus đánh giá tách biệt trước khi fine-tune. Sau đó đo một baseline prompt đủ mạnh, thu output từng câu, kiểm tra mask và đặt cổng regression phù hợp với phạm vi sản phẩm. Nếu ứng dụng chỉ là triage, cần thêm ticket miền thật và các tình huống thiếu thông tin hoặc ngoài miền. Nếu model phải làm nhiều tác vụ, mức giảm regression của run này là lý do dừng triển khai và thiết kế lại dữ liệu train.

## Nguồn và trách nhiệm rà soát

Phần phản tư được hỗ trợ biên soạn từ trao đổi sửa lỗi và artifact đã lưu. Các số liệu được đối chiếu với `results/` và notebook đã thực thi. Tôi cần đọc lại trước khi nộp để bảo đảm phần diễn giải đúng trải nghiệm của mình; không nhận điểm thưởng hoặc khẳng định output chưa có bằng chứng.
