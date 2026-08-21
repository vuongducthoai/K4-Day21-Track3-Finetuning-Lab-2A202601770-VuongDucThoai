# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**

Điều làm tôi ngạc nhiên nhất là chỉ cần tối ưu prompt, điểm target của base model đã tăng từ 0 lên 0.765 và format tăng từ 0 lên 1.0. Fine-tune tiếp tục nâng target lên 0.97, nhưng đồng thời làm regression giảm từ 0.7578 xuống 0.6111. Kết quả này cho thấy một model có thể trông rất tốt trên nhiệm vụ đích nhưng vẫn chưa đủ an toàn để triển khai.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**

Tôi mất nhiều thời gian nhất ở Core pipeline, đặc biệt là NB4 với ba run đối chứng, mất 51,6 phút trong tổng 76,8 phút của lượt thử. Ban đầu tôi nghĩ phần setup môi trường và NB3 sẽ lâu nhất, nhưng NB4 mới là phần tốn thời gian vì phải train ba cấu hình riêng. Ngoài ra, tôi cũng mất thời gian phân biệt Jupyter local với Colab và xử lý xung đột package `tests` trong smoke test.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**

Trước lab, tôi nghĩ fine-tune chỉ cần làm điểm nhiệm vụ đích tăng là có thể xem là thành công. Sau lab, tôi không còn tin điều đó: target tăng 0.205 nhưng verdict vẫn FAILED vì năng lực chung giảm 0.147. Tôi cũng không còn xem train loss thấp hơn là bằng chứng model tốt hơn, vì `attn_only` có loss thấp hơn `correct` nhưng hai cấu hình chỉ hòa nhau ở target 0.97.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**

Tôi dùng AI assistant để đọc hướng dẫn repo, thiết lập môi trường Windows, phân tích log pytest, giải thích các stage NB1–NB5, kiểm tra Gatekeeper và hoàn thiện report dựa trên artefact thật. AI giúp tìm ra xung đột `tests.fake_tokenizer` trên Colab và hướng dẫn thêm `tests/__init__.py`. Tuy nhiên, ban đầu AI đề nghị tạo lại `.venv` khi Jupyter vẫn đang sử dụng môi trường đó, dẫn đến lỗi permission denied; nó cũng cần ảnh chụp màn hình mới nhận ra tôi đang dùng Jupyter local chứ không phải Google Colab. Vì vậy tôi phải kiểm tra đường dẫn Python, trạng thái runtime và log thực tế thay vì làm theo mọi đề xuất ngay lập tức.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**

Bước đầu tiên của tôi là cùng khách hàng định nghĩa tiêu chí thành công và đóng băng bộ đánh giá trước khi train: tập target, tập regression, format, latency và ngưỡng chấp nhận. Sau đó tôi sẽ đo ít nhất một base model với naive prompt và một prompt tối ưu để biết fine-tune có thật sự cần thiết hay không. Tôi chỉ bắt đầu chuẩn bị dữ liệu huấn luyện sau khi baseline và cổng đánh giá đã rõ ràng.
