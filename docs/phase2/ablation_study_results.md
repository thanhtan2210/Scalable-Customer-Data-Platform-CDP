# Kết quả Thử nghiệm Ablation Study (Continual Learning)

Tài liệu này ghi nhận kết quả thử nghiệm thực tế của cuộc khảo sát Ablation Study nhằm đánh giá đóng góp của hai thành phần **Elastic Weight Consolidation (EWC)** và **Replay Buffer** trong việc ngăn chặn hiện tượng quên tri thức cũ (Catastrophic Forgetting) của mô hình `MTLChurnModel`.

---

## 1. Kết quả thực nghiệm chính thức

### 1.1. Thử nghiệm trên Single Seed (SEED = 42)
Kết quả thu được khi chạy thử nghiệm đơn lẻ với hạt giống cố định:

| Kịch bản | Sử dụng EWC | Sử dụng Replay | $AUC_{A\_before}$ | $AUC_{A\_after}$ | $AUC_{B}$ | Forgetting Rate (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | ❌ | ❌ | 0.7036 | 0.6473 | 0.8487 | 7.99% |
| **Replay only** | ❌ | ✅ | 0.7036 | 0.6504 | 0.8356 | 7.56% |
| **EWC only** | ✅ | ❌ | 0.7036 | 0.6981 | 0.8508 | 0.78% |
| **Full (EWC+Replay)** | ✅ | ✅ | 0.7036 | **0.6931** | 0.8638 | **1.49%** |

### 1.2. Đánh giá ý nghĩa thống kê (3 Seeds: 42, 123, 456)
Kết quả trung bình kèm độ lệch chuẩn (Mean ± Std) qua nhiều hạt giống ngẫu nhiên:

| Kịch bản | Sử dụng EWC | Sử dụng Replay | $AUC_{A\_before}$ | $AUC_{A\_after}$ | $AUC_{B}$ | Forgetting Rate (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | ❌ | ❌ | 0.7295 ± 0.0183 | 0.6640 ± 0.0118 | 0.7953 ± 0.0423 | 8.95% ± 0.70% |
| **Replay only** | ❌ | ✅ | 0.7295 ± 0.0183 | 0.6659 ± 0.0115 | 0.7938 ± 0.0345 | 8.69% ± 1.00% |
| **EWC only** | ✅ | ❌ | 0.7295 ± 0.0183 | 0.6879 ± 0.0079 | 0.8018 ± 0.0350 | 5.61% ± 3.46% |
| **Full (EWC+Replay)** | ✅ | ✅ | 0.7295 ± 0.0183 | **0.6853 ± 0.0072** | 0.8227 ± 0.0360 | **5.98% ± 3.21%** |

---

## 2. Discussion (Thảo luận)

> [!IMPORTANT]
> **Disclaimer**: Các giải thích dưới đây là hypotheses dựa trên kết quả thực nghiệm và được đối chiếu với literature. Việc xác nhận cơ chế cụ thể nào đang diễn ra đòi hỏi thêm các thí nghiệm phân tích (ví dụ: visualize gradient alignment, Fisher Information Matrix analysis).

Dựa trên kết quả thực nghiệm, chúng ta nhận thấy kịch bản kết hợp `Full (EWC+Replay)` (tỷ lệ quên **5.98% ± 3.21%**) không tối ưu tốt hơn kịch bản chỉ sử dụng EWC đơn lẻ (`EWC only`, tỷ lệ quên **5.61% ± 3.46%**). Dưới đây là 4 giả thuyết khoa học giải thích cho hiện tượng này:

### Luận điểm 1: Xung đột Gradient và Nhiễu Tối ưu hóa (Gradient Interference & Conflict)
*   **Nội dung**: EWC áp dụng một ràng buộc phạt bậc hai kéo các trọng số về vùng tối ưu của tác vụ cũ A. Khi kết hợp Replay, mô hình nhận thêm vector cập nhật gradient thực tế $\nabla \mathcal{L}_{\text{replay}}$ từ các mẫu lịch sử trong buffer. Do hướng đi của gradient từ các mẫu này có thể mâu thuẫn trực tiếp với hướng co giãn của EWC (cosine similarity âm), chúng tạo ra xung đột vector cập nhật (gradient conflict), dẫn đến nhiễu tối ưu và làm giảm một phần khả năng giữ lại thông tin cũ.
*   **Trích dẫn khoa học**: Hiện tượng xung đột và giao thoa gradient giữa các tác vụ được thảo luận chi tiết bởi Riemer et al. (2019) [3].

### Luận điểm 2: Nghịch lý Stability-Plasticity & Quá điều hòa (Over-Regularization)
*   **Nội dung**: Việc áp dụng hệ số phạt EWC cứng nhắc ($\lambda_{\text{ewc}} = 100.0$) giới hạn đáng kể không gian thay đổi tham số (tính ổn định cao). Khi ép mô hình đồng thời phải fit dữ liệu Task B và dữ liệu Replay trong không gian bị bó hẹp này, mô hình gặp hiện tượng quá điều hòa (over-regularization). Sự thiếu hụt không gian tự do cập nhật (plasticity loss) khiến mô hình rơi vào trạng thái thỏa hiệp cục bộ.
*   **Trích dẫn khoa học**: Cơ chế cân bằng giữa giữ vững tri thức cũ và học tri thức mới này tương ứng với nghiên cứu của Kirkpatrick et al. (2017) [1].

### Luận điểm 3: Sự suy giảm tính chính xác của xấp xỉ bậc hai EWC (Taylor Basins Drift)
*   **Nội dung**: EWC sử dụng xấp xỉ Taylor bậc nhất của đạo hàm log-likelihood (Fisher diagonal) xung quanh điểm tối ưu cũ $\theta_A^*$. Khi quá trình huấn luyện Replay kéo các tham số trôi ra xa khỏi vùng lân cận Taylor này, xấp xỉ bậc hai mất đi tính chính xác, khiến hàm phạt EWC hoạt động không còn hiệu quả.
*   **Trích dẫn khoa học**: `[cần tìm]` (Các phân tích giới hạn của việc điều hòa trọng số khi tham số trôi xa).

### Luận điểm 4: Quá khớp cục bộ do giới hạn kích thước bộ đệm Replay (Buffer Overfitting)
*   **Nội dung**: Bộ đệm Replay bị giới hạn kích thước ở mức $M = 1000$ mẫu. Việc huấn luyện lặp đi lặp lại trên một tập dữ liệu nhỏ không đại diện hoàn toàn cho phân phối xác suất thực tế của Task A ($3800+$ mẫu) dễ dẫn đến hiện tượng quá khớp cục bộ vào các mẫu đệm, làm sai lệch ranh giới quyết định tổng quát của mô hình trên tập kiểm thử Task A, không đủ để ngăn chặn catastrophic forgetting (quan sát thực nghiệm trong nghiên cứu này; xem Bảng 1: Replay only Forgetting Rate = 7.56% so với Baseline 7.99%).
*   **Trích dẫn khoa học**: Không có (dựa trên quan sát thực nghiệm trong nghiên cứu này).

---

## 3. References (Tài liệu tham khảo)

[1] Kirkpatrick, J., Pascanu, R., Rabinowitz, N., et al. (2017).
    Overcoming catastrophic forgetting in neural networks.
    Proceedings of the National Academy of Sciences, 114(13), 3521–3526.
    arXiv: 1612.00796

[2] Nguyen, C. V., Li, Y., Bui, T. D., & Turner, R. E. (2018).
    Variational Continual Learning.
    International Conference on Learning Representations (ICLR 2018).
    arXiv: 1710.10628
    (Note: Mô hình variational/Bayesian continual learning).

[3] Riemer, M., Cases, I., Ajemian, R., et al. (2019).
    Learning to Learn without Forgetting by Maximizing Transfer and Minimizing Interference.
    International Conference on Learning Representations (ICLR 2019).
    arXiv: 1810.11910

[4] Chaudhry, A., Ranzato, M., Rohrbach, M., & Elhoseiny, M. (2019).
    Efficient Lifelong Learning with A-GEM.
    International Conference on Learning Representations (ICLR 2019).
    arXiv: 1812.00420
