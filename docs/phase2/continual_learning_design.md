# Thiết kế Cơ chế Continual Learning (EWC + Replay Buffer) cho MTL Churn Model

Tài liệu này trình bày chi tiết thiết kế hệ thống Continual Learning tích hợp vào mô hình Multi-Task Learning (MTL) nhằm khắc phục hiện tượng quên tri thức cũ (Catastrophic Forgetting) khi mô hình được cập nhật định kỳ trên các tập dữ liệu khách hàng mới.

---

## 1. Điều kiện kích hoạt ContinualMTL thay vì MTL thường

Khi kích hoạt tiến trình huấn luyện AutoML, hệ thống sẽ tự động phân tích các điều kiện để quyết định xem có chạy chế độ huấn luyện liên tục (**ContinualMTL**) hay không.

### Các điều kiện bắt buộc:
*   **Có `prior_model_uri`**: Client phải cung cấp đường dẫn URI đến mô hình đã được huấn luyện trước đó trên hệ thống MLflow (ví dụ: `runs:/<run_id>/model`). Đường dẫn này dùng để tải bộ tham số cũ và cấu trúc mô hình cũ để fine-tune.
*   **Môi trường có thư viện PyTorch (`torch`)**: Do kiến trúc MTL của chúng ta được cài đặt bằng PyTorch, quá trình tính toán Fisher Matrix và áp dụng EWC Penalty bắt buộc phải có PyTorch.
*   **Số lượng dòng dữ liệu lớn hơn 500 (`n_rows > 500`)**: Nếu kích thước tập dữ liệu huấn luyện mới quá nhỏ, việc huấn luyện Continual Learning có thể dẫn đến overfitting hoặc ước lượng Fisher không chính xác.

> [!IMPORTANT]
> **Cơ chế Fallback**: Nếu thiếu bất kỳ điều kiện nào trong 3 điều kiện trên (ví dụ: không truyền `prior_model_uri`, môi trường thiếu thư viện `torch`, hoặc dữ liệu mới có ít hơn hoặc bằng 500 dòng) ➔ Hệ thống sẽ tự động **fallback** về tiến trình huấn luyện Standard MTL hoặc Standard AutoML thông thường.

---

## 2. Phương pháp Elastic Weight Consolidation (EWC)

EWC là thuật toán regularization giúp hạn chế việc thay đổi các trọng số quan trọng đối với các tác vụ cũ bằng cách phạt (penalize) các thay đổi trọng số dựa trên độ quan trọng của chúng.

### 2.1. Tính toán Fisher Information Matrix (Độ quan trọng của trọng số)
Để tối ưu bộ nhớ và hiệu năng tính toán, chúng ta sử dụng **Diagonal Fisher Approximation** (xấp xỉ đường chéo của ma trận Fisher) thay vì lưu trữ toàn bộ ma trận:

$$F_i = \frac{1}{N} \sum_{n=1}^N \left( \frac{\partial \log p(x_n | \theta)}{\partial \theta_i} \right)^2$$

*   Với mỗi trọng số $\theta_i$, hệ thống sẽ tính đạo hàm bậc nhất của log-likelihood (gradients) của mô hình trên tập dữ liệu cũ/hiện tại trước khi cập nhật dữ liệu mới.
*   Trọng số $F_i$ càng lớn thể hiện tham số $\theta_i$ đó đóng vai trò càng quan trọng trong việc đưa ra dự báo đúng cho tập dữ liệu cũ.

### 2.2. Tham số Lambda EWC (`lambda_ewc`)
*   **Giá trị mặc định**: `100.0`
*   **Cấu hình qua Environment Variable**: `EWC_LAMBDA`
*   Ý nghĩa: Tham số này điều khiển mức độ phạt khi thay đổi trọng số cũ. Nếu `EWC_LAMBDA` quá cao, mô hình sẽ không thể học được tri thức mới; nếu quá thấp, mô hình sẽ quên nhanh tri thức cũ.

### 2.3. Cách EWC Penalty được thêm vào Loss Function
Trong quá trình huấn luyện với dữ liệu mới, hàm loss tổng hợp của mô hình PyTorch sẽ được cộng thêm một thành phần EWC regularization penalty:

$$\mathcal{L}_{total}(\theta) = \mathcal{L}_{new}(\theta) + \sum_{i} \frac{\lambda_{ewc}}{2} F_i (\theta_i - \theta_i^*)^2$$

*   Trong đó $\mathcal{L}_{new}(\theta)$ là loss thông thường của MTL (gồm Binary Cross Entropy cho Churn + MSE/Smooth L1 cho CPI).
*   $\theta_i^*$ là giá trị trọng số của mô hình cũ được tải từ `prior_model_uri`.

---

## 3. Cơ chế Replay Buffer

Replay Buffer giúp duy trì hiệu năng của mô hình trên phân phối dữ liệu cũ bằng cách trộn một phần dữ liệu cũ vào quá trình huấn luyện dữ liệu mới.

### 3.1. Lưu trữ và Đường dẫn trên R2
Dữ liệu của Replay Buffer được lưu trữ dưới dạng tệp tin Parquet trên Cloudflare R2:
*   **Đường dẫn**: `ml_artifacts/{dataset_id}/replay_buffer.parquet`

### 3.2. Cấu hình Kích thước (Max Size)
*   **Giá trị mặc định**: `1000` dòng
*   **Cấu hình qua Environment Variable**: `REPLAY_BUFFER_MAX_SIZE`

### 3.3. Chiến lược Lấy mẫu (Sampling Strategy)
*   Sử dụng phương pháp **Stratified Sampling by Target Label** (Lấy mẫu phân tầng theo nhãn Churn).
*   Mục tiêu: Đảm bảo tỷ lệ nhãn Churn (ví dụ: tỷ lệ khách hàng rời bỏ thực tế là 15%) được duy trì nhất quán giữa tập dữ liệu cũ được giữ lại và tập dữ liệu huấn luyện mới, tránh hiện tượng lệch nhãn (class imbalance shift).

### 3.4. Tỷ lệ Trộn Dữ liệu (Mixing Ratio)
*   **Tỷ lệ mặc định**: `20%` dữ liệu từ Replay Buffer và `80%` dữ liệu mới.
*   **Cấu hình qua Environment Variable**: `REPLAY_BUFFER_RATIO` (chấp nhận giá trị từ `0.0` đến `1.0`).

---

## 4. Sự thay đổi trong Vòng lặp Huấn luyện (Training Loop)

*   **Trước đây (Cũ)**: 
    *   Hệ thống tải dữ liệu mới `X_new, y_new` ➔ Gọi `fit(X_new, y_new)` tối ưu hóa loss thông thường.
*   **Hiện tại (Mới)**:
    1.  Tải dữ liệu mới `X_new, y_new`.
    2.  Tải dữ liệu lịch sử từ Replay Buffer trên R2 (`X_replay, y_replay`) nếu có.
    3.  Lấy mẫu phân tầng và trộn dữ liệu theo tỷ lệ `REPLAY_BUFFER_RATIO` để tạo tập dữ liệu tổng hợp:
        $$X_{mixed} = \text{Concat}(X_{new}, X_{replay})$$
        $$y_{mixed} = \text{Concat}(y_{new}, y_{replay})$$
    4.  Tải trọng số mô hình cũ $\theta^*$ từ `prior_model_uri` và tính toán ma trận đường chéo Fisher $F_i$.
    5.  Chạy vòng lặp tối ưu trọng số trên tập dữ liệu $X_{mixed}, y_{mixed}$ với hàm loss tổng hợp có chứa EWC penalty.
    6.  Cập nhật lại Replay Buffer trên R2 bằng cách chọn lọc và lưu trữ một tập dữ liệu phân tầng mới từ $X_{mixed}$.

---

## 5. Tác động đến API Surface

### 5.1. Thay đổi Schema `TrainingRequest`
Trường `prior_model_uri` được bổ sung làm trường tùy chọn trong Pydantic schema:

```python
class TrainingRequest(BaseModel):
    dataset_id: str
    target_column: str
    prior_model_uri: Optional[str] = None  # Thêm mới phục vụ Continual Learning
```

### 5.2. Luồng xử lý khi nhận Yêu cầu
1.  Client gửi yêu cầu `POST /api/v1/jobs/datasets/{dataset_id}/train` kèm theo `prior_model_uri`.
2.  Hệ thống kiểm tra tính hợp lệ của `prior_model_uri` trên MLflow.
3.  Tải Replay Buffer từ R2 nếu tệp tin tồn tại tại đường dẫn chỉ định.
4.  Bắt đầu Background Task chạy AutoML định tuyến tới Continual Learning.

---

## 6. Hạn chế kỹ thuật (Known Limitations)

*   **Diagonal Fisher Approximation**: Việc chỉ lưu trữ đường chéo Fisher Information Matrix ($O(n_{params})$) thay vì ma trận đầy đủ ($O(n_{params}^2)$) giúp tiết kiệm bộ nhớ và thời gian tính toán, nhưng sẽ bỏ qua tương quan chéo giữa các tham số, làm giảm độ chính xác của EWC penalty ở mức độ nhỏ.
*   **R2 I/O Latency**: Quá trình tải và lưu Replay Buffer lên Cloudflare R2 qua mạng có thể tăng thời gian khởi chạy tác vụ huấn luyện (khoảng từ vài trăm miliseconds đến vài giây tùy thuộc kích thước buffer).
*   **Bất biến Kiến trúc Mô hình (Fixed Model Architecture)**: Cơ chế EWC và Weight Mapping yêu cầu cấu trúc mạng nơ-ron (số lượng lớp, số lượng nơ-ron mỗi lớp, các tên trọng số) của mô hình cũ và mô hình mới phải hoàn toàn trùng khớp. Nếu kiến trúc mô hình thay đổi giữa các lần cập nhật, hệ thống sẽ tự động hủy tiến trình Continual Learning và quay về huấn luyện Standard MTL từ đầu.
