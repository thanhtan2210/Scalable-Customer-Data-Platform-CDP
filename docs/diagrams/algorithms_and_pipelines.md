# Thuật toán & Đường truyền Dữ liệu (ML Pipelines & Algorithms)

Tài liệu này chứa các sơ đồ mô tả luồng xử lý dữ liệu và thuật toán nghiệp vụ chính trong **Churn Prediction Platform**.

---

## 1. Diagram M1: ML Pipeline / Data Flow (Quy trình Pipeline)

### Mô tả
Sơ đồ mô tả chi tiết đường truyền dữ liệu đi qua 7 giai đoạn chính từ lúc upload file cho đến khi phục vụ dự đoán:
1.  **Ingestion**: Tải file, kiểm tra dung lượng (max 50MB), giải mã định dạng dữ liệu (CSV/Excel/Parquet...).
2.  **Profiling**: Phân tích thống kê Layer 1, tự phát hiện cột target (Layer 2), kiểm tra rò rỉ (leakage), và tổng hợp target CPI.
3.  **Feature Engineering**: Xây dựng sklearn pipeline thích ứng với kiểu dữ liệu của từng cột và tự sinh Pandera validation schema.
4.  **Model Routing**: Định tuyến mô hình tối ưu theo đặc tính của dataset (Logistic Regression cho sparse/dữ liệu nhỏ, Random Forest và XGBoost cho dữ liệu dense/lớn).
5.  **Optimization**: Chạy Optuna tìm kiếm siêu tham số, tối ưu hóa threshold tìm điểm F1-max trên Precision-Recall curve. Kích hoạt huấn luyện Multi-Task Learning (MTL) nếu cấu hình CPI và PyTorch sẵn sàng.
6.  **MLflow Registry**: Lưu vết các runs, ghi nhận metrics, tags, schemas và đăng ký mô hình lên MLflow registry.
7.  **Serving**: Model serving qua cơ chế ModelCache (double-check locking, TTL 10 phút) và offload CPU qua `run_in_executor`.

### Preview
![ML Pipeline / Data Flow](../img/ML%20Pipeline_%20Dataflow.drawio.png)

### draw.io XML
Xem mã XML trong artifact `diagrams_missing_M1_M3` (Diagram M1).

---

## 2. Diagram 4: Algorithm Flowcharts (3 Thuật toán cốt lõi)

### Preview
![Algorithm Flowcharts](../img/Algorithm%20Flowcharts.drawio.png)

Tải mã nguồn XML cho từng thuật toán trong artifact `diagrams_fixed` (Diagram 4):
*   **Algorithm 1 — 3-Layer Profiling Flowchart**: Quy trình 3 lớp điều phối phân tích cột dữ liệu.
*   **Algorithm 2 — Target Detection Scoring**: Các bước chấm điểm theo entropy, vị trí, từ khóa để đề xuất cột target churn.
*   **Algorithm 3 — PSI Drift Calculation**: Thuật toán phân thùng dữ liệu (quantile binning), làm mịn phân phối (smoothing epsilon=1e-4) và tính độ ổn định dân số (PSI) để bắt data drift.

---

## 3. Diagram M5: Composite Target Synthesis (CPI Synthesis)

### Mô tả
Sơ đồ thuật toán tổng hợp chỉ số churn ảo (Composite Performance Index) từ các trường thông tin phụ (auxiliary columns). Khi số lượng cột phụ lớn hơn 2, hệ thống sẽ chặn và yêu cầu người dùng xác nhận cấu hình qua endpoint `/confirm-composite`:
*   **Chiến lược WEIGHTED**: Tính trọng số của từng cột phụ dựa trên độ tương quan tuyệt đối với cột target chính, sau đó chuẩn hóa tuyến tính dữ liệu trước khi nhân trọng số.
*   **Chiến lược PCA**: Sử dụng giải thuật Phân tích Thành phần Chính (Principal Component Analysis) để trích xuất 1 chiều đặc trưng (first principal component) giải thích phương sai lớn nhất làm target CPI.

### Preview
![CPI Synthesis Algorithm](../img/CPI%20Synthesis%20Algorithm.drawio.png)

### draw.io XML
Xem mã XML trong artifact `diagrams_missing_M4_M7` (Diagram M5).

---

## 4. Diagram M6: Continual Learning Loop (Vòng lặp Huấn luyện Liên tục)

### Mô tả
Mô tả tiến trình theo trục thời gian của Continual Learning:
*   `t=0`: Huấn luyện mô hình ban đầu $M_0$ trên dữ liệu $D_0$.
*   `t ∈ [0, T1]`: Phục vụ dự đoán, dữ liệu đầu vào (inference input) được ghi nhận và lưu dưới dạng file `.parquet` theo ngày.
*   `t=T1`: Chạy task kiểm tra drift định kỳ. Nếu phát hiện drift (`PSI >= 0.2` hoặc `KS p-value < 0.05`), hệ thống kích hoạt train lại mô hình.
*   `t=T2`: Gọi module `ContinualMTLTrainer` nạp trọng số của mô hình cũ $M_0$, fine-tune mô hình với tốc độ học thấp (learning rate nhỏ) trên tập dữ liệu mới $D_1$.
*   `t=T3`: Đăng ký phiên bản mô hình mới $M_1$. Data Analyst gọi lệnh promote để đổi `is_active` của mô hình mới lên `True`, hệ thống serving tự động nhận diện mô hình mới sau 10 phút cache expire.

### Preview
![Continual Learning Loop](../img/Continual%20Learning%20Loop.drawio.png)

### draw.io XML
Xem mã XML trong artifact `diagrams_missing_M4_M7` (Diagram M6).
