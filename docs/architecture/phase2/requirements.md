# Phase 2: Dynamic Pipeline & Training Requirements

**Input:** `(List[ColumnProfile], suggested_target: str)`
**Output:** Fit `sklearn.Pipeline` object + MLflow registered model

---

## 1. WHAT — Hệ thống cần làm được gì?

### Transform Registry
- **Hệ thống phải** cung cấp một từ điển (registry) ánh xạ các chiến lược (strategy) từ `ColumnProfile` sang các đối tượng transformer của `scikit-learn` thực tế (ví dụ: `standard` -> `StandardScaler`, `tfidf` -> `TfidfVectorizer`).
- **Hệ thống phải** hỗ trợ dễ dàng mở rộng thêm các transformer mới trong tương lai mà không làm ảnh hưởng đến mã nguồn cốt lõi.

### Pipeline Builder
- **Hệ thống phải** tự động đọc danh sách `ColumnProfile` và phân loại các cột thành các nhóm tương ứng (Numeric, Categorical, Datetime, Text, v.v.).
- **Hệ thống phải** lắp ghép các transformer từ Transform Registry vào một `ColumnTransformer` tổng hợp.
- **Hệ thống phải** tạo ra một `sklearn.pipeline.Pipeline` thống nhất, bao gồm bước tiền xử lý dữ liệu và bước thuật toán học máy (Model).

### Schema tự sinh (Pandera Schema Generation)
- **Hệ thống phải** tự động sinh ra một Data Contract (Pandera Schema) ngay sau khi chốt danh sách `ColumnProfile`.
- **Hệ thống phải** định nghĩa rõ kiểu dữ liệu, bắt buộc hay không bắt buộc (nullable) cho từng cột đầu vào dựa trên Profile.

### Model Router
- **Hệ thống phải** đánh giá bản chất của dữ liệu (số lượng mẫu, số lượng features) để đề xuất danh sách các thuật toán học máy phù hợp nhất (ví dụ: Logistic Regression, Random Forest, XGBoost).
- **Hệ thống phải** trả về mô hình chưa fit kèm theo không gian siêu tham số (hyperparameter search space) tương ứng cho từng thuật toán.

### AutoML Optuna
- **Hệ thống phải** tự động chạy tối ưu hóa siêu tham số (Hyperparameter Tuning) sử dụng framework Optuna trên các model do Model Router đề xuất.
- **Hệ thống phải** tích hợp trực tiếp với MLflow để log từng Trial (metrics, params) và tự động đăng ký (register) mô hình tốt nhất lên DagsHub Model Registry.

---

## 2. WHY — Tại sao cần component này?

- **Transform Registry:** Nếu không có component này, các thư viện scikit-learn sẽ bị hardcode trực tiếp vào code logic, dẫn đến code rối rắm (spaghetti code) và cực kỳ khó thêm bớt các phương pháp xử lý dữ liệu mới.
- **Pipeline Builder:** Giải quyết bài toán tiền xử lý thủ công. Nếu không có nó, Data Scientist phải tự viết tay hàng chục dòng code để `fit_transform` từng cột, dễ gây rò rỉ dữ liệu (data leakage) giữa tập train/test và gây ác mộng khi đưa lên serving.
- **Schema tự sinh:** Ngăn chặn lỗi "Garbage In, Garbage Out". Nếu không có schema, dữ liệu rác hoặc thiếu cột từ API ở Phase 3 sẽ lọt trực tiếp vào model gây crash hệ thống mà không rõ nguyên nhân.
- **Model Router:** Tránh hội chứng "No Free Lunch". Không có mô hình nào tốt cho mọi bài toán. Nếu không có router, hệ thống sẽ mù quáng chạy các model quá nặng cho data nhỏ hoặc model quá yếu cho data phức tạp.
- **AutoML Optuna:** Tiết kiệm hàng chục giờ tinh chỉnh thủ công. Nếu không có nó, mô hình chỉ chạy với tham số mặc định (default params), dẫn đến kết quả (accuracy/AUC) không bao giờ đạt ngưỡng tối ưu.

---

## 3. CONSTRAINTS — Giới hạn không được vi phạm

- **Không hardcode tên cột:** Tuyệt đối không được nhắc đến tên bất kỳ cột cụ thể nào trong logic của Phase 2. Mọi quyết định phải dựa vào `inferred_role` và thuộc tính của `ColumnProfile`.
- **Pipeline phải serializable (joblib):** Toàn bộ Pipeline (từ bước tiền xử lý đến model) phải được gói gọn trong 1 object duy nhất và lưu được bằng `joblib` để MLflow có thể register và API load lên nguyên vẹn.
- **TEXT column:** Cột có role `TEXT` bắt buộc phải được impute bằng hằng số `""` (chuỗi rỗng) trước khi đưa vào TF-IDF transformer để tránh lỗi.
- **Optuna Trials:** Số lượng `n_trials` chạy tối ưu hóa phải có giá trị mặc định là **15** (để tránh Render sleep/timeout), và có thể ghi đè được thông qua biến môi trường `OPTUNA_N_TRIALS`.
- **Pandera Schema Lifecycle:** Schema tự sinh phải được kích hoạt tại đúng 2 mốc thời gian: 
    1. Ngay sau profiling để khóa cấu trúc dữ liệu trước khi train.
    2. Xuất ra để tái sử dụng làm validation gate lúc Serving (Phase 3).
- **Model Router:** KHÔNG được assume domain (ví dụ: không được `if domain == 'telco'`). Quyết định model phải hoàn toàn dựa trên cấu trúc dataset (shape, class distribution).