import os
import sys
import pandas as pd
import numpy as np
import uuid

# Tự động thêm thư mục gốc vào PYTHONPATH
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, root_dir)

from backend.app.core.profiler.orchestrator import run_profiling
from backend.app.core.training.automl import run_automl
from backend.app.core.config import MLFLOW_TRACKING_URI

# Ép Optuna chạy nhanh (3 trials) để test logic thay vì chờ lâu
os.environ["OPTUNA_N_TRIALS"] = "3"

def validate_automl():
    print("🚀 Khởi động bài Test: AutoML + Optuna HPO + MLflow...")
    print(f"🔗 MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
    
    # 1. Tạo một dataset giả lập bài toán Churn khó
    print("\n📦 1. Đang tạo Mock Dataset (1000 dòng, 10 cột)...")
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        "customer_id": [f"C{i}" for i in range(n)],
        "age": np.random.randint(18, 80, n),
        "monthly_spend": np.random.uniform(10.0, 200.0, n),
        "tenure_months": np.random.randint(1, 72, n),
        "contract_type": np.random.choice(["Month-to-month", "One year", "Two year"], n),
        "internet_service": np.random.choice(["Fiber optic", "DSL", "No"], n),
        "churn": np.random.choice([0, 1], n, p=[0.7, 0.3]) # Target mất cân bằng 70/30
    })
    
    # 2. Chạy Profiling để lấy Contract
    print("\n🔍 2. Đang chạy Profiling Engine để chốt Data Contract...")
    profiles, target_col = run_profiling(df)
    print(f"   -> Đã chốt Target: '{target_col}'")
    
    # 3. Kích hoạt AutoML
    dataset_id = f"test-automl-{str(uuid.uuid4())[:6]}"
    print(f"\n🧠 3. Bắt đầu AutoML Sweep cho Dataset: {dataset_id}")
    print(f"   -> Sẽ chạy Optuna HPO + Cross-Validation...")
    print(f"   -> Sẽ tự động log mọi Trial lên MLflow...")
    
    try:
        model_uri, schema_path = run_automl(df, profiles, target_col, dataset_id)
        
        print("\n✅ KẾT QUẢ AUTOML:")
        print(f"🏆 Best Model URI: {model_uri}")
        print(f"📄 Schema Path lưu tại: {schema_path}")
        
        print("\n🎯 BẠN HÃY KIỂM TRA TRÊN DAGSHUB:")
        print("1. Mở trang DagsHub của bạn -> Tab Experiments.")
        print("2. Bạn sẽ thấy các Run mới.")
        print("3. Bấm vào Run tốt nhất, bạn sẽ thấy:")
        print("   - Đã log Metric 'best_roc_auc' (Cross-Validated).")
        print("   - Đã log bộ Parameters chiến thắng của Optuna.")
        print("   - Đã đóng gói (Log Model) toàn bộ scikit-learn Pipeline!")
        
    except Exception as e:
        print(f"\n❌ Lỗi quá trình AutoML: {e}")

if __name__ == "__main__":
    validate_automl()
