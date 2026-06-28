import os
import sys
import uuid
import mlflow

# Tự động thêm thư mục gốc vào PYTHONPATH
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, root_dir)

try:
    import backend.app.core.config as config
    
    # Lấy thông tin từ config (cho phép None)
    uri = getattr(config, "MLFLOW_TRACKING_URI", None)
    username = getattr(config, "MLFLOW_TRACKING_USERNAME", None)
    password = getattr(config, "MLFLOW_TRACKING_PASSWORD", None)
    
    # Kiểm tra xem người dùng đã điền .env chưa
    if not username or "your-dagshub-username" in username:
        print("⚠️ Hướng dẫn: Bạn chưa cấu hình MLFLOW_TRACKING_USERNAME trong file .env")
        print("👉 Hãy mở file .env và điền tên đăng nhập DagsHub của bạn.")
        sys.exit(0)
        
    if not password or "your-dagshub-token" in password:
        print("⚠️ Hướng dẫn: Bạn chưa cấu hình MLFLOW_TRACKING_PASSWORD (Token) trong file .env")
        print("👉 Hãy lấy Token từ DagsHub Settings -> Tokens và dán vào .env.")
        sys.exit(0)

except Exception as e:
    print(f"❌ Lỗi hệ thống: {e}")
    sys.exit(1)

def test_mlflow_foundation():
    print("🚀 Starting MLflow Remote Tracking Validation (Phase 0)...")
    
    # Cấu hình xác thực cho MLflow
    os.environ['MLFLOW_TRACKING_USERNAME'] = username
    os.environ['MLFLOW_TRACKING_PASSWORD'] = password
    
    print(f"🔗 Tracking URI: {uri}")
    mlflow.set_tracking_uri(uri)
    
    # Tạo Experiment ID ngẫu nhiên
    dataset_id = f"infrastructure-test-{str(uuid.uuid4())[:4]}"
    
    try:
        print(f"🧪 Gửi dữ liệu test lên DagsHub (Experiment: {dataset_id})...")
        mlflow.set_experiment(dataset_id)
        
        with mlflow.start_run(run_name="Phase 0 Check"):
            mlflow.log_param("status", "success")
            mlflow.log_metric("latency_ms", 150.0)
            print("✅ Đã ghi nhận (log) tham số thành công!")
            
        print("\n🎯 KẾT NỐI MLFLOW CLOUD THÀNH CÔNG!")
        print(f"🔗 Xem kết quả tại: {uri.replace('.mlflow', '/experiments')}")
        
    except Exception as e:
        print(f"❌ Kết nối thất bại: {str(e)}")
        print("\n💡 Gợi ý: Kiểm tra xem link MLFLOW_TRACKING_URI trong .env có đúng đuôi .mlflow không.")

if __name__ == "__main__":
    test_mlflow_foundation()
