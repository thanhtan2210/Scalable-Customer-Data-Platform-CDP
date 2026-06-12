import os
import sys
import uuid
import mlflow

# Tự động thêm thư mục gốc vào PYTHONPATH
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_dir)

print(f"DEBUG: PYTHONPATH: {sys.path[:3]}")

try:
    import backend.app.core.config as config
    print(f"DEBUG: Config file location: {config.__file__}")
    
    MLFLOW_TRACKING_URI = getattr(config, "MLFLOW_TRACKING_URI", None)
    MLFLOW_TRACKING_USERNAME = getattr(config, "MLFLOW_TRACKING_USERNAME", None)
    MLFLOW_TRACKING_PASSWORD = getattr(config, "MLFLOW_TRACKING_PASSWORD", None)
    
    if MLFLOW_TRACKING_USERNAME is None:
        raise AttributeError("MLFLOW_TRACKING_USERNAME not found in config module")
        
except Exception as e:
    print(f"❌ Lỗi: {e}")
    sys.exit(1)

def test_mlflow_foundation():
    print("🚀 Starting MLflow Remote Tracking Validation (Phase 0)...")
    
    # 1. Cấu hình xác thực DagsHub
    if MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD:
        os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_TRACKING_USERNAME
        os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_TRACKING_PASSWORD
    
    print(f"🔗 Tracking URI: {MLFLOW_TRACKING_URI}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # 2. Tạo một Experiment giả lập theo dataset_id
    dataset_id = f"test-dataset-{str(uuid.uuid4())[:8]}"
    print(f"🧪 Creating test experiment: {dataset_id}...")
    
    try:
        mlflow.set_experiment(dataset_id)
        
        with mlflow.start_run(run_name="Phase 0 Validation"):
            print("📝 Logging test parameters and metrics...")
            mlflow.log_param("infrastructure_mode", "remote_dagshub")
            mlflow.log_metric("connection_status", 1.0)
            
            print("🏷️ Setting industry tag...")
            mlflow.set_tag("industry", "infrastructure_test")
            
            print("✅ Successfully logged data to DagsHub!")
            
        print("\n🎯 MLFLOW FOUNDATION IS READY!")
        print(f"🔗 View your experiment at: {MLFLOW_TRACKING_URI.replace('.mlflow', '/experiments')}")
        
    except Exception as e:
        print(f"❌ Validation failed: {str(e)}")
        print("\n💡 HINT: Check your MLFLOW_TRACKING_URI and DagsHub Token in .env")

if __name__ == "__main__":
    test_mlflow_foundation()
