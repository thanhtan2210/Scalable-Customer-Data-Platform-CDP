import os
import yaml
# pyrefly: ignore [missing-import]
import pytest
from pathlib import Path

# Thư mục gốc dự án
ROOT_DIR = Path(__file__).resolve().parents[2]

def test_docker_compose():
    """
    Kiểm tra cấu hình Docker Compose:
    - File docker-compose.yml tồn tại ở root
    - Có đủ 5 service: db, minio, mlflow, backend, analytics
    - Mỗi service có healthcheck block
    """
    docker_compose_path = ROOT_DIR / "docker-compose.yml"
    assert docker_compose_path.exists(), "File docker-compose.yml không tồn tại ở thư mục root!"
    
    with open(docker_compose_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    assert "services" in config, "Không tìm thấy phần 'services' trong docker-compose.yml!"
    services = config["services"]
    
    expected_services = ["db", "minio", "mlflow", "backend", "analytics"]
    for service_name in expected_services:
        assert service_name in services, f"Thiếu service '{service_name}' trong docker-compose.yml!"
        assert "healthcheck" in services[service_name], f"Service '{service_name}' không có cấu hình 'healthcheck'!"

def test_storage_client():
    """
    Kiểm tra StorageClient:
    - Import backend.app.core.storage.StorageClient không lỗi
    - Khởi tạo với STORAGE_MODE=local không raise exception
    - Hàm upload_file() và download_file() tồn tại
    """
    # Thiết lập environment variable để mock STORAGE_MODE
    os.environ["STORAGE_MODE"] = "local"
    
    try:
        import backend.app.core.config as config
        config.STORAGE_MODE = "local"
        import backend.app.core.storage as storage_mod
        storage_mod.STORAGE_MODE = "local"
        from backend.app.core.storage import StorageClient, storage
    except ImportError as e:
        pytest.fail(f"Không thể import StorageClient: {e}")
        
    client = StorageClient()
    assert client.mode == "local", "StorageClient mode không phải là local!"
    
    assert hasattr(client, "upload_file"), "Hàm upload_file() không tồn tại trong StorageClient!"
    assert hasattr(client, "download_file"), "Hàm download_file() không tồn tại trong StorageClient!"

def test_config():
    """
    Kiểm tra Config:
    - Tất cả env var sau đọc được từ .env.example:
      S3_ENDPOINT_URL, S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY,
      S3_BUCKET_NAME, DATABASE_URL, MLFLOW_TRACKING_URI
    - Không có giá trị nào hardcode URI cụ thể trong config.py
    """
    env_example_path = ROOT_DIR / ".env.example"
    assert env_example_path.exists(), "File .env.example không tồn tại!"
    
    with open(env_example_path, "r", encoding="utf-8") as f:
        env_content = f.read()
        
    required_env_vars = [
        "S3_ENDPOINT_URL",
        "S3_ACCESS_KEY_ID",
        "S3_SECRET_ACCESS_KEY",
        "S3_BUCKET_NAME",
        "DATABASE_URL",
        "MLFLOW_TRACKING_URI"
    ]
    
    for var in required_env_vars:
        assert f"{var}=" in env_content, f"Thiếu env var '{var}' trong .env.example!"
        
    config_path = ROOT_DIR / "backend" / "app" / "core" / "config.py"
    assert config_path.exists(), "File config.py không tồn tại!"
    
    with open(config_path, "r", encoding="utf-8") as f:
        config_content = f.read()
        
    # Đảm bảo các biến được đọc thông qua os.getenv (hoặc tương đương) chứ không gán cứng URI trực tiếp
    # Một cách kiểm tra là xem các giá trị gán cho các biến cấu hình này có sử dụng os.getenv
    for var in ["S3_ENDPOINT_URL", "DATABASE_URL", "MLFLOW_TRACKING_URI"]:
        assert f"os.getenv(\"{var}\"" in config_content or f"os.getenv('{var}'" in config_content or f"os.environ.get(\"{var}\"" in config_content or f"os.environ.get('{var}'" in config_content, f"Biến cấu hình {var} không được đọc qua os.getenv trong config.py!"

def test_mlflow_switch():
    """
    Kiểm tra MLflow switch:
    - backend/app/core/training/mlflow_utils.py không rỗng
    - Có hàm setup_mlflow() đọc MLFLOW_TRACKING_URI từ env var
    - Không có string "file:///" hardcode trong file này
    """
    mlflow_utils_path = ROOT_DIR / "backend" / "app" / "core" / "training" / "mlflow_utils.py"
    assert mlflow_utils_path.exists(), "File mlflow_utils.py không tồn tại!"
    
    # Kiểm tra không rỗng
    assert mlflow_utils_path.stat().st_size > 0, "File mlflow_utils.py hoàn toàn rỗng!"
    
    with open(mlflow_utils_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Kiểm tra hàm setup_mlflow
    assert "def setup_mlflow" in content, "Hàm setup_mlflow() không tồn tại trong mlflow_utils.py!"
    
    # Kiểm tra đọc MLFLOW_TRACKING_URI từ env var
    assert "MLFLOW_TRACKING_URI" in content, "Không đọc MLFLOW_TRACKING_URI từ env var trong mlflow_utils.py!"
    
    # Kiểm tra không có string "file:///"
    assert "file:///" not in content, "Phát hiện chuỗi 'file:///' được hardcode trong mlflow_utils.py!"
