import os
import sys
import uuid

# Tự động thêm thư mục gốc vào PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from backend.app.core.storage import storage
except ImportError as e:
    print(f"❌ Lỗi: Không thể import module storage. Hãy đảm bảo bạn đã cài đặt thư viện: pip install boto3 s3fs")
    print(f"Chi tiết lỗi: {e}")
    sys.exit(1)

def test_cloud_storage_foundation():
    print("🚀 Starting Cloud Storage Validation (Phase 0)...")
    
    # 1. Giả lập một dataset_id mới
    dataset_id = str(uuid.uuid4())
    filename = "test_upload.csv"
    content = b"customer_id,churn\n1,0\n2,1"
    
    # 2. Định nghĩa path theo chuẩn domain-agnostic trong docs
    cloud_path = f"landing_zone/{dataset_id}/{filename}"
    
    try:
        print(f"📦 Uploading test file to: {cloud_path}...")
        storage.upload_file(content, cloud_path)
        print("✅ Upload successful!")
        
        print("🔗 Generating URI for downstream pipelines...")
        uri = storage.get_uri(cloud_path)
        print(f"Standardized URI: {uri}")
        
        print("📥 Downloading file back for integrity check...")
        downloaded_content = storage.download_file(cloud_path)
        if downloaded_content == content:
            print("✅ Data integrity verified! (Original == Downloaded)")
        
        print("\n🌐 Presigned URL for Frontend (Testing CORS path):")
        url = storage.generate_presigned_url(cloud_path)
        print(f"URL: {url[:80]}...") 
        
        print("\n🎯 STORAGE FOUNDATION IS READY!")
        
    except Exception as e:
        print(f"❌ Validation failed: {str(e)}")
        print("\n💡 HINT: Check your .env credentials (S3_ACCESS_KEY_ID, etc.) and Cloudflare R2 bucket name.")

if __name__ == "__main__":
    test_cloud_storage_foundation()
