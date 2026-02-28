import os
import s3fs


def upload_to_minio():
    # 1. MinIO connection configuration
    fs = s3fs.S3FileSystem(
        client_kwargs={'endpoint_url': 'http://localhost:9000'},
        key='admin',
        secret='password',
        use_listings_cache=False
    )

    # 2. Paths
    # Get project root directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Local file (created by csv_to_parquet step)
    local_path = os.path.join(
        base_dir, 'data', 'parquet', 'raw', 'telco_churn.parquet')

    # Destination on MinIO
    s3_path = 's3://datalake/raw/telco_churn.parquet'

    print(f"⏳ Uploading from: {local_path}")
    print(f"➡️ To: {s3_path}")

    try:
        if not os.path.exists(local_path):
            print("❌ Error: Local file not found! Have you run 'csv_to_parquet.py'?")
            return

        # Upload
        fs.put(local_path, s3_path)
        print("✅ Upload successful!")

        # Kiểm tra lại xem file có tồn tại không
        if fs.exists(s3_path):
            print(f"🔍 Confirmed file exists on MinIO: {s3_path}")
            print(f"📦 Size: {fs.info(s3_path)['size']} bytes")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    upload_to_minio()
