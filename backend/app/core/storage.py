import os
from pathlib import Path
import boto3
from botocore.config import Config

from .config import (
    STORAGE_MODE,
    S3_ENDPOINT_URL,
    S3_ACCESS_KEY_ID,
    S3_SECRET_ACCESS_KEY,
    S3_REGION,
    S3_BUCKET_NAME,
)

class StorageClient:
    """
    Unified storage client integrating with S3/Cloudflare R2 and Local File System fallback.
    Designed to serve the Data Lake Architectural Zones.
    """

    def __init__(self):
        self.mode = STORAGE_MODE.lower()
        self.bucket = S3_BUCKET_NAME
        self.local_base_path = Path("data")
        
        if self.mode == "s3":
            self.s3 = boto3.client(
                's3',
                endpoint_url=S3_ENDPOINT_URL,
                aws_access_key_id=S3_ACCESS_KEY_ID,
                aws_secret_access_key=S3_SECRET_ACCESS_KEY,
                config=Config(signature_version='s3v4'),
                region_name=S3_REGION
            )
        else:
            self.s3 = None

    def get_uri(self, path: str) -> str:
        """
        Returns the full URI for integration with Pandas (s3fs) or PySpark.
        Creates parent directories automatically if in local fallback mode.
        """
        path = path.lstrip("/")
        if self.mode == "s3":
            return f"s3://{self.bucket}/{path}"
        else:
            full_path = (self.local_base_path / path).resolve()
            full_path.parent.mkdir(parents=True, exist_ok=True)
            return str(full_path)

    def upload_file(self, file_content: bytes, path: str):
        """Upload raw bytes to the specified path within the Data Lake."""
        path = path.lstrip("/")
        if self.mode == "s3":
            self.s3.put_object(Bucket=self.bucket, Key=path, Body=file_content)
        else:
            full_path = self.local_base_path / path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            with open(full_path, "wb") as f:
                f.write(file_content)

    def download_file(self, path: str) -> bytes:
        """Download a file as bytes from the Data Lake."""
        path = path.lstrip("/")
        if self.mode == "s3":
            response = self.s3.get_object(Bucket=self.bucket, Key=path)
            return response['Body'].read()
        else:
            full_path = self.local_base_path / path
            if not full_path.exists():
                raise FileNotFoundError(f"Local file not found: {full_path}")
            with open(full_path, "rb") as f:
                return f.read()

    def generate_presigned_url(self, path: str, expiration: int = 3600, method: str = 'put_object') -> str:
        """Generate a presigned URL to allow direct frontend uploads/downloads via CORS."""
        path = path.lstrip("/")
        if self.mode == "s3":
            return self.s3.generate_presigned_url(
                ClientMethod=method,
                Params={'Bucket': self.bucket, 'Key': path},
                ExpiresIn=expiration
            )
        else:
            # Fallback for local testing (can be intercepted by a local mock endpoint if necessary)
            return f"http://localhost:8000/mock-presigned?path={path}"

    def delete_prefix(self, prefix: str):
        """Delete all objects with the specified prefix (simulating folder deletion)."""
        prefix = prefix.lstrip("/")
        if self.mode == "s3":
            paginator = self.s3.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.bucket, Prefix=prefix)
            delete_keys = []
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        delete_keys.append({'Key': obj['Key']})
            
            for i in range(0, len(delete_keys), 1000):
                self.s3.delete_objects(
                    Bucket=self.bucket,
                    Delete={'Objects': delete_keys[i:i+1000]}
                )
        else:
            full_path = self.local_base_path / prefix
            import shutil
            if full_path.exists():
                if full_path.is_dir():
                    shutil.rmtree(full_path)
                else:
                    full_path.unlink()

    def delete_file(self, path: str) -> bool:
        path = path.lstrip("/")
        if self.mode == "local":
            full_path = (self.local_base_path / path).resolve()
            if full_path.exists():
                os.remove(full_path)
                return True
            return False
        else:
            self.s3.delete_object(
                Bucket=self.bucket,
                Key=path
            )
            return True

    def ping(self) -> bool:
        if self.mode == "local":
            return self.local_base_path.exists()
        else:
            self.s3.list_objects_v2(
                Bucket=self.bucket,
                MaxKeys=1
            )
            return True

    def list_files(self, prefix: str = "") -> list:
        prefix = prefix.lstrip("/")
        if self.mode == "s3":
            paginator = self.s3.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.bucket, Prefix=prefix)
            keys = []
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        keys.append(obj['Key'])
            return keys
        else:
            full_path = self.local_base_path / prefix
            if not full_path.exists():
                return []
            if not full_path.is_dir():
                return [prefix] if full_path.exists() else []
            keys = []
            import os
            for root, _, files in os.walk(full_path):
                for f in files:
                    file_p = Path(root) / f
                    rel_p = file_p.relative_to(self.local_base_path)
                    keys.append(str(rel_p).replace("\\", "/"))
            return keys

storage = StorageClient()
