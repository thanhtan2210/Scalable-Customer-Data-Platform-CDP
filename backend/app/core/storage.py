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

storage = StorageClient()
