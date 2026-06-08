import os
import boto3
from botocore.config import Config

class R2Storage:
    def __init__(self):
        self.endpoint_url = os.getenv("S3_ENDPOINT_URL", "http://localhost:9000")
        self.access_key = os.getenv("S3_ACCESS_KEY", "admin")
        self.secret_key = os.getenv("S3_SECRET_KEY", "password")
        self.bucket_name = os.getenv("S3_BUCKET", "churn-assets")
        
        self.s3 = boto3.client(
            's3',
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            config=Config(signature_version='s3v4'),
            region_name=os.getenv("S3_REGION", "auto")
        )

    def upload_file(self, file_content: bytes, path: str):
        self.s3.put_object(Bucket=self.bucket_name, Key=path, Body=file_content)

    def download_file(self, path: str) -> bytes:
        response = self.s3.get_object(Bucket=self.bucket_name, Key=path)
        return response['Body'].read()

storage = R2Storage()
