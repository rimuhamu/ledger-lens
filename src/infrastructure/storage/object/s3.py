import os
import boto3
from botocore.exceptions import ClientError
from src.infrastructure.storage.object.base import ObjectStore
from src.utils.logger import get_logger

class S3ObjectStore(ObjectStore):
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.bucket_name = os.getenv("S3_BUCKET_NAME", "ledgerlens-documents")
        
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION", "us-east-1"),
            endpoint_url=os.getenv("AWS_ENDPOINT_URL")
        )
        self._create_bucket_if_not_exists()

    def _create_bucket_if_not_exists(self):
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
        except ClientError:
            try:
                region = os.getenv("AWS_REGION", "us-east-1")
                if region == "us-east-1":
                    self.s3_client.create_bucket(Bucket=self.bucket_name)
                else:
                    self.s3_client.create_bucket(
                        Bucket=self.bucket_name,
                        CreateBucketConfiguration={'LocationConstraint': region}
                    )
                self.logger.info(f"Created S3 bucket: {self.bucket_name}")
            except ClientError as e:
                self.logger.error(f"Failed to create bucket: {e}")

    def upload_file(self, file_path: str, key: str) -> str:
        try:
            self.s3_client.upload_file(file_path, self.bucket_name, key)
            self.logger.info(f"Uploaded to S3: s3://{self.bucket_name}/{key}")
            return key
        except ClientError as e:
            self.logger.error(f"S3 upload failed: {e}")
            raise

    def download_file(self, key: str, local_path: str) -> str:
        try:
            self.s3_client.download_file(self.bucket_name, key, local_path)
            self.logger.info(f"Downloaded from S3: {key} -> {local_path}")
            return local_path
        except ClientError as e:
            self.logger.error(f"S3 download failed: {e}")
            raise

    def delete_file(self, key: str) -> None:
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=key)
            self.logger.info(f"Deleted from S3: {key}")
        except ClientError as e:
            self.logger.error(f"Failed to delete from S3: {e}")

    def save_json(self, data: dict, key: str) -> str:
        import json
        try:
            json_str = json.dumps(data)
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=json_str,
                ContentType='application/json'
            )
            self.logger.info(f"Saved JSON to S3: {key}")
            return key
        except ClientError as e:
            self.logger.error(f"Failed to save JSON to S3: {e}")
            raise

    def get_json(self, key: str) -> dict:
        import json
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            content = response['Body'].read().decode('utf-8')
            return json.loads(content)
        except ClientError as e:
            self.logger.error(f"Failed to get JSON from S3: {e}")
            return {}
