import os
import sys
from minio import Minio
from minio.error import S3Error
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common.config import MINIO_IP, MINIO_PORT, MINIO_USER, MINIO_PASSWORD


class MinioService:
    def __init__(self):
        self.client = Minio(
            f"{MINIO_IP}:{MINIO_PORT}",
            access_key=MINIO_USER,
            secret_key=MINIO_PASSWORD,
            secure=False
        )
    
    def ensure_bucket_exists(self, bucket_name):
        """确保存储桶存在"""
        if not self.client.bucket_exists(bucket_name):
            self.client.make_bucket(bucket_name)
    
    def upload_file(self, bucket_name, object_name, file_path, content_type=None):
        """上传文件到MinIO"""
        # 确保存储桶存在
        self.ensure_bucket_exists(bucket_name)
        
        # 上传文件
        try:
            self.client.fput_object(
                bucket_name,
                object_name,
                file_path,
                content_type=content_type
            )
            return True
        except S3Error as e:
            print(f"Error uploading file: {e}")
            return False
    
    def get_presigned_url(self, bucket_name, object_name):
        """获取预签名URL"""
        try:
            return self.client.presigned_get_object(bucket_name, object_name)
        except S3Error as e:
            print(f"Error getting presigned URL: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error in get_presigned_url: {e}")
            return None
    
    def download_file(self, bucket_name, object_name, file_path):
        """从MinIO下载文件"""
        try:
            self.client.fget_object(bucket_name, object_name, file_path)
            return True
        except S3Error as e:
            print(f"Error downloading file: {e}")
            return False