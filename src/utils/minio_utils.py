from minio import Minio
from minio.error import S3Error

class MinioClient:
    def __init__(self, url, access_key, secret_key, secure=False):
        self.client = Minio(
            url,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure
        )

    def bucket_exists(self, bucket_name):
        return self.client.bucket_exists(bucket_name)

    def download_file(self, bucket_name, object_name, dest_path):
        self.client.fget_object(bucket_name, object_name, dest_path)
        print(f"File '{object_name}' downloaded from bucket '{bucket_name}' to '{dest_path}'.")

    def list_objects(self, bucket_name, prefix=""):
        objects = self.client.list_objects(bucket_name, prefix=prefix)
        return[obj.object_name for obj in objects]
