from typing import List, Iterator

from minio import Minio, S3Error
from minio.datatypes import Object


class MinioBucketWrapper:
    def __init__(
            self, url: str,
            access_key: str,
            secret_key: str,
            bucket: str
    ) -> None:
        self.client = Minio(
            url,
            access_key=access_key,
            secret_key=secret_key,
            secure=False
        )
        self.bucket = bucket

    def get_obj(self, obj: str) -> str:
        try:
            res = self.client.get_object(self.bucket, obj)

            with open(obj, 'wb') as f:
                for data in res.stream(32 * 1024):
                    f.write(data)

            print(f"{obj} downloaded!")
            return obj

        except S3Error as e:
            print(f"Error occurred: {e}")

    def put_obj(self, name: str, path: str) -> None:
        try:
            self.client.fput_object(self.bucket, name, path)
            print(f"{name} uploaded!")

        except S3Error as e:
            print(f"Error occurred: {e}")

    def list_obj(self) -> list[str]:
        return [o.object_name for o in self.client.list_objects(self.bucket, recursive=True)]
