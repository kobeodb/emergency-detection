from __future__ import annotations

import re
from typing import Iterator

from minio import Minio, S3Error
import os


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

    def get_obj_file(self, obj: str, path: str = '.') -> tuple[str, str]:
        try:
            res = self.client.get_object(self.bucket, obj)

            with open(os.path.join(path, obj.split('/')[-1]), 'wb') as f:
                for data in res.stream(32 * 1024):
                    f.write(data)

            return obj, f"{obj} downloaded!"

        except S3Error as e:
            print(f"Error occurred: {e}")

    def get_obj_bytes(self, obj: str) -> Iterator[bytes]:
        res = None
        try:
            res = self.client.get_object(self.bucket, obj)
            return res.stream()

        except S3Error as e:
            print(f"Error occurred: {e}")

        finally:
            res.close()
            res.release_conn()

    def put_obj(self, name: str, path: str) -> str:
        try:
            self.client.fput_object(self.bucket, name, path)
            return f"{name} uploaded!"

        except S3Error as e:
            print(f"Error occurred: {e}")

    def list_obj(self) -> list[str]:
        return [o.object_name for o in self.client.list_objects(self.bucket, recursive=True)]

    def del_obj(self, pattern: str) -> str:
        try:
            objects = self.list_obj()

            for obj in objects:
                if re.match(pattern, obj):
                    self.client.remove_object(self.bucket, obj)

            return f"{len(objects) - len(self.list_obj())} deleted objects!"

        except S3Error as e:
            print(f"Error occurred: {e}")