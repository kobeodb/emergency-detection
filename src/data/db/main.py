import tempfile

import cv2
import dotenv
from minio import Minio, S3Error

dotenv.load_dotenv()

minio_url = dotenv.get_key("MINIO_URL")
minio_user = dotenv.get_key("MINIO_USER")
minio_password = dotenv.get_key("MINIO_PASSWORD")
minio_bucket_name = dotenv.get_key("MINIO_BUCKET_NAME")

minio = Minio(
    minio_url,
    access_key=minio_user,
    secret_key=minio_password,
    secure=False
)

obj = "path/to/object"

temp = tempfile.NamedTemporaryFile(delete=False)
try:
    data = minio.get_object(minio_bucket_name, obj)
    temp.write(data.read())
    temp.close()

    video = cv2.VideoCapture(temp.name)

except S3Error as exc:
    print(f"Error occurred: {exc}")

