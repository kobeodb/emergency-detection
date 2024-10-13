import dotenv
import os
from util import process_videos
from yolov5 import train, detect
import logging
import cv2

from data.db.main import MinioBucketWrapper

# TODO: Find videos and images to get frames and export them
# TODO: Test our system
if __name__ == '__main__':
    logging.basicConfig(filename='../log/app.log', level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    dotenv.load_dotenv()

    minio_url = os.getenv("MINIO_URL")
    minio_user = os.getenv("MINIO_USER")
    minio_password = os.getenv("MINIO_PASSWORD")
    minio_bucket_name = os.getenv("MINIO_BUCKET_NAME")

    client = MinioBucketWrapper(
        minio_url,
        minio_user,
        minio_password,
        minio_bucket_name
    )

    path = '../out/dataset/train'
    out = '../out/dataset/frames'
    data = './data/data.yaml'

    # for v in os.listdir(put):
    #     client.put_obj(v, put + '/' + v)

    if not os.listdir(path):
        videos = [client.get_obj(o)
                  for o in client.list_obj()]
    else:
        videos = os.listdir(path)

    process_videos([path + '/' + v for v in videos], out, frame_rate=5)

    # train.run(
    #     data_yaml=data,
    #     img_size=640,
    #     batch_size=16,
    #     epochs=50,
    #     weights='yolov5s.pt'
    # )
    #
    # detect.run(
    #     video_source=test,
    #     weights_path='runs/train/exp/weights/best.pt',
    #     img_size=640,
    #     output_dir='../datasets/detect'
    # )
