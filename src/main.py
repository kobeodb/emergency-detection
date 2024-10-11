import dotenv

from util import process_videos
from yolov5 import train, detect

from data.db.main import MinioBucketWrapper

# TODO:
# - Get Minio started by entering password
# - Find videos and images to get frames and export them
# - Check entire codebase
if __name__ == '__main__':
    dotenv.load_dotenv()

    minio_url = dotenv.get_key("MINIO_URL")
    minio_user = dotenv.get_key("MINIO_USER")
    minio_password = dotenv.get_key("MINIO_PASSWORD")
    minio_bucket_name = dotenv.get_key("MINIO_BUCKET_NAME")

    client = MinioBucketWrapper(
        minio_url,
        minio_user,
        minio_password,
        minio_bucket_name
    )

    out = '../out/dataset/frames'
    data = './data/data.yaml'

    glob = client.list_obj()
    videos = [client.get_obj(v) for v in glob]

    process_videos(videos, out, frame_rate=3)

    train.run(
        data_yaml=data,
        img_size=640,
        batch_size=16,
        epochs=100,
        weights='yolov5s.pt'
    )

    detect.run(
        video_source='test_video.mp4',
        weights_path='runs/train/exp/weights/best.pt',
        img_size=640,
        output_dir='runs/detect'
    )
