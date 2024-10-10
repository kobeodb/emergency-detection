from util import process_videos
from yolov5 import train, detect

if __name__ == '__main__':
    videos = [...]
    out = '../out/dataset/frames'
    data = './data/data.yaml'

    process_videos(videos, out, frame_rate=5)

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