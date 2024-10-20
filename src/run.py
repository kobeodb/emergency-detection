import argparse

from src.main import minio_init, detect


def main():
    client = minio_init()

    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str, help='Path to video file on Minio to detect people on')
    parser.add_argument('-w', '--weights', type=str, help='Weights file for training the YOLO model')

    args = parser.parse_args()
    # model = _train_model(client, weights='yolo11n.pt', amt=10)

    detect(client, args.filename, args.weights)


if __name__ == '__main__':
    main()
