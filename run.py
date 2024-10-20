import argparse

from src.main import minio_init, detect


if __name__ == '__main__':
    client = minio_init()

    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('-w', '--weights')

    args = parser.parse_args()

    # model = _train_model(client, weights='yolo11n.pt', amt=10)

    detect(client, args.filename, args.weights)