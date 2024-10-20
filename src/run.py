import argparse

from src.main import minio_init, detect


def main():
    client = minio_init()

    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('-w', '--weights')

    args = parser.parse_args()

    # model = _train_model(client, weights='yolo11n.pt', amt=10)

    detect(client, args.filename, args.weights)


if __name__ == '__main__':
    main()
