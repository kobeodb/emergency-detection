import os
from ultralytics import YOLO


def initialize_model(weights: str) -> YOLO:
    """Initialize YOLO model with specified weights."""
    return YOLO(weights)


def evaluate_model(weights: str, data_yaml: str, img_size: int = 640):
    """Evaluate the YOLO model on the test dataset and calculate Accuracy."""
    model = initialize_model(weights)
    metrics = model.val(data=data_yaml, imgsz=img_size, split='test')

    # Get overall metrics
    precision = metrics.box.mp  # Mean Precision
    recall = metrics.box.mr     # Mean Recall
    map50 = metrics.box.map50   # mAP@0.5
    map5095 = metrics.box.map   # mAP@0.5:0.95

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"mAP@0.5: {map50:.4f}")
    print(f"mAP@0.5-95: {map5095:.4f}")




if __name__ == "__main__":

    data_yaml_path = os.path.abspath("../data/data.yaml")
    weights_path = os.path.abspath("../../runs/detect/train7/weights/best.pt")


    # Evaluate the model
    print("\nEvaluating the model on the test set:")
    evaluate_model(weights=weights_path, data_yaml=data_yaml_path)