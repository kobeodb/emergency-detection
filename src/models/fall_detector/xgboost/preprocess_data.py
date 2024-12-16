
import pandas as pd
from pathlib import Path

current_file = Path().resolve()
models_dir = current_file
print(models_dir)
while models_dir.name != "models" and models_dir != models_dir.parent:
    models_dir = models_dir.parent

csv_name = 'fall_pose_keypoint.csv'
print(models_dir)
csv_path = Path(models_dir / "classifiers" / "rf_classifier" / "pose" / "yolo_pose" / csv_name)

df = pd.read_csv(csv_path)
print("loading csv")
df = df.drop('image_name', axis=1)
df.head()