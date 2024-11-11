# Bot Brigade - Eyes4Resque

## How to run the application:
1. Clone this repository to your computer
2. install the requirements
   ```
   pip install -r requirements
   ```
3. Navigate to the file ../src/application/improved_app.py
4. When the app is running and open, you will have to choose your yolo model weights (these are not included in the repository).
5. After choosing the weights, you'll have to select a video (of a person) you want to use for the detection
6. After choosing the video, the detection should start automatically ->  if not, press the start detection button.

## How does it work
1. When the video is playing inside the app, you'll see a box around the person in the video. Above the box there will be a label "Fall Detected", "Not Fall", "Sitting", "Walking" depending on what the person is doing. The models confidence will also be displayed next to the label
2. When a fall is detected, a timer will start
3. After the timer hits 4 seconds, there will be another label, either "Need Help" or "Does not need help"
4. When the "Need Help" label is shown, this is when the alert should be sent to emergency services.

## Components used
- Fine-Tuned YOLOv11 weights
- Implemented Motion Tracking for post fall behavior
- MediaPipe (Keypoint Extraction)
- XGBoost (Classifier using the extracted Keypoints)
  
