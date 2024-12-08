# Bot Brigade - Eyes4Resque

## General Architecture + Ideas
***

### Detect and track people.
We accomplished this task by using [YOLOv11 from ultralytics](https://docs.ultralytics.com/models/yolo11/). Ultralytics offers tools for object detection and tracking. When processing a video with these trackers, they generate a bounding box for each tracked individual in every frame. 


### Detect a fall (or a person laying on the ground).
For the detection of a fall we fine-tuned a yolov11n model. this fine-tuned model predicts if a person is on the floor or not. The fine-tuning is done by using a custom set of training data which contain images and labels of people on the floor and people not on the floor.

However, we have observed that the fine-tuned model is not performing as accurately as expected. Despite retraining multiple models on enhanced datasets, the performance has reached a plateau, showing no further improvement. This presents a significant challenge as we aim to achieve optimal performance.

To address this, we are exploring alternative approaches for fall detection. Our current ideas include:
- Lightweight Classifier: Developing a classifier specifically trained to distinguish between individuals lying on the ground and those who are not. Running this classifier on every (third) frame could provide an effective solution.
- Custom Logic Using Bounding Box Coordinates: Designing a rule-based approach leveraging the coordinates of the bounding boxes over time to detect falls. This method is also under active consideration.

We think making our own logic would be more robust than training a classifier for this specific task. Therefore we will implement the custom logic first and evaluate the performance. If we are not happy with the performance, and we are out of idea's for enhancing the logic, then we will implement the classifier


### Motion detection of people that fell on the ground (or lay down).
Detecting individuals falling to the ground is not sufficient to trigger a 112 alert, as the person may simply stand up and walk away afterward. In contrast, individuals who have suffered a cardiac arrest or stroke typically remain motionless after falling, requiring immediate assistance.

To check for motion after a person is detected as being on the floor, we implemented a [simple motion tracker](https://medium.com/@adeevmardia/building-a-simple-motion-detector-with-python-and-opencv-916a39a4f2cb).

he process involves converting frames to grayscale and applying blurring to minimize noise. Each frame is then compared to a static background to detect differences. These detected changes are binarized, and contours are extracted to identify regions of motion.

This simple yet effective method allows real-time detection and logging of motion events, making it ideal for surveillance applications.

### Detection of motionless people that fell on the floor and effectively need help.
Detecting whether an individual is motionless on the ground is not sufficient to trigger a 112 alert, as it’s possible for someone to be on the ground (in the grass taking a nap) without motion without.
Our goal is to differentiate between situations where motionless individuals are fine and those where they are experiencing an emergency.

So far, we have explored 2 techniques:
- We used MediaPipe to extract pose keypoints, which were then input into an XGBoost classifier trained on keypoints from images of individuals both fine and in need of assistance. While this approach produced decent results, we found that MediaPipe's keypoint extraction lacked sufficient accuracy.
- We shifted focus to a 2D CNN built from scratch. The network takes cropped frames from bounding boxes as input, using three RGB channels. This classifier was trained on the same dataset and has shown reasonable performance during evaluation. However, it still occasionally fails to classify correctly.

To improve upon these methods, we are revisiting pose keypoint extraction. This time, the extracted keypoints will be used as input to a random forest classifier.

Our plan is to compare the performance of the 2D CNN and the pose-based random forest classifier. Based on the results, we will decide which approach to refine further.

## Results
For a set of 23 videos

## The sources
***

## How to run scripts and reproduce the results
***

# Open Issues
***

## How to run the inference
- clone this repository
- install the requirements
- make sure the config.yaml file contains the right paths
- navigate to src/pipeline/inference.py
- at the bottom of this file make sure the paths are correct and choose a video you want to run the inference on.
- run this file.

## Components used
- Fine-Tuned YOLOv11 weights
- Implemented Motion Tracking for post fall behavior
- CNN (Classifier) -> frame input

## Classifier Report
<img width="5000" alt="image" src="https://github.com/user-attachments/assets/475f9201-66cf-4a4e-ba6c-b4a1b0c99b80">

## Fine-tuned yolov11 model
<img width="5000" alt="image" src="https://github.com/user-attachments/assets/27f85c1b-a367-4c7d-a98a-08f695d7341a">


## Algorithm Evaluation
<img width="726" alt="Screenshot 2024-11-24 at 22 21 27" src="https://github.com/user-attachments/assets/14dd31e3-5dc0-4cc7-8338-87b2ee9ca28c">




  
