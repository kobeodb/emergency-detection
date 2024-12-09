# Bot Brigade - Eyes4Resque


# Bot Brigade - Eyes4Rescue

## Introduction
Eyes4Rescue is an AI-powered project aimed at detecting and responding to emergency situations involving individuals who have fallen and may need urgent assistance. The primary goal of this project is to reduce the time it takes to interfere with a given case. We want to be able to determine, based on signals like motion, if a person is in need of help and thus notify the emergency services as quickly as possible. 
 
By leveraging computer vision techniques such as YOLOv11 for object detection, MediaPipe for pose estimation and custom built classifiers, the project aims to make a difference in the time it takes to interfere in emergencies. The system analyzes video feeds to detect falls, track motion, and assess whether a person is in need of help based on their behavior after the fall. This project is a step toward creating more efficient and reliable emergency response systems through real-time video analysis and AI.
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

After testing the 2d cnn classifier on a test set we had some great results:
<img width="946" alt="Screenshot_2024-12-07_at_21 52 08" src="https://github.com/user-attachments/assets/d23b6747-f97e-4e5b-84bb-eec250fefc01">

But as you can see in the evaluation below, we still want to improve upon these methods, that's why we are revisiting pose keypoint extraction. This time, the extracted keypoints will be used as input to a random forest classifier.

Our plan is to compare the performance of the 2D CNN and the pose-based random forest classifier. Based on the results, we will decide which approach to refine further.

## Results
For a set of 23 videos (10 negative/13 positive) the algorithm scored

- 71% precision
- 71% recall
- 29% false alert rate

<img width="667" alt="Screenshot 2024-12-08 at 22 10 10" src="https://github.com/user-attachments/assets/27fd0369-d4d4-449c-b266-242c51765a0e">

This marks a significant improvement compared to previous weeks. However, the evaluation highlights areas where our algorithm is still falling short. In certain scenarios, the classifier continues to make incorrect classifications (red), which is unacceptable in critical situations. Additionally, the fine-tuned YOLO model occasionally fails to perform accurately, resulting in missed alert detections (yellow).


## The sources
***

### alert detector: inference_single_person.py (in ./src/pipeline dir)

This Python script contains the final alert detection algorithm, containing all the logic for how emergencies are identified.


### classifier.py (in ./src/models/classifier dir)

This python script hold the architecture of the 2d cnn and the data loading.


### algorithm_eval_table.py (in ./src/models/metrics dir)

This Python script generates the evaluation table by utilizing the alert detection algorithm and iterating through specified videos.


### config.yaml (in ./config dir)

This yaml file hold all the adjustable parameters across all models (yolo and classifier) and all paths.

## How to run scripts and reproduce the results
***

### Setup your python environment

- python version: 3.11
- pip install -r requirements

### Setup all the right paths and device

- In the config.yaml file, update the paths to match your own directory structure. Additionally, if you want to train the YOLO model or the classifier, make sure to set the device field to the hardware you are using (e.g., cpu, cuda, etc.).

### training yolo model and classifier
- The repository does not include any data or model files, so you will need to provide your own datasets and pre-trained models to proceed.

## Run the alert detector script.
- If all goes well you should be able to run the alert detector script after you have specified the video you want to run the algorithm on.

# Open Issues
***

- Things to do are registered in [github tasks](https://github.com/mxttywxtty/bot-brigade/issues?q=is%3Aissue+is%3Aopen+label%3Atask)
- Bugs are registered in [known open bugs](https://github.com/mxttywxtty/bot-brigade/issues?q=is%3Aissue+is%3Aopen+label%3Abug+)



  
