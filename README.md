# Bot Brigade - Eyes4Resque

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

To address this, we maybe want to explore alternative approaches for fall detection in the future. Our current ideas include:
- Adaboost: Creating a few weak classifiers (stomps) for example: bbox area over time, aspect ratio, ... and feeding these classifiers to an adaboost algorithm to create 1 good lightweight classifier for detecting a fall alongside yolo for the person detection.


### Motion detection of people that fell on the ground (or lay down).
Detecting individuals falling to the ground is not sufficient to trigger a 112 alert, as the person may simply stand up and walk away afterward. In contrast, individuals who have suffered a cardiac arrest or stroke typically remain motionless after falling, requiring immediate assistance.

To check for motion after a person is detected as being on the floor, we implemented a simple distance based motion tracking that will look at the distance between the bbox coordinates in the current frame and in the previous frame. This distance is then campared to a factor calculated by a dynamic number (area of bbox) devided by a hard coded motion threshold.

This simple yet effective method allows real-time detection and logging of motion events, making it ideal for surveillance applications.

### Detection of motionless people that fell on the floor and effectively need help.
Detecting whether an individual is motionless on the ground is not sufficient to trigger a 112 alert, as it’s possible for someone to be on the ground (in the grass taking a nap) without motion without.
Our goal is to differentiate between situations where motionless individuals are fine and those where they are experiencing an emergency.

Our current classifiers are all pose based and consist of:
- mediapipe w/ random forest classifier
- yolo-pose w/ random forest classifier
- yolo-pose w/ neural network

As of right now our yolo-pose w/ neural network is the best performing classifier we have:

<img width="385" alt="Screenshot 2024-12-18 at 03 22 50" src="https://github.com/user-attachments/assets/adf39d9c-f0d6-491a-8168-5d1bdd5034e0" />

Collecting more data 

## Results
For a set of 38 videos (... negative/... positive) the algorithm scored

- ...% precision
- ...% recall
- ...% false alert rate

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



  
