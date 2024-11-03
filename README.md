# Bot Brigade

## Report 06/10/2024 - (intro)

### Goal
The goal of this project is to improve the time to react (TTR) so we can improve the overall time to intervention (TTI). As a result of improving the time to react (TTR) efficiency we can improve the time to intervention (TTI). We are searching for new innovative methods making use of AI to look for an improvement in the TTR. We have already thought of several methods on how we can achieve our goal. 

### Existing technology
Several technological methods already exist that are fairly accessible. One of the most common examples is a wearable health monitor that monitors several biometric attributes of the person wearing it. The data is used as a sign to see if the person wearing it is in good health. If there happens to be some suspicious patterns that the device picks up on within the data, then intervention is planned. This method significantly reduces the TTI since the TTR is lower compared to someone that isn't making use of a wearable health monitor. 

Another type of device that ties into the function of the wearable health monitor is a predictive analytics tool. These tools are used to make predictions about a person’s health. They share some similarity with the wearable health monitors because they both analyze biometric data of a person. The predictive analytics tool is more powerful in the sense that it has the capability to inform medical intervention teams. 

The tool we are most interested in is fine motion camera detection. These are cameras that make use of specialized software to detect patterns in human movement. These movements are analyzed by AI to detect health issues in patients, more specifically chronic diseases. 

### Our approach
Our prototype will consist of a combination of attributes of the three tools discussed above. Each tool will play a role in improving the TTI. The prototype will consist of a camera being able to detect a person’s movement. Live camera feed will be fed into the algorithm to simulate a realistic situation. When the camera detects a certain movement pattern it will send a message to the medical intervention team. The camera will be able to detect and classify the severity of the movement, this allows us to regulate the priority of several cases at once. (All the requirements are written out in a requirements txt file located on our GitHub )

## Testing our system
We will test our prototype by conducting controlled experiments where volunteers simulate various movement patterns combined with already existing video material that can be found online, including medical emergencies. The camera system will monitor these scenarios to evaluate its ability to detect and classify critical movements accurately. By analyzing the system's performance in these tests, we can refine our algorithms to improve detection accuracy and ensure the system effectively reduces the Time to React (TTR) and improves the overall Time to Intervention (TTI).

### Sources:
1. Article 1 (PMC8800083):
Chen, Y., Lin, H., Qiu, C., & Peng, Z. (2022). Wearable motion sensors for healthcare: On-body sensor applications and challenges. Frontiers in Sensors, 3, 1-10. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8800083/
2. Article 2 (PMC8741942):
Li, X., Ma, S., Liu, L., & Zhang, P. (2021). Deep learning approaches in human activity recognition using wearables: A review. Frontiers in Digital Health, 2, 100-115. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8741942/
3. Article 3 (Abto Software Blog):
Abto Software. (2023, August 20). How motion recognition technology is improving public health. Abto Software. https://www.abtosoftware.com/blog/motion-recognition-to-improve-public-health
4. Article 4 (ScienceDirect):
Wei, H., Zhang, S., & Liu, F. (2023). Development of deep learning models for wearable motion data analysis in healthcare. Wearable Technology, 12(4), 102-116. https://www.sciencedirect.com/science/article/pii/S0146280623003328

## Report 13/10/2024

### Starting the project

Add your weights files to the weights directory. These weights are needed to run the project.
```commandline
cd /src/data/weights 
tree /f
```

You can run the `bb` command by supplying the file you want to detect objects on. You can also supply the weights you put in your `weights` directory. When no weights file is supplied, then the weights file is downloaded from yolo and defaults to `yolo11.pt`.
```commandline
pip install -e .
bb video_1.avi -w yolo11n.pt 
```

The code will look for a filename in the minio bucket, download it locally, run detection on it and then clean it up by deleting it (locally). The temporary files are kept in `out/temp` directory.

### Results 

Since this project only focuses on human detection, I have narrowed the class detection list down to one to detect only people. When staying stationary, even the smallest YOLOv11 model seems to have good accuracy. When the person is moving however the accuracy drops. In some cases the detection stops working entirely for a few moments, but picks back up a while later.

## Report 20/10/2024

### Fine-tuning the model 

#### Identifying stationary, moving and falling people

##### Falling people
When detecting if an object (or in this case a person) has fallen, our first approach was looking at the correlation between the height and the width of the box surrounding the person. 
We first decided to check if the width was greater than the height to conclude if the person had fallen over. Sadly, this was not a good approach, because there were many instances where a person 
had not fallen over, but the width was still greater than the height (when a person first comes into frame for example). 
```python
"""
 Approach 1.
"""
height = 2
width = 1

if width > height:
    # is falling
    pass
else: 
    # not falling
    pass
```
\
For our next approach we took a look at the aspect ratio of the width and height rather than comparing them head on. For a smaller aspect ratio, we determined that the person had fallen, but for a higher aspect ratio we determined that the person was either in movement or stationary (not falling).
```python
"""
 Approach 2.
"""
height = 2
width = 1

if height / width < 1:
    # is falling
    pass
else: 
    # not falling
    pass
```

##### Stationary & moving people 
When looking at stationary people we decided to compare the distance of two centers of the boxes surrounding the person. We keep track of a single center point first for one frame. The next frame, we calculate the new center. We can compare these centers and check if the distance of these two boxes has shifted to some extent. 
If the difference between the two distances is greater than a certain value then we can conclude that the person is in movement. If the differences between the distances are small then we can conclude that the person can be somewhat stationary. 

##### Conclusion
By combining these properties we can make several distinctions of classifications about the movement state of people. A person can be stationary or in movement. When a person is in movement they can either be falling or not falling.  

##### Dataset
We found a dataset consisting of roughly 170 videos of people falling. The videos are recorded in different settings. Only the videos of one room which are in total 27 videos are on minio to start with because that's enough for now. Later we can add more videos if needed.
Reference of the dataset: 
```
I. Charfi, J. Mitéran, J. Dubois, M. Atri, R. Tourki, "Optimised spatio-temporal descriptors for real-time fall detection: comparison of SVM and Adaboost based classification”, Journal of Electronic Imaging (JEI), Vol.22. Issue.4, pp.17, October 2013.
```

### User interface
We have also started working on an user interface. This could help us for testing and it is a beginning of working towards an user friendly application too for our MVP. For the moment, it is only for inputting fields to our code but we are working towards an application that can showcase the tracking of people and info in realtime when choosing a video. The code is currently just in a branch: main/tkinter.


## Report 3/11/2024


### Overview

* Machine learning model for fall detection using pose keypoints data.
* Built with an XGBoost classifier and an advanced preprocessing pipeline.
* Uses MediaPipe Pose estimation for keypoint extraction.
* Binary classification: "Need help" vs. "No need for help".


### Data Processing

#### MediaPipe Pose for Keypoint Extraction

* 33 body landmarks extracted (x, y, z coordinates).
* Processes images from train/val directories.
* Automatic handling of missing/invalid poses.

#### Dataset Creation Pipeline

* Converts YOLO format labels to binary classes.
* Extracts pose keypoints from images.
* Saves processed data to CSV format.


### Feature Engineering

#### Advanced Techniques on Keypoints

* Torso-normalized keypoint coordinates for better invariance.
* Angle calculations for shoulders and elbows.
* Distance-based features:
    * Shoulder and hip widths.
    * Ratios like shoulder-to-hip width.

#### Preprocessing Pipeline

* Missing value imputation using median strategy.
* Feature scaling with standardization.
* Label encoding for target variable.

#### Model Architecture

* XGBoost classifier with advanced configurations.
* Hyperparameter optimization via RandomizedSearchCV.
* Early stopping to prevent overfitting.
* Multi-class classification support for extensibility.

#### Performance

* Validation accuracy metrics provided.
* Classification report for detailed performance analysis.
* Model evaluation on a separate validation dataset.


### Fall Detection Application

#### Overview

* Developed a GUI application using PyQt5 for real-time fall detection.
* Allows users to:
    * Load a video file or use the laptop's camera for live detection.
    * Start and stop detection with ease.
    * View the video with predictions overlaid on each frame.

#### Key Features

* Utilizes the trained XGBoost model for classification.
* Integrates MediaPipe Pose for real-time keypoint extraction from video frames.
* Applies the same feature engineering pipeline used during training for consistency.
* Displays predictions ("Need help" or "No need for help") on each video frame.
* Includes a progress bar and status updates within the GUI.
* Handles missing or invalid pose data gracefully.

#### Technical Details

* GUI developed using PyQt5 for cross-platform compatibility.
* Real-time video capture and processing handled via OpenCV.
* MediaPipe Pose used for efficient and accurate pose estimation.
* Feature engineering functions mirror those used during training to ensure consistent data processing.
* Prediction results are overlaid on the video frames using OpenCV functions.
* Suppresses specific warnings for a cleaner user experience.

#### Usage

* Run the application script fall_detection_app.py to launch the GUI.
* Use "Load Video" to select a video file, or "Use Camera" to use the laptop's camera.
* Click "Start Detection" to begin processing.
* The application displays the video with prediction labels and updates the progress bar accordingly.
* Upon completion, the status updates, and the user is notified.

#### Technical Stack

* GUI: PyQt5.
* Computer Vision: OpenCV, MediaPipe.
* ML/Data Processing: pandas, numpy, xgboost, scikit-learn.
* Additional Tools: SMOTE for class balancing.
* Model Persistence: joblib.

### Model and Resources

* Trained model saved as improved_fall_detection_model_xgb.pkl.
* Label encoder saved as improved_label_encoder.pkl.
* Located in ../src/models/ directory.
* The application script is fall_detection_app.py located in ../src/app/ directory.

