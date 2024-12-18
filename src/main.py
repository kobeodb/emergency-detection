"""
project: EyesForResque
Author: BotBrigade
"""

import cv2
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
import os
import numpy as np

from ultralytics import YOLO

from data.movies_2_use.my_files import *
from data.movies_2_use.student_files import *

videos_2b_tested = my_videos_2b_tested+student_videos_2b_tested
# videos_2b_tested = student_positive_2b_tested
# videos_2b_tested = my_videos_negative_2b_tested
# videos_2b_tested = my_videos_positive_2b_tested
# videos_2b_tested = my_videos_2b_tested_false_alert + student_videos_2b_tested_false_alert
# videos_2b_tested = student_videos_2b_tested_false_alert
# videos_2b_tested = my_videos_2b_tested_false_alert
# videos_2b_tested = laying_but_okay

device = 'mps'
# device = 'cuda'
# device = 'cpu'

visualize_bbox = False
make_eval_table = True

############################################################

##read tracked detection from file
# store_tracked_detections_in_file=False
# read_detections_from_file=True
# detect_alerts=True

##Store tracked detection in file
# store_tracked_detections_in_file=True
# read_detections_from_file=False
# detect_alerts=False

##Do everything without files
store_tracked_detections_in_file=False
read_detections_from_file=False
detect_alerts=True

#############################################################
## load videos configuration ##
#############################################################

use_minio = True
use_local = False

download_from_minio_and_store_in_file = False

assert not (use_minio and use_local and download_from_minio_and_store_in_file), "Only one variable can be True"

##############################################################
## algorithm parameter configuration ##
##############################################################

conf_thres_detection              = 0.4     #minimum confidence for yolo detection.
secs_fall_motion_tracking         = 0       #maximum seconds between when a fall is detected and when motion tracking starts.
secs_motion_tracking_double_check = 3       #seconds between the start of motion tracking and the double check with the classifier

prob_thres_img_classifier         = 0.5     #if prob < threshold -> emergency
                                            #if prob > threshold -> fine
prob_thres_pose_classifier        = 0.7     # =

acc_in_sec_of_alert               = 6       #amount of seconds in where a frame alert is considered correct.

motion_sensitivity                = 30      #if you increase this number more motion is needed to detect motion


#######################################################
## fall detection configuration ##
#######################################################

use_custom_fall_detection   = False
use_ft_yolo                 = True
assert not (use_custom_fall_detection and use_ft_yolo),"only 1 variable can be true at the same time"


#######################################################
## Classifier configuration ##
#######################################################

double_check_through_img_classifier     = False
double_check_through_pose_classifier    = True
assert not (double_check_through_img_classifier and double_check_through_pose_classifier), "both variables cannot be True at the same time"


########################################################
## pose estimator configuration ##
########################################################

if double_check_through_pose_classifier:
    use_mediapipe_pose = False
    use_yolo_pose      = True
    assert not (use_yolo_pose and use_mediapipe_pose), "both variable cannot be true at the same time"

    if use_yolo_pose:
        use_yolo_rf = True
        use_yolo_nn = False


#######################################################
## Motion tracking configuration ##
#######################################################

use_static_back_motion  = False
use_distance_motion     = True
use_std_dev_motion      = False
assert not (use_static_back_motion and use_distance_motion and use_std_dev_motion), "Only 1 variable can be True"

#########################################################
#########################################################


if use_minio or download_from_minio_and_store_in_file:
    from minio import Minio
    from minio.error import S3Error
    import tempfile

    load_dotenv()

    minio_url = os.getenv('MINIO_ENDPOINT')
    minio_access_key = os.getenv('MINIO_ACCESS_KEY')
    minio_secret_key = os.getenv('MINIO_SECRET_KEY')
    secure=False

    minio_bucket_name = os.getenv('MINIO_BUCKET_NAME')


current_dir = Path.cwd()

detections_dir = Path(current_dir / "data" / "detections")
video_file_dir = Path(current_dir / "data" / "movies_2_use")
model_weight_dir = Path(current_dir / "data" / "weights")
classifier_dir = Path(current_dir / "models" / "classifiers")

if use_custom_fall_detection:
    from utils.all_yolo_classes import yolo_class_dict

    yolo_detect_model = "yolo11n.pt"
    yolo_detect_model_path = Path(model_weight_dir / "yolo_detection" / yolo_detect_model)

    yolo_model = YOLO(yolo_detect_model_path)

if use_ft_yolo:
    from utils.all_yolo_ft_classes import yolo_class_dict

    yolo_detect_model="best.pt" #if this is selected -> switch the classes in /utils/all_yolo_ft.py
    # yolo_detect_model="best_2.pt"
    # yolo_detect_model="best_3.pt"
    # yolo_detect_model="best_4.pt"
    yolo_detect_model_path = Path(model_weight_dir / "yolo_detection" / yolo_detect_model)

    yolo_model = YOLO(yolo_detect_model_path)


if double_check_through_img_classifier:
    import torch
    import yaml
    from src.models.classifiers.img.cnn.classifier import CNN

    config_path = '../config/config.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)

    img_classifier_path = Path(classifier_dir / "img" / "cnn")

    img_classifier  = CNN(config).to(device)
    checkpoint = torch.load(Path(img_classifier_path / 'best_model.pth'), map_location=device)
    img_classifier.load_state_dict(checkpoint['classifier_state_dict'])
    img_classifier.eval()

    classifier_input_size = 128

if double_check_through_pose_classifier and use_mediapipe_pose:
    import mediapipe as mp
    import joblib

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)

    rf_model_name = 'best_rf_model_M.pkl'
    rf_model_path = Path(current_dir / 'data' / 'weights' / 'media_pipe_rfclassifier' / rf_model_name)

    rf_model = joblib.load(rf_model_path)

if double_check_through_pose_classifier and use_yolo_pose:
    from models.classifiers.pose.detection_keypoints_yolo import DetectKeypoint

    if use_yolo_nn:
        import torch
        from models.classifiers.pose.nn.yolo_pose.classification_keypoint import KeypointClassification


        # yolo_pose_nn_name = 'pose_classification.pt'
        yolo_pose_nn_name = 'pose_classification_v2.pt'

        yolo_pose_nn_path = Path(model_weight_dir / "yolo_pose_nnClassifier" / yolo_pose_nn_name)

        yolo_pose_keypoint_detection = DetectKeypoint()
        yolo_pose_nn_classification = KeypointClassification(yolo_pose_nn_path)

    if use_yolo_rf:
        import joblib

        yolo_pose_keypoint_detection = DetectKeypoint()

        yolo_rf_model_name = 'pose_classification_v1.pkl'

        yolo_rf_model_path = Path(current_dir / 'data' / 'weights' / 'yolo_pose_rfclassifier' / yolo_rf_model_name)

        yolo_rf_model = joblib.load(yolo_rf_model_path)







#colors
GREEN = (0, 255, 0)
BLUE  = (255, 0, 0)
RED   = (0, 0, 255)
ORANGE= (0, 165, 255)
WHITE = (255, 255, 255)
YELLOW= (0, 255, 255)
CYAN  = (255, 255, 0)


detection_color=GREEN
fall_detected_color=ORANGE
motion_tracking_color=YELLOW
emergency_detected=RED

##########################################################
## helper functions ##
##########################################################

if use_static_back_motion:
    columns = ['frame_nb', 'xmin_bbox', 'ymin_bbox', 'w_bbox', 'h_bbox', 'area_bbox', 'motion', 'on_the_ground', 'trigger_classifier', 'alert', 'static_back', 'on_ground_count' ,'no_motion_on_ground_count']
if use_distance_motion:
    columns = ['frame_nb', 'xmin_bbox', 'ymin_bbox', 'w_bbox', 'h_bbox', 'area_bbox', 'motion', 'on_the_ground', 'trigger_classifier', 'alert', 'on_ground_count' ,'no_motion_on_ground_count']
if use_std_dev_motion:
    columns = ['frame_nb', 'xmin_bbox', 'ymin_bbox', 'w_bbox', 'h_bbox', 'area_bbox', 'aspr_bbox' ,'motion', 'on_the_ground', 'trigger_classifier', 'alert', 'on_ground_count' ,'no_motion_on_ground_count']


def check_for_motion(frame, xmin, ymin, h, w, track_history, track_id, factor):
    if use_static_back_motion:
        motion_threshold = 1000

        xmax = xmin + w
        ymax = ymin + h

        roi = frame[ymin:ymax, xmin:xmax]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if track_id not in track_history or track_history[track_id].empty:
            print(f"[DEBUG] No track history found for track_id: {track_id}. Initializing...")
            track_history[track_id] = pd.DataFrame(columns=["static_back"])
            last_idx = 0
            static_back = None
        else:
            last_idx = track_history[track_id].index[-1]
            static_back = track_history[track_id].loc[last_idx, 'static_back']

        if static_back is None or static_back.shape != gray.shape:
            print(f"[DEBUG] Initializing static_back for track_id: {track_id}")
            static_back = gray
            track_history[track_id].at[last_idx, 'static_back'] = static_back
            return False

        if gray.shape != static_back.shape:
            print(f"[DEBUG] Resizing gray to match static_back shape for track_id: {track_id}")
            gray = cv2.resize(gray, (static_back.shape[1], static_back.shape[0]))

        diff_frame = cv2.absdiff(static_back, gray)

        thresh_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)[1]
        thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

        contours, _ = cv2.findContours(thresh_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            contour_area = cv2.contourArea(contour)
            print(f"[DEBUG] Detected contour with area: {contour_area}")
            if contour_area >= motion_threshold:
                print(f"[DEBUG] Motion detected for track_id: {track_id}")
                return True  # Motion detected

        # If no motion detected, update the static background
        print(f"[DEBUG] No motion detected. Updating static_back for track_id: {track_id}")
        track_history[track_id].at[last_idx, 'static_back'] = gray
        return False

    if use_distance_motion:
        motion = False

        P = min(10, len(track_history))
        track_history = track_history[track_id]
        history_df_last_P = track_history[-P:]

        col_x = 'xmin_bbox'
        col_y = 'ymin_bbox'
        col_w = 'w_bbox'
        col_h = 'h_bbox'

        for i, row in history_df_last_P.iterrows():
            x_prev = row[col_x]
            y_prev = row[col_y]
            w_prev = row[col_w]
            h_prev = row[col_h]

            if (abs(x_prev - xmin) > w_prev / factor or abs(y_prev - ymin) > h_prev / factor or abs(
                    w_prev - w) > w_prev / factor or abs(h_prev - h) > h_prev / factor):
                motion = True
                break

        return motion

    # if use_std_dev_motion:
        # nb_of_frames = min(10, len(track_history))
        # track_history = track_history[track_id]
        # track_history_last_nb_frames = track_history[-nb_of_frames:]
        #
        # col_area = "area_bbox"
        # col_aspect_ratio = "aspr_bbox"

        # if len(track_history) < 2:
        #     return True

        # length = len(track_history)
        # print(f"length of history: {length}")

        # if col_area in track_history_last_nb_frames.columns:
            # area_change_series = track_history_last_nb_frames[col_area].dropna().diff().abs().dropna()
            # print(f"area_change_series: {area_change_series}")

            # area_change_avg = area_change_series.mean()
            # area_change_std = area_change_series.std()
            # area_prev = track_history_last_nb_frames[col_area].iloc[-1]
            # low_motion_area_threshold = min(area_change_avg + motion_std_multiplier_area * area_change_std, motion_sensitivity)

        # if col_aspect_ratio in track_history_last_nb_frames.columns:
        #     aspr_change_series = track_history_last_nb_frames[col_aspect_ratio].dropna().diff().abs().dropna()
        #     aspr_change_avg = aspr_change_series.mean()
        #     aspr_change_std = aspr_change_series.std()

        # area_low_motion_thres = area_change_avg + motion_std_multiplier_area * area_change_std
        # aspr_low_motion_thres = aspr_change_avg + motion_std_multiplier_aspr * aspr_change_std
        # print(f"Area Change Avg: {area_change_avg}, Area Change Std: {area_change_std}")
        # print(f"Area Low Motion Threshold: {area_low_motion_thres}")
        # motion = area_low_motion_thres > area_motion_sensitivity

        # return motion


def check_if_person_is_on_the_ground(cls_name) -> bool:
    if use_custom_fall_detection:
        #todo -> this shit should probably not be in the same function but who cares
        return False

    if use_ft_yolo:
        if cls_name == 'not_on_ground':
            return False
        elif cls_name == 'on_the_ground':
            return True

def check_for_alert_in_history(track_history, number_max_frames) -> bool:
    number_frames = min(number_max_frames, len(track_history))
    last_frames = track_history[-number_frames:]
    alerts = last_frames[last_frames['alert']]

    if not alerts.empty:
        return True
    else:
        return False


def check_if_last_frame_was_alert(track_history) -> bool:
    last_alert = track_history.iloc[-1]
    is_alert = last_alert['alert']

    return bool(is_alert)

# def get_last_on_the_ground_frame_in_history(track_history):
#     frames_on_the_ground = track_history[track_history['on_the_ground']]
#
#     if frames_on_the_ground.empty:
#         return None
#
#     return frames_on_the_ground.iloc[-1]['frame_nb']



def crop_bbox(xmin, ymin, w, h, frame):
    xmax = xmin + w
    ymax = ymin + h

    cropped_frame = frame[ymin:ymax, xmin:xmax]

    return cropped_frame


def update_track_history(track_history, track_id, frame, frame_nb, xmin_bbox, ymin_bbox, w_bbox, h_bbox, area_bbox, aspr_bbox ,cls_name,max_frames_fall_motion_tracking, max_frames_motion_tracking_double_check, process_frames_reduction_factor):
    """
    :param track_history:  track history containing tracking history of al all tracked objects
                           Each key is unique using track_id, and the value is a dataframe containing
                           historical tracking information\
    :param track_id:       Unique identifier assigned by yolo track
    :param frame:          Current frame being processed
    :param frame_nb:       index of the frame
    :param xmin_bbox:      top left coordinate of the tracked bbox
    :param ymin_bbox:      top left coordinate of the tracked bbox
    :param w_bbox:         width of the tracked bbox
    :param h_bbox:         height of the tracked bbox
    :param area_bbox:      calculated area of the tracked bbox
    :param cls_name:       name of the detected class given by yolo

    """

    alert = False
    static_back = None
    generate_alert = False
    motion = False
    on_the_ground = False
    no_motion_on_ground_count = 0

    if track_id not in track_history:

        if use_static_back_motion:
            new_record = pd.DataFrame([[frame_nb, xmin_bbox, ymin_bbox, w_bbox, h_bbox, area_bbox, motion, on_the_ground, False, False, None, 0,0]], columns=columns)

        if use_distance_motion:
            new_record = pd.DataFrame([[frame_nb, xmin_bbox, ymin_bbox, w_bbox, h_bbox, area_bbox, motion,on_the_ground, False, False, 0, 0]], columns=columns)

        if use_std_dev_motion:
                    new_record = pd.DataFrame([[frame_nb, xmin_bbox, ymin_bbox, w_bbox, h_bbox, area_bbox, aspr_bbox, motion,on_the_ground, False, False, 0 ,0]], columns=columns)

        track_history[track_id] = new_record

    else:
        last_record = track_history[track_id].iloc[-1]
        prev_on_ground_count = last_record.get('on_ground_count')
        prev_no_motion_count = last_record.get('no_motion_on_ground_count', 0)

        on_the_ground = check_if_person_is_on_the_ground(cls_name)

        if on_the_ground:
            on_ground_count = prev_on_ground_count + 1
            # print(f"on_ground_count: {on_ground_count}")
        else:
            on_ground_count = 0

        if on_ground_count >= max_frames_fall_motion_tracking:
            motion = check_for_motion(frame, xmin_bbox, ymin_bbox, h_bbox, w_bbox, track_history, track_id, motion_sensitivity/process_frames_reduction_factor)

            if on_the_ground and not motion:
                no_motion_on_ground_count = prev_no_motion_count + 1
                # print(f"no_motion_on_ground_count: {no_motion_on_ground_count}")
            else:
                no_motion_on_ground_count = 0
                motion = False

        generate_alert = no_motion_on_ground_count >= max_frames_motion_tracking_double_check

        if use_static_back_motion:
            static_back = track_history[track_id].iloc[-1]['static_back']


        alr_alerted = check_for_alert_in_history(track_history[track_id], number_max_frames=len(track_history[track_id]))

        cropped_frame = crop_bbox(xmin, ymin, w, h, frame)

        if double_check_through_img_classifier and not alr_alerted:
            if generate_alert:
                with torch.no_grad():

                    crop_tensor = cv2.resize(cropped_frame, (128, 128))
                    crop_tensor = torch.tensor(crop_tensor, dtype=torch.float32).permute(2, 0, 1)
                    crop_tensor = crop_tensor.unsqueeze(0).to(device)
                    prediction = img_classifier(crop_tensor)
                    score = prediction.item()

                    if score > prob_thres_img_classifier:
                        pred_label_index = 1
                    if score < prob_thres_img_classifier:
                        pred_label_index = 0

                    class_names = ("emergency", "no emergency")
                    predicted_label = class_names[pred_label_index]

                    if predicted_label == 'no emergency':
                        generate_alert = False
                    print(f'Classifier predicted: {predicted_label} with probability: {score} on frame {frame_nb}')


        if double_check_through_pose_classifier and use_mediapipe_pose and not alr_alerted:
            if generate_alert:
                cropped_frame_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
                cropped_frame_rgb = cv2.resize(cropped_frame_rgb, (256, 256))

                landmarks = [0, 11, 12, 23, 24, 25, 26]

                result = pose.process(cropped_frame_rgb)

                kps = np.zeros(len(landmarks) * 3)

                if result.pose_landmarks:
                    for i, landmark in enumerate(landmarks):
                        landmark = result.pose_landmarks.landmark[landmark]
                        kps[i * 3:i * 3 + 3] = [landmark.x, landmark.y, landmark.z]

                pose_keypoints = kps.reshape(1, -1)

                prediction = rf_model.predict_proba(pose_keypoints)[0, 1]


                if prediction > prob_thres_pose_classifier:
                    pred_label_index = 1
                if prediction < prob_thres_pose_classifier:
                    pred_label_index = 0

                class_names = ("emergency", "no emergency")
                predicted_label = class_names[pred_label_index]

                if predicted_label == 'no emergency':
                    generate_alert = False
                print(f'Classifier predicted: {predicted_label} with probability: {prediction} on frame {frame_nb}')

        if double_check_through_pose_classifier and use_yolo_pose and not alr_alerted:
            if generate_alert:

                results = yolo_pose_keypoint_detection(cropped_frame)

                results_keypoint = yolo_pose_keypoint_detection.get_xy_keypoint(results)

                input_classification = results_keypoint[10:]

                if use_yolo_nn:
                    prediction = yolo_pose_nn_classification(input_classification)
                    print(f"prediction: {prediction}")

                    emergency_proba = prediction[0]
                    no_emergency_proba = prediction[1]

                    if emergency_proba >= prob_thres_pose_classifier:
                        prediction = 0 #fine
                    elif emergency_proba <= prob_thres_pose_classifier:
                        prediction = 1 #need help

                    class_names = ("no emergency", "emergency")
                    predicted_label = class_names[int(prediction)]

                    if predicted_label == 'no emergency':
                        generate_alert = False

                    proba = emergency_proba
                    if prediction == 1:
                        proba = no_emergency_proba

                    print(f'Classifier predicted: {predicted_label}, with probability: {proba} on frame {frame_nb}')

                if use_yolo_rf:

                    pose_keypoints = np.array(input_classification).reshape(1, -1)

                    prediction = yolo_rf_model.predict_proba(pose_keypoints)[0, 1]  # probability of class 1 (need help)


                    if prediction > prob_thres_pose_classifier: # prediction of class 1 (need help) is larger than the threshold
                        pred_label_index = 0
                    if prediction < prob_thres_pose_classifier:
                        pred_label_index = 1

                    class_names = ("emergency", "no emergency")
                    predicted_label = class_names[pred_label_index]

                    if predicted_label == 'no emergency':
                        generate_alert = False
                    print(f'Classifier predicted: {predicted_label} with probability: {prediction} on frame {frame_nb}')


        if generate_alert:
            alert = True

        new_record = None
        if use_static_back_motion:
            new_record = pd.DataFrame([[frame_nb, xmin_bbox, ymin_bbox, w_bbox, h_bbox, area_bbox, motion, on_the_ground, generate_alert, alert, static_back, on_ground_count ,no_motion_on_ground_count]], columns=columns)

        elif use_distance_motion:
            new_record = pd.DataFrame([[frame_nb, xmin_bbox, ymin_bbox, w_bbox, h_bbox, area_bbox, motion, on_the_ground, generate_alert, alert, on_ground_count ,no_motion_on_ground_count]], columns=columns)

        elif use_std_dev_motion:
                    new_record = pd.DataFrame([[frame_nb, xmin_bbox, ymin_bbox, w_bbox, h_bbox, area_bbox, aspr_bbox, motion, on_the_ground, generate_alert, on_ground_count ,alert, no_motion_on_ground_count]], columns=columns)

        track_history[track_id] = pd.concat([track_history[track_id], new_record], ignore_index=True)

        if len(track_history[track_id]) > max_history_len:
            track_history[track_id] = track_history[track_id].drop(track_history[track_id]['frame_nb'].idxmin())
    return track_history, motion, on_the_ground, alert




############################################################################
## Run the Algorithm over all te benchmark videos
## Videos are being retrieved from minio and stored in a temporary file
## Algorithm steps:
##       - Detection
##       - Alert


results_df = pd.DataFrame(columns=['video','truth','found','correct','false','missed'])

tot_vids=0
for vid in videos_2b_tested:
    tot_vids += 1
    emergency = False

    ground_truth = vid['ground_truth']

    if use_minio or download_from_minio_and_store_in_file:
        minioClient = Minio(
            minio_url,
            minio_access_key,
            minio_secret_key,
            secure=False
        )

        object_name = "movies/"+vid['sub_dir']+"/"+vid['file']+"."+vid['ext']

        if download_from_minio_and_store_in_file:
            local_file_path = Path("local_movies") / vid['sub_dir'] / f"{vid['file']}.{vid['ext']}"
            local_file_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                data = minioClient.get_object(minio_bucket_name + '-' + vid['owner_group_name'], object_name)
                with local_file_path.open('wb') as file:
                    file.write(data.read())
                print(f"File downloaded successfully to {local_file_path}")

                video_cap = cv2.VideoCapture(str(local_file_path))

            except S3Error as exc:
                print("Error occurred:", exc)

        if use_minio:
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            try:
                data = minioClient.get_object(minio_bucket_name+'-'+vid['owner_group_name'], object_name)
                temp_file.write(data.read())
                temp_file.close()
                video_cap = cv2.VideoCapture(temp_file.name)
            except S3Error as exc:
                print("Error occured:", exc)

    elif use_local:
        local_file_path = Path("local_movies") / vid['sub_dir'] / f"{vid['file']}.{vid['ext']}"
        video_cap = cv2.VideoCapture(str(local_file_path))


    tot_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("processing:"+vid['file'])
    print("total frames:", tot_frames)
    print("total videos:",tot_vids)

    h_res = video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    v_res = video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps_orig = video_cap.get(cv2.CAP_PROP_FPS)
    print('resolution:' + str(h_res) + 'x' + str(v_res) + ' fps:' + str(fps_orig))

    process_frames_reduction_factor = 3 #this will perform frameskipping, for example
                                        # -> if the fps of the video is 30 fps, this will now be reduced to 10fps
    fps_reduced=round(fps_orig/process_frames_reduction_factor)

    max_frames_fall_motion_tracking = round(fps_reduced*secs_fall_motion_tracking)
    max_frames_motion_tracking_double_check = round(fps_reduced*secs_motion_tracking_double_check)

    detection_times = []
    frame_count = 0
    alerts_generated = []
    track_history = {}
    max_history_len=max_frames_fall_motion_tracking+max_frames_motion_tracking_double_check+10

    detections_file = (
        Path(current_dir)
        / detections_dir
        / f"{yolo_detect_model}_{conf_thres_detection}"
        / vid['owner_group_name']
        / vid['sub_dir']
        / f"tracked_detections_{vid['nickname']}_{vid['file']}.pkl"
    )
    if store_tracked_detections_in_file:
        detections_df = pd.DataFrame(columns=['frame', 'detections'])
    if read_detections_from_file:
        detections_df = pd.read_pickle(detections_file)

    standard_resolution = (1920, 1080)

    while True:

        ret, frame = video_cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, standard_resolution, interpolation=cv2.INTER_LINEAR)

        clean_frame = frame.copy()

        frame_count=frame_count+1
        if (frame_count % process_frames_reduction_factor) == 0:

            #########################
            ## DETECTION ##
            #########################

            tracked_results = yolo_model.track(frame, verbose=False, persist=True)[0]

            bboxs_and_conf_clss = []

            xyxys = tracked_results.boxes.xyxy.int().tolist()           #min_x, min_y -> upper left corner of bbox
                                                                        #max_x, max_y -> bottom right corner of bbox
            xywhs = tracked_results.boxes.xywh.int().tolist()           #average x, y -> middle x/y value
                                                                        #w, h -> width, height of bbox
            if tracked_results.boxes.id is not None:
                track_ids = tracked_results.boxes.id.int().tolist()     #Unique track id for each bbox
            else:
                track_ids = []
            cls_ids = tracked_results.boxes.cls.int().tolist()          #class idea fir each bbox
            confs = tracked_results.boxes.conf.tolist()                 #confidence scores

            for xyxy, xywh, track_id, cls_id, conf in zip(xyxys, xywhs, track_ids, cls_ids, confs):

                xmin, ymin, xmax, ymax = xyxy
                x_av, y_av, bbox_w, bbox_h = xywh

                if float(conf) < conf_thres_detection:
                    continue

                bboxs_and_conf_clss.append([[xmin, ymin, bbox_w, bbox_h], conf, cls_id, track_id])

                if store_tracked_detections_in_file:
                    new_row = {'frame': frame_count, 'detections': bboxs_and_conf_clss}
                    new_row_df = pd.DataFrame.from_records([new_row])
                    detections_df = pd.concat([detections_df, new_row_df], ignore_index=True, axis=0)


            if detect_alerts:
                #######################################
                ## ALERT ##
                #######################################
                for bbox_and_conf_cls in bboxs_and_conf_clss:
                    xywh = bbox_and_conf_cls[0]
                    confidence = bbox_and_conf_cls[1]
                    cls_id = bbox_and_conf_cls[2]
                    cls_name = yolo_class_dict[cls_id]
                    track_id = bbox_and_conf_cls[3]

                    xmin, ymin, w, h = xywh
                    xmax = xmin + w
                    ymax = ymin + h

                    area = abs(w * h)
                    aspr = abs(w / h)

                    cropped_frame = frame[ymin:ymax, xmin:xmax]

                    new_track_history, no_motion, on_the_ground, alert = update_track_history(track_history, track_id, clean_frame, frame_count, xmin, ymin, w, h, area, aspr,cls_name, max_frames_fall_motion_tracking, max_frames_motion_tracking_double_check, process_frames_reduction_factor)

                    if check_if_last_frame_was_alert(new_track_history[track_id]):
                        bbox_color = RED
                        if not emergency:
                            emergency = True
                            alerts_generated.append(frame_count)
                            print("frame count of emergency: ",frame_count)

                            if make_eval_table:
                                break
                    elif on_the_ground and no_motion:
                        bbox_color = YELLOW
                    elif on_the_ground:
                        bbox_color = ORANGE
                    else:
                        bbox_color = GREEN

                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), bbox_color, thickness=2)
                    # cv2.putText(f"{frame}, id: {track_id}, {(xmin + 5, ymin - 8)}, {cv2.FONT_HERSHEY_SIMPLEX}, {1}, ")

                if visualize_bbox:
                    standard_width, standard_height = 640, 480
                    frame = cv2.resize(frame, (standard_width, standard_height))

                    cv2.imshow("frame", frame)

                    if cv2.waitKey(1) == ord("q"):
                        break

    video_cap.release()
    cv2.destroyAllWindows()

    if make_eval_table:

        alerts_correct = []
        alerts_false = []
        alerts_missed = []

        for ground_truth_frame in ground_truth:
            found = False
            for alerted_frame in alerts_generated:
                print(f"alerted frames: {alerted_frame}")
                if (ground_truth_frame+round(secs_motion_tracking_double_check * fps_orig) - alerted_frame) <= fps_orig * acc_in_sec_of_alert:
                    found = True

            if found:
                alerts_correct.append(ground_truth_frame)

            else:
                 alerts_missed.append(ground_truth_frame)

        for alerted_frame in alerts_generated:
            was_correct = False
            for ground_truth_frame in ground_truth:
                if (ground_truth_frame+round(secs_motion_tracking_double_check * fps_orig) - alerted_frame) <= fps_orig * acc_in_sec_of_alert:
                    was_correct = True

            if not was_correct:
                alerts_false.append(alerted_frame)

        new_eval_row = {'video': vid['file'],'nicknamae': vid['nickname'] ,'truth': vid['ground_truth'], 'found':alerts_generated, 'correct': alerts_correct, 'false':alerts_false, 'missed': alerts_missed}
        new_eval_row_df = pd.DataFrame.from_records([new_eval_row])
        results_df = pd.concat([results_df, new_eval_row_df], ignore_index=True)



####################################################################
## Calculate precision, recall and the false alert rate.
####################################################################

if make_eval_table:

    tot_truth = sum(len(row['truth']) for _, row in results_df.iterrows())
    tot_found = sum(len(row['found']) for _, row in results_df.iterrows())
    tot_correct = sum(len(row['correct']) for _, row in results_df.iterrows())
    tot_false = sum(len(row['false']) for _, row in results_df.iterrows())
    tot_missed = sum(len(row['missed']) for _, row in results_df.iterrows())

    metrics = {
        'precision': round((tot_correct / (tot_correct + tot_false)) * 100, 2) if tot_found > 0 else 0,
        'recall': round((tot_correct / (tot_correct + tot_missed)) * 100, 2) if tot_found > 0 else 0,
        'false_alert_rate': round((tot_false / tot_found) * 100, 2) if tot_found > 0 else 0
    }

    print(metrics)




    results_df.to_excel('results.xlsx', index=False)