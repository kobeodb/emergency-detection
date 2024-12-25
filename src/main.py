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
videos_2b_tested = laying_but_okay
# videos_2b_tested = student_videos_2b_tested_missed_alert


device = 'mps'
# device = 'cuda'
# device = 'cpu'

############################################################

visualize_bbox = True
make_eval_table = False

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

use_minio = False
use_local = True

download_from_minio_and_store_in_file = False

assert not (use_minio and use_local and download_from_minio_and_store_in_file), "Only one variable can be True"

##############################################################
## algorithm parameter configuration ##
##############################################################

conf_thres_detection                = 0.4     #minimum confidence for yolo detection.

secs_fall_motion_tracking           = 0       #maximum seconds between when a fall is detected and when motion tracking starts.
secs_motion_tracking_double_check   = 3       #seconds between the start of motion tracking and the double check with the classifier

prob_thres_img_classifier           = 0.5     #if prob < threshold -> emergency
                                            #if prob > threshold -> fine
prob_thres_pose_classifier          = 0.7     # =

acc_in_sec_of_alert                 = 6       #amount of seconds in where a frame alert is considered correct.

motion_sensitivity                  = 30      #if you increase this number more motion is needed to detect motion

#######################################################
## fall detection configuration ##
#######################################################

use_custom_fall_detection   = True
use_ft_yolo                 = False
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

use_distance_motion     = True
use_std_dev_motion      = False
assert not (use_distance_motion and use_std_dev_motion), "Only 1 variable can be True"

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
    import joblib

    yolo_detect_model = "yolo11s.pt"
    yolo_detect_model_path = Path(model_weight_dir / "yolo_detection" / yolo_detect_model)
    yolo_model = YOLO(yolo_detect_model_path)

    adaboost_model_path = Path(model_weight_dir / 'adaboost_fall_detection' / 'adaboost_fall_detector.pkl')
    adaboost_model = joblib.load(adaboost_model_path)

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

if use_distance_motion:
    columns = ['frame_nb', 'xmin_bbox', 'ymin_bbox', 'w_bbox', 'h_bbox', 'area_bbox', 'motion', 'on_the_ground', 'trigger_classifier', 'alert', 'on_ground_count' ,'no_motion_on_ground_count']
if use_std_dev_motion:
    columns = ['frame_nb', 'xmin_bbox', 'ymin_bbox', 'w_bbox', 'h_bbox', 'area_bbox', 'aspr_bbox' ,'motion', 'on_the_ground', 'trigger_classifier', 'alert', 'on_ground_count' ,'no_motion_on_ground_count']


def check_for_motion(frame, xmin, ymin, h, w, track_history, track_id):
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

            factor = max(2, abs((w_prev + h_prev) / motion_sensitivity))

            if (abs(x_prev - xmin) > w_prev / factor or abs(y_prev - ymin) > h_prev / factor or abs(
                    w_prev - w) > w_prev / factor or abs(h_prev - h) > h_prev / factor):
                motion = True
                break

        return motion


def check_if_person_is_on_the_ground(track_data, bbox_w, bbox_h, cls_name) -> bool:
    if use_custom_fall_detection:
        area = bbox_w * bbox_h
        aspect_ratio = bbox_w / bbox_h

        area_change_avg = 0
        area_change_std = 0
        aspect_ratio_change_avg = 0
        aspect_ratio_change_std = 0
        aspect_ratio_avg = aspect_ratio
        aspect_ratio_std = 0

        P = min(10, len(track_data))

        if P > 5:
            track_data_df = pd.DataFrame(track_data)

            track_data_last_p = track_data_df.iloc[-P:]

            col_area = 'area'
            col_ar = 'aspect_ratio'

            area_change_avg = track_data_last_p[col_area].dropna().diff().dropna().mean()
            area_change_std = track_data_last_p[col_area].dropna().diff().dropna().std()

            aspect_ratio_change_avg = track_data_last_p[col_ar].dropna().diff().dropna().mean()
            aspect_ratio_change_std = track_data_last_p[col_ar].dropna().diff().dropna().mean()

            aspect_ratio_avg = track_data_last_p[col_ar].mean()
            aspect_ratio_std = track_data_last_p[col_ar].std()

        input_features = [[area, aspect_ratio, area_change_avg, area_change_std, aspect_ratio_change_avg, aspect_ratio_change_std, aspect_ratio_avg, aspect_ratio_std]]

        predictions = adaboost_model.predict(input_features)[0]

        return True if predictions == 0 else False

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
    generate_alert = False
    motion = True
    on_the_ground = False
    no_motion_on_ground_count = 0

    if track_id not in track_history:

        if use_distance_motion:
            new_record = pd.DataFrame([[frame_nb, xmin_bbox, ymin_bbox, w_bbox, h_bbox, area_bbox, motion, on_the_ground, False, False, 0, 0]], columns=columns)

        if use_std_dev_motion:
                    new_record = pd.DataFrame([[frame_nb, xmin_bbox, ymin_bbox, w_bbox, h_bbox, area_bbox, aspr_bbox, motion,on_the_ground, False, False, 0 ,0]], columns=columns)

        track_history[track_id] = new_record

    else:
        last_record = track_history[track_id].iloc[-1]
        prev_on_ground_count = last_record.get('on_ground_count')
        prev_no_motion_count = last_record.get('no_motion_on_ground_count', 0)

        on_the_ground = check_if_person_is_on_the_ground(track_history, w_bbox, h_bbox, cls_name)

        if on_the_ground:
            on_ground_count = prev_on_ground_count + 1
            # print(f"on_ground_count: {on_ground_count}")
        else:
            on_ground_count = 0

        # print(f"[DEBUG] on the ground count: {on_ground_count}")
        if on_ground_count > max_frames_fall_motion_tracking:
            motion = check_for_motion(frame, xmin_bbox, ymin_bbox, h_bbox, w_bbox, track_history, track_id)
            # print(f"[DEBUG] motion: {motion}")

            if on_the_ground and not motion:
                no_motion_on_ground_count = prev_no_motion_count + 1
                # print(f"no_motion_on_ground_count: {no_motion_on_ground_count}")
            else:
                no_motion_on_ground_count = 0
                motion = False

        generate_alert = no_motion_on_ground_count >= max_frames_motion_tracking_double_check



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

                    emergency_proba = prediction[1] # [fine, needhelp]
                    no_emergency_proba = prediction[0]

                    if emergency_proba >= prob_thres_pose_classifier:
                        prediction_idx = 0
                    elif emergency_proba <= prob_thres_pose_classifier:
                        prediction_idx = 1

                    class_names = ("emergency", "no emergency")
                    predicted_label = class_names[int(prediction_idx)]

                    proba = emergency_proba

                    if predicted_label == 'no emergency':
                        generate_alert = False
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

        if use_distance_motion:
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

                    new_track_history, motion, on_the_ground, alert = update_track_history(track_history, track_id, clean_frame, frame_count, xmin, ymin, w, h, area, aspr,cls_name, max_frames_fall_motion_tracking, max_frames_motion_tracking_double_check, process_frames_reduction_factor)

                    # print(f"[DEBUG] motion: {motion}")

                    if check_if_last_frame_was_alert(new_track_history[track_id]):
                        bbox_color = RED
                        txt = 'EMERGENCY'
                        if not emergency:
                            emergency = True
                            alerts_generated.append(frame_count)
                            print("frame count of emergency: ",frame_count)

                            if make_eval_table:
                                break
                    elif on_the_ground and motion:
                        bbox_color = ORANGE
                        'on_the_ground'
                    elif on_the_ground and not motion:
                        bbox_color = YELLOW
                        txt = 'motion tracking'

                    else:
                        bbox_color = GREEN
                        txt = 'not_on_ground'

                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), bbox_color, thickness=2)
                    cv2.putText(frame,  txt, (xmin + 5, ymin - 8), cv2.FONT_HERSHEY_SIMPLEX, 1, bbox_color , 2)

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