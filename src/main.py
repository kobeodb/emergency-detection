"""
project: EyesForResque
Author: BotBrigade
"""

from datetime import datetime

import cv2
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
import os

from torch.backends.mkl import verbose

from data.movies_2_use.student_files import *
from data.movies_2_use.my_files import *

videos_2b_tested = my_videos_2b_tested+student_videos_2b_tested

visualize_bbox = True

############################################################

#read tracked detection from file
# store_tracked_detections_in_file=False
# read_detections_from_file=True
# detect_alerts=True

#Store tracked detection in file
store_tracked_detections_in_file=True
read_detections_from_file=False
detect_alerts=False

#Do everything without files
# store_tracked_detections_in_file=False
# read_detections_from_file=False
# detect_alerts=True

use_minio = True

##############################################################

conf_thres_detection = 0.3 #minimum confidence for yolo detection.
secs_fall_motion_tracking = 2 #maximum seconds between when a fall is detected and when motion tracking starts.
secs_motion_tracking_double_check = 4 #seconds between the start of motion tracking and the double check with the classifier

prob_thres_classifier = 0.5 #if prob < threshold
                            # -> emergency, if prob < threshold -> fine.

acc_in_sec_of_alert = 5 #amount of seconds in where a frame alert is considered correct.


#######################################################

double_check_through_img_classifier = True
double_check_through_pose_classifier = False
assert not (double_check_through_img_classifier and double_check_through_pose_classifier), "both variables cannot be True at the same time"

#######################################################

use_custom_fall_detection = True
use_ft_yolo = False
assert not (use_custom_fall_detection and use_ft_yolo),"both variables cannot be True at the same time"

#######################################################

if use_minio:
    from minio import Minio
    from minio.error import S3Error
    from utils.minio_utils import MinioClient
    import tempfile

    load_dotenv()

    minio_url = os.getenv('MINIO_URL')
    minio_access_key = os.getenv('MINIO_USER')
    minio_secret_key = os.getenv('MINIO_PASSWORD')
    secure=False

    minio_bucket_name = os.getenv('MINIO_BUCKET_NAME')


current_dir = Path.cwd()

detections_dir = Path(current_dir / "data" / "detections")
video_file_dir = Path(current_dir / "data" / "movies_2_use")
yolo_model_dir = Path(current_dir / "data" / "weights")
classifier_dir = Path(current_dir / "models" / "classifiers")

if use_custom_fall_detection:
    from utils.all_yolo_classes import yolo_class_dict

    yolo_detect_model = "yolo11n.pt"
    yolo_detect_model_path = Path(yolo_model_dir / yolo_detect_model)

if use_ft_yolo:
    from utils.all_yolo_ft_classes import yolo_class_dict

    yolo_detect_model="best.pt"
    # yolo_detect_model="best_2.pt"
    # yolo_detect_model="best_3.pt"
    # yolo_detect_model="best_4.pt"
    yolo_detect_model_path = Path(yolo_model_dir / yolo_detect_model)



if double_check_through_img_classifier:
    import torch
    import yaml

    #import the trained classifier weights


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

def check_if_person_has_fallen():
    #custom logic for fall detection
    return

def update_track_history(track_history, track_id, frame, frame_nb, xmin_bbox, ymin_bbox, w_bbox, h_bbox, area_bbox, aspect_ratio_bbox):
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
    :param aspect_ratio:   aspect ration of the tracked bbox

    """

    alert = False
    fall_detected_in_last_x_frames = False
    latest_fall_frame = None
    nb_frames_no_motion = None



##########################################################

tot_vids=0
for vid in videos_2b_tested:
    tot_vids += 1
    emergency_detected = False

    ground_truth = vid['ground_truth']

    if use_minio:
        minioClient = Minio(
            minio_url,
            minio_access_key,
            minio_secret_key,
            secure=False
        )

        object_name = "movies/"+vid['sub_dir']+"/"+vid['file']+"."+vid['ext']

        temp_file = tempfile.NamedTemporaryFile(delete=False)
        try:
            data = minioClient.get_object(minio_bucket_name+'-'+vid['owner_group_name'], object_name)
            temp_file.write(data.read())
            temp_file.close()
            video_cap = cv2.VideoCapture(temp_file.name)
        except S3Error as exc:
            print("Error occured:", exc)


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
    tracks_history = {}
    max_history_len=max_frames_fall_motion_tracking+max_frames_fall_motion_tracking+10

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

    while True:

        ret, frame = video_cap.read()
        if not ret:
            break
        clean_frame = frame.copy()

        frame_count=frame_count+1
        if (frame_count % process_frames_reduction_factor) == 0:

            start = datetime.now()

            #########################
            # Detection
            #########################


            tracked_results = yolo_detect_model_path.track(frame, verbose=False, persist=True)[0]

            bboxs_and_conf_clss = []

            xyxys = tracked_results.boxes.xyxy.int().tolist()       #min_x, min_y -> upper left corner of bbox
                                                                    #max_x, max_y -> bottom right corner of bbox
            xywhs = tracked_results.boxes.xywh.int().toList()       #average x, y -> middle x/y value
                                                                    #w, h -> width, height of bbox
            track_ids = tracked_results.boxes.id.int().tolist()     #Unique track id for each bbox
            class_ids = tracked_results.boxes.cls.int().tolist()    #class idea fir each bbox
            confs = tracked_results.boxes.conf().tolist()           #confidence scores

            for xyxy, xywh, track_id, class_id, conf in zip(xyxys, xywhs, track_ids, class_ids, confs):

                xmin, ymin, xmax, ymax = xyxy
                x_av, y_av, bbox_w, bbox_h = xywh

                cls_name = yolo_class_dict[class_id]

                if float(conf) < conf_thres_detection:
                    continue

                bboxs_and_conf_clss.append([[xmin, ymin, bbox_w, bbox_h], conf, class_id, track_id])

                if store_tracked_detections_in_file:
                    new_row = {'frame': frame_count, 'detections': bboxs_and_conf_clss}
                    new_row_df = pd.DataFrame.from_records([new_row])
                    detections_df = pd.concat([detections_df, new_row_df], ignore_index=True, axis=0)




            if detect_alerts:
                #######################################
                # Alert
                #######################################
                if double_check_through_img_classifier:



































