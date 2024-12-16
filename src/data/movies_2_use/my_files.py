v1   = {'nickname':'v1' ,'ground_truth':[]    ,'person going down': 0 ,'owner_group_name':'lector','sub_dir':'negative','ext':'MTS','file':"person_walking_back_01"                                 ,'comment':'self_recorded,high quality'}
v2   = {'nickname':'v2' ,'ground_truth':[]    ,'person going down': 0 ,'owner_group_name':'lector','sub_dir':'negative','ext':'MTS','file':"person_walking_front_02"                                ,'comment':'self_recorded,high quality'}
v3   = {'nickname':'v3' ,'ground_truth':[]    ,'person going down': 0 ,'owner_group_name':'lector','sub_dir':'negative','ext':'MTS','file':"person_tying_laces_front"                               ,'comment':'self_recorded,high quality'}
v4   = {'nickname':'v4' ,'ground_truth':[]    ,'person going down': 0 ,'owner_group_name':'lector','sub_dir':'negative','ext':'MTS','file':"person_tying_laces_back"                                ,'comment':'self_recorded,high quality'}
v5   = {'nickname':'v5' ,'ground_truth':[]    ,'person going down': 0 ,'owner_group_name':'lector','sub_dir':'negative','ext':'mp4','file':"crowd_mon_cam_oh_straat (5)_fooling_algo_spreading_arms",'comment':''} 
v6   = {'nickname':'v6' ,'ground_truth':[]    ,'person going down': 0 ,'owner_group_name':'lector','sub_dir':'negative','ext':'mp4','file':"crowd_mon_cam_oh_straat (4)_tripping"                   ,'comment':'lost detection of person that fell!'} 
v7   = {'nickname':'v7' ,'ground_truth':[]    ,'person going down': 0 ,'owner_group_name':'lector','sub_dir':'negative','ext':'mp4','file':"crowd_mon(7)_fooling_algo_spreading_arms"               ,'comment':''} 
v8   = {'nickname':'v8' ,'ground_truth':[]    ,'person going down': 0 ,'owner_group_name':'lector','sub_dir':'negative','ext':'mp4','file':"dawn_fooling_spreading arms_vertical_shot"              ,'comment':''} 
v9   = {'nickname':'v9' ,'ground_truth':[]    ,'person going down': 0 ,'owner_group_name':'lector','sub_dir':'negative','ext':'mp4','file':"dawn_person_tying_laces"                                ,'comment':''} 
v10  = {'nickname':'v10','ground_truth':[]    ,'person going down': 0 ,'owner_group_name':'lector','sub_dir':'negative','ext':'MTS','file':"2_persons_crossing"                                     ,'comment':'self_recorded,high quality'}

v1000  = {'nickname':'v1000','ground_truth':[] ,'person going down': 1 ,'owner_group_name':'lector','sub_dir':'negative','ext':'mp4','file':"person_laying_but_ok_1"          ,'comment':'self_recorded, smartphone, jerre in garden on chair'}
v1001  = {'nickname':'v1001','ground_truth':[] ,'person going down': 1 ,'owner_group_name':'lector','sub_dir':'negative','ext':'mp4','file':"person_laying_but_ok_2"          ,'comment':'self_recorded, smartphone, jerre in garden on chair'}
v1002  = {'nickname':'v1002','ground_truth':[] ,'person going down': 1 ,'owner_group_name':'lector','sub_dir':'negative','ext':'mp4','file':"person_laying_but_ok_3"          ,'comment':'self_recorded, smartphone, jerre in garden on chair'}
v1003  = {'nickname':'v1003','ground_truth':[] ,'person going down': 1 ,'owner_group_name':'lector','sub_dir':'negative','ext':'mp4','file':"person_laying_but_ok_4"          ,'comment':'self_recorded, smartphone, jerre in garden on chair'}
v1004  = {'nickname':'v1004','ground_truth':[] ,'person going down': 1 ,'owner_group_name':'lector','sub_dir':'negative','ext':'mp4','file':"young_man_watching_tv"           ,'comment':'self_recorded, smartphone, jerre in living..zapping '}

# only detected well yolo nano fine tuned v2
v100   = {'nickname':'v100','ground_truth':[135] ,'person going down': 1 ,'owner_group_name':'lector','sub_dir':'positive','ext':'MTS','file':"simulation_proximus_2_fall_in_line_with_cam_view"       ,'comment':'no fall detected ..because gradual shrinking of aspecty ratio (fall in line with camera view), there is a sudden decrease in area...but the aspect ration is decreasing aswell...'} 
v101   = {'nickname':'v101','ground_truth':[90]  ,'person going down': 1 ,'owner_group_name':'lector','sub_dir':'positive','ext':'MTS','file':"proximus (4)"                                           ,'comment':'no fall detected ..because detection lost..and tracker ar increases smoothly.......dannys box also triggered a fall sometimes...but the sudden decrease in area is in sync with an decrease of the aspect ratio...luckely enough' }
v102   = {'nickname':'v102','ground_truth':[500] ,'person going down': 1 ,'owner_group_name':'lector','sub_dir':'positive','ext':'mp4','file':"simulation_chantier"                                    ,'comment':'no fall detected ..because detection lost..and tracker ar increases smoothly.......dannys box also triggered a fall sometimes...but the sudden decrease in area is in sync with an decrease of the aspect ratio...luckely enough' }
v103   = {'nickname':'v103','ground_truth':[700] ,'person going down': 1 ,'owner_group_name':'lector','sub_dir':'positive','ext':'mp4','file':"simulation_chantier_2"                                  ,'comment':'no fall detected ..because detection lost..and tracker ar increases smoothly.......dannys box also triggered a fall sometimes...but the sudden decrease in area is in sync with an decrease of the aspect ratio...luckely enough' }

# only detected well with yolo-x
v110   = {'nickname':'v110','ground_truth':[260] ,'person going down': 1 ,'owner_group_name':'lector','sub_dir':'positive','ext':'MTS','file':"simulation_real_accident_001"                           ,'comment':'self_recorded'}
v111   = {'nickname':'v111','ground_truth':[170] ,'person going down': 1 ,'owner_group_name':'lector','sub_dir':'positive','ext':'webm','file':"sudden cardiac arrest tatami"                          ,'comment':'from web'}
v112   = {'nickname':'v112','ground_truth':[850] ,'person going down': 1 ,'owner_group_name':'lector','sub_dir':'positive','ext':'mp4','file':"young_man_living_1"                                     ,'comment':'self_recorded'}
v113   = {'nickname':'v113','ground_truth':[340] ,'person going down': 1 ,'owner_group_name':'lector','sub_dir':'positive','ext':'mp4','file':"young_man_living_2"                                    ,'comment':'self_recorded'}


# not supported
v120   = {'nickname':'v120','ground_truth':[180] ,'person going down': 1 ,'owner_group_name':'lector','sub_dir':'positive','ext':'mp4','file':"crowd_mon_cam_oh_straat (7)_simulating_fall"            ,'comment':'self_recorded, crowd mon cam, person in diagonal of detection box..so fail to detect fall'}
v121   = {'nickname':'v121','ground_truth':[140] ,'person going down': 1 ,'owner_group_name':'lector','sub_dir':'positive','ext':'mp4','file':"sudden cardiac arrest parking"                          ,'comment':'no person detected by yolo after fall, video quality not good enough'}
v122   = {'nickname':'v122','ground_truth':[260] ,'person going down': 1 ,'owner_group_name':'lector','sub_dir':'positive','ext':'webm','file':"sudden cardiac arrest chantier"                        ,'comment':'also relative bad quality video...so the small boxes tend to jump...and gost boxes tend to appear/dissapear'}

v17    = {'nickname':'v17','ground_truth':[]    ,'person going down': 0 ,'owner_group_name':'lector','sub_dir':'negative','ext':'mp4','file':"crowd_mon_cam_oh_straat (6)_fooling_algo_spreading_arms",'comment':'no fall detected (correctly) because angle of person in box is in favor'} 
v18    = {'nickname':'v18','ground_truth':[]    ,'person going down': 0 ,'owner_group_name':'lector','sub_dir':'negative','ext':'mp4','file':"crowd_mon_cam_oh_straat (1)_multi_people_walking"       ,'comment':'no fall detected (correctly) because angle of person in box is in favor'} 
v19    = {'nickname':'v19','ground_truth':[]    ,'person going down': 0 ,'owner_group_name':'lector','sub_dir':'negative','ext':'webm','file':"man_sunbathing_beach"                                  ,'comment':'persons are small, relative low quality'} #often false detections(dustbins), false fall detections...several times...so false alarms


# these are the latest greatest set of videos to be tested
my_videos_2b_tested =[v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v1000,v1001,v1002,v1003,v1004,  v100,v101,v102,v103,v110,v111,v112,v113]
my_videos_negative_2b_tested = [v101,v102,v1000,v100,v1001,v103,v113,v1002,v1003,v1004]
my_videos_positive_2b_tested = [v110,v103,v111,v112,v113]
my_videos_2b_tested_false_alert = [v103,v113, v1001, v1004]
laying_but_okay = [v1002,v110,v103,v1003]


# these are the videos on which my latest algo fails
#my_videos_2b_tested =[v1004,v112,v102,v110]




# the more challenging ones
# my_videos_2b_tested =[v1000,v1001,v1002,v1003,v1004, v100,v101,v102,v103,v110,v111,v112,v113]

# the ones that go wrong
#my_videos_2b_tested =[tbd]

#my_videos_2b_tested=[v102,v103,v110,v111]
#my_videos_2b_tested=[v112,v113,v1004]
#videos_2b_tested=[v1000,v1001,v1002,v1003]

#videos_2b_tested=[v10,v1,v2,v3,v4,v5,v105,v10,v107,v108,v110,v111,v6]
#videos_2b_tested=[v10,v1,v2,v3,v4,v6,v7,v8,v9,v10,v11,v12]
#videos_2b_tested=[v10,v1,v2,v3,v4,v7,v8,v9,v110,v111] # x-model

# final tests on versison 1 
#videos_2b_tested=[v10,v1,v2,v3,v4,v6,v7,v8,v9,v110,v111] # x-model
#videos_2b_tested=[v100,v101] # nano-fine-tuned


