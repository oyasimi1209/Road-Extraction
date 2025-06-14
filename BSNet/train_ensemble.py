from datagathering.adaboost_dataset import generate_boosting_data
from train_tiny import train_operation
import os
import time
import sys
import subprocess  # 使用subprocess替代cmd


total_classifiers = 5
image_list_dir = os.path.join(os.path.dirname(__file__), "boost_train/txt_files/massa/")
logfile_dir = os.path.join(os.path.dirname(__file__), "boost_train/logs/")
model_dir = os.path.join(os.path.dirname(__file__), 'boost_train/weights/')
image_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/Shaoxing/test_satellite/')
gt_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/Shaoxing/test_label/')
image_prob_file_save_dir = image_list_dir
image_name_file_save_dir = image_list_dir
if os.path.isdir(image_list_dir):
    pass
else:
    os.mkdir(image_list_dir)

# 添加默认的训练ID
train_id = sys.argv[1] if len(sys.argv) > 1 else "1"
# train_id=1
if os.path.exists(image_list_dir+"train_image_file_1.txt"):
    pass
else:
    print("generate image list txt")
    with open(image_list_dir+"train_image_file_1.txt", "w") as f:
        for name in os.listdir(image_dir):
            f.write(name+"\n")

train_paras = {"learning_rate": 0.005,
               "total_epoch": 200,
               "train_id": 0,
               "image_dir": image_dir,
               "gt_dir": gt_dir,
               "image_list_dir": image_list_dir,
               "logfile_dir": logfile_dir,
               "model_dir": model_dir,
               "model_name": "no name now",
               "image_prob_file_save_dir":image_prob_file_save_dir,
               "image_name_file_save_dir":image_name_file_save_dir}

timestamp = "{}{}{}".format(time.localtime().tm_hour, time.localtime().tm_min, time.localtime().tm_sec)
model_name = timestamp+'massa_roadnet_' + str(train_id)
train_paras["model_name"] = model_name
train_paras["train_id"] = int(train_id)
print(model_name)
train_operation(train_paras)
generate_boosting_data(train_paras)
