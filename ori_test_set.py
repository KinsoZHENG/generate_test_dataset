import os
import shutil
import time

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

path = '/home/kinsozheng/Desktop/EAD2020_dataType_framesOnly/EAD2020_dataType_framesOnly/frames/'
test_path = '/home/kinsozheng/Desktop/generate_test_dataset/ori_test_set/'

txt_path = "/home/kinsozheng/Desktop/EAD2020_dataType_framesOnly/EAD2020_dataType_framesOnly/gt_bbox/"
test_txt_path = "/home/kinsozheng/Desktop/generate_test_dataset/ori_test_txt/"

# files = os.listdir(path)
# print(files.__len__())
# ori_test_txt = open('ori_test_txt.txt', 'w')
#
# video_files = [name for name in files]
# trains, tests = train_test_split(video_files, test_size=0.025, random_state=0)
#
# for test in tqdm(tests):
#     ori_test_txt.write(test + '\n')
#
# ori_test_txt.close()
# time.sleep(10)
test_name_file = "/home/kinsozheng/Desktop/generate_test_dataset/ori_test_txt.txt"

# Copy img
# with open(test_name_file) as f:
#     count = 0
#     # print(f.readnames())
#     name_list = f.readlines()
#     for name in name_list:
#         source = path + name.strip('\n')
#         dst = test_path + name.strip('\n')
#         print(source, dst)
#         shutil.copyfile(source, dst)
#         count += 1
#     print(count)

# Copy txt
with open(test_name_file) as f:
    count = 0
    name_list = f.readlines()
    for name in name_list:
        name = os.path.splitext(name)[0]
        source = txt_path + name + ".txt"
        dst = test_txt_path + name + ".txt"
        print(source, dst)
        shutil.copyfile(source, dst)
        count += 1
    print(count)
