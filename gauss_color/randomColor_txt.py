import os
import shutil

from natsort import natsort

img_path = "/home/kinsozheng/Desktop/generate_test_dataset/ori_test_set"
txt_path = "/home/kinsozheng/Desktop/generate_test_dataset/ori_test_txt/"
txt_save_path = "/home/kinsozheng/Desktop/generate_test_dataset/gauss_color/randomColor_txt/"
files = natsort.natsorted(os.listdir(img_path))
names = []

for file in files:
    name = os.path.splitext(file)[0]
    names.append(name)

for name in names:
    source = txt_path + name + ".txt"
    dst = txt_save_path + "randomColor0" + name + ".txt"
    shutil.copyfile(source, dst)

for name in names:
    source = txt_path + name + ".txt"
    dst = txt_save_path + "randomColor1" + name + ".txt"
    shutil.copyfile(source, dst)
