import xml.etree.ElementTree as ET
from multiprocessing import Pool

import cv2
import os

from natsort import natsort
from tqdm import tqdm

name_idx = {"specularity": 0,
            "saturation": 1,
            "artifact": 2,
            "blur": 3,
            "contrast": 4,
            "bubbles": 5,
            "instrument": 6,
            "blood": 7}


def xml_to_txt(name):
    xml_path = "/home/kinsozheng/Desktop/generate_test_dataset/rotated_set/rotated_xml/" + name + ".xml"
    img_path = "/home/kinsozheng/Desktop/generate_test_dataset/rotated_set/rotated_img/" + name + ".jpg"
    im_read = cv2.imread(img_path, -1)
    shape = im_read.shape
    img_h = shape[0]
    img_w = shape[1]

    tree = ET.parse(xml_path)
    root = tree.getroot()

    txt_path = "/home/kinsozheng/Desktop/generate_test_dataset/rotated_set/rotated_txt/" + name + ".txt"
    result_txt = open(txt_path, 'w')
    for infos in root.iter('object'):
        name = infos.find('name').text
        for info in infos.iter('bndbox'):
            x1 = float(info.find('xmin').text)
            y1 = float(info.find('ymin').text)
            x2 = float(info.find('xmax').text)
            y2 = float(info.find('ymax').text)

            idx_name = name_idx[name]
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h
            x = (x1 / img_w) + (w / 2.)
            y = (y1 / img_h) + (h / 2.)
            txt_info = ("%g " * 5 + "\n") % (idx_name, x, y, w, h)
            result_txt.write(txt_info)


if __name__ == '__main__':
    rotated_img_path = "/home/kinsozheng/Desktop/generate_test_dataset/rotated_set/rotated_img"
    files = natsort.natsorted(os.listdir(rotated_img_path))
    names = []

    for file in files:
        name = os.path.splitext(file)[0]
        names.append(name)

    pool = Pool()

    pool.map(xml_to_txt, tqdm(names))

    pool.close()
    pool.join()
