#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 13:24:24 2018

@author: ead2019
"""
import os
import seaborn as sns

import cv2
from tqdm import tqdm

class_dic = {0: 'specularity',
             1: 'saturation',
             2: 'artifact',
             3: 'blur',
             4: 'contrast',
             5: 'bubbles',
             6: 'instrument',
             7: 'blood'
             }
color_dic = {0: (0, 0, 0),
             1: (128, 0, 0),
             2: (0, 128, 0),
             3: (128, 128, 0),
             4: (0, 0, 128),
             5: (128, 0, 128),
             6: (0, 128, 128),
             7: (128, 128, 128), }


def read_img(imfile):
    import cv2

    return cv2.imread(imfile)[:, :, ::-1]


def read_boxes(txtfile):
    import numpy as np
    lines = []

    with open(txtfile, "r") as f:
        for line in f:
            line = line.strip()
            box = np.hstack(line.split()).astype(np.float)
            box[0] = int(box[0])
            lines.append(box)

    return np.array(lines)


def yolo2voc(boxes, imshape):
    import numpy as np
    m, n = imshape[:2]

    box_list = []
    for b in boxes:
        cls, x, y, w, h = b

        x1 = (x - w / 2.)
        x2 = x1 + w
        y1 = (y - h / 2.)
        y2 = y1 + h

        # absolute:
        x1 = x1 * n;
        x2 = x2 * n
        y1 = y1 * m;
        y2 = y2 * m

        box_list.append([cls, x1, y1, x2, y2])

    if len(box_list) > 0:
        box_list = np.vstack(box_list)

    return box_list


def plot_boxes(ax, boxes, labels):
    color_pal = sns.color_palette('hls', n_colors=len(labels))

    for b in boxes:
        cls, x1, y1, x2, y2 = b
        ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], lw=2, color=color_pal[int(cls)])

    return []


def read_obj_names(textfile):
    import numpy as np
    classnames = []

    with open(textfile) as f:
        for line in f:
            line = line.strip('\n')
            if len(line) > 0:
                classnames.append(line)

    return np.hstack(classnames)


if __name__ == "__main__":
    """
    Example script to read and plot bounding box annotations (which are provided in <x,y,w,h> format)

    (x,y) - box centroid
    (w,h) - width and height of box in normalised. 
    """
    import pylab as plt

    # imgfile = '/home/kinsozheng/Desktop/data_pre/generate_gt/annotationImages_and_labels/00003.jpg'
    # bboxfile = '/home/kinsozheng/Desktop/data_pre/generate_gt/annotationImages_and_labels/00003.txt'
    img_path = '/home/kinsozheng/Desktop/generate_test_dataset/generate_ground_truth/all_img/'
    txt_path = '/home/kinsozheng/Desktop/generate_test_dataset/generate_ground_truth/all_txt/'
    gt_txt_path = '/home/kinsozheng/Desktop/generate_test_dataset/generate_ground_truth/all_gt_txt/'
    # img_path = '/home/kinsozheng/Desktop/data_pre/trainingData_detection/'
    # txt_path = '/home/kinsozheng/Desktop/data_pre/txt/'
    # gt_txt_path = '/home/kinsozheng/Desktop/data_pre/all_gt_txt/'
    classfile = '/home/kinsozheng/Desktop/data_pre/generate_gt/class_list.txt'
    classes = read_obj_names(classfile)
    img_files = sorted(os.listdir(img_path))
    txt_files = sorted(os.listdir(txt_path))

    for img_file in tqdm(img_files):
        name = os.path.splitext(img_file)[0]
        img_file = img_path + img_file
        bboxfile = txt_path + name + '.txt'

        img = read_img(img_file)
        boxes = read_boxes(bboxfile)
        # print(boxes)
        # convert boxes from (x,y,w,h) to (x1,y1,x2,y2) format for plotting
        boxes_abs = yolo2voc(boxes, img.shape)
        # print(boxes_abs[0][0])
        im_read = cv2.imread(img_file)
        gt_txt_file = gt_txt_path + name + '.txt'
        gt_pic_file = "/home/kinsozheng/Desktop/generate_test_dataset/generate_ground_truth/bbox_img/" + name + ".png"
        txt_write = open(gt_txt_file, 'w')
        for k in range(boxes_abs.__len__()):
            index = int(boxes_abs[k][0])
            name = class_dic[index]
            txt_write.write(('%s\t' + '%g\t' * 4 + '\n') % (
                name, boxes_abs[k][1], boxes_abs[k][2], boxes_abs[k][3], boxes_abs[k][4]))
            x1, y1 = int(boxes_abs[k][1]), int(boxes_abs[k][2])
            x2, y2 = int(boxes_abs[k][3]), int(boxes_abs[k][4])
            # color_pal = sns.color_palette('hls', n_colors=len(classes))
            cv2.rectangle(im_read, (x1, y1), (x2, y2), color_dic[index], 2)
            cv2.putText(im_read, name, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_dic[index], 1)

        cv2.imwrite(gt_pic_file, im_read)
