#coding:utf-8
import cv2
import os
import numpy as np
import random


def get_image_list(image_dir, suffix=['jpg']):
    '''get all image path ends with suffix'''
    if not os.path.exists(image_dir):
        print("PATH:%s not exists" % image_dir)
        return []
    imglist = []
    for root, sdirs, files in os.walk(image_dir):
        if not files:
            continue
        for filename in files:
            filepath = os.path.join(root, filename)
            if filename.split('.')[-1] in suffix:
                imglist.append(filepath)
    return imglist


def get_list(img_list):
    split_point = int(len(img_list)/100.0)
    train_list = img_list[:-split_point]
    test_list = img_list[-split_point:]

    train_lines = []
    for img_path_train in train_list:
        xml_path_train = img_path_train.replace("JPEGImages", "Annotations").replace(".jpg", ".xml")
        line_train = "%s %s\n" % (img_path_train, xml_path_train)
        train_lines.append(line_train)
    with open("trainval.txt", "w") as f:
        f.writelines(train_lines)

    test_lines, size_lines = [], []
    for img_path_test in test_list:
        xml_path_test = img_path_test.replace("JPEGImages", "Annotations").replace(".jpg", ".xml")
        line_test = "%s %s\n" % (img_path_test, xml_path_test)
        test_lines.append(line_test)
        img = cv2.imread(img_path_test)
        height, width, _ = img.shape
        img_name = os.path.basename(img_path_test)[:-4]
        size_line = "%s %s %s\n" % (img_name, height, width)
        size_lines.append(size_line)
    with open("test.txt", "w") as f:
        f.writelines(test_lines)
    with open("test_name_size.txt", "w") as f:
        f.writelines(size_lines)
        

if __name__ == "__main__":
    img_dir = "MyDataSet"
    img_list = get_image_list(img_dir)
    random.shuffle(img_list)
    get_list(img_list)
