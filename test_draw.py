import numpy as np
import sys
import os
import cv2
import caffe
import glob
import time
from tqdm import tqdm


class SSDDetect(object):
    def __init__(self, net_file='example/MobileNetSSD_deploy.prototxt', weights='snapshot/mobilenet_v2_300x300_iter_10000.caffemodel', input_size=(300, 300), scale=0.007843, gpu=0):
        caffe.set_mode_gpu()
        caffe.set_device(gpu)
        self.net_file = net_file
        self.weights = weights
        self.net = caffe.Net(self.net_file, self.weights, caffe.TEST)
        self.input_size = input_size
        self.scale = scale

    def preprocess(self, src):
        img = cv2.resize(src, (self.input_size))
        img = img - 127.5
        img = img * self.scale
        return img

    def postprocess(self, img, out):
        h = img.shape[0]
        w = img.shape[1]
        box = out['detection_out'][0,0,:,3:7] * np.array([w, h, w, h])
        cls = out['detection_out'][0,0,:,1]
        conf = out['detection_out'][0,0,:,2]
        return (box.astype(np.int32), conf, cls)

    def detect(self, img_path):
        origimg = cv2.imread(img_path)
        img = self.preprocess(origimg)

        img = img.astype(np.float32)
        img = img.transpose((2, 0, 1))
        self.net.blobs['data'].data[...] = img
        start = time.time()
        out = self.net.forward()
        end = time.time()
        print(end-start)
        box, conf, cls = self.postprocess(origimg, out)
        return box, conf, cls


def draw(img_path, box, conf, cls, save_dir):
    class_list = ["background", "car"]
    img_name = os.path.basename(img_path)
    img = cv2.imread(img_path)
    save_path = os.path.join(save_dir, img_name)
    if len(box) > 0:
        for i, j, k in zip(box, conf, cls):
            p1 = (i[0], i[1])
            p2 = (i[2], i[3])
            width = p2[0] - p1[0]
            height = p2[1] - p1[1]
            p3 = (max(p1[0], 15), max(p1[1], 15))
            cv2.rectangle(img, p1, p2, (0, 0, 255), 2)
            if j > 0.7:
                title = "%s:%.2f" % (class_list[int(k)], j)
                cv2.putText(img, title, p3, cv2.FONT_ITALIC, 1, (0, 255, 0), 2)
        cv2.imwrite(save_path, img)
    else:
        cv2.imwrite(save_path, img)


if __name__ == "__main__":
    detector = SSDDetect()
    img_list = glob.glob("imgs_my/*.jpg")
    for img_path in tqdm(img_list):
        box, conf, cls = detector.detect(img_path)
        draw(img_path, box, conf, cls, "imgs_res")


