# /usr/bin/python
# -*- encoding:utf-8 -*-
# 人脸检测

import dlib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2


def rect_to_bb(rect): # 获得人脸矩形的坐标信息
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)


def resize(image, width=1200):  # 将待检测的image进行resize
    r = width * 1.0 / image.shape[1]
    dim = (width, int(image.shape[0] * r))
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized


def detect():
    path = './image/'
    image_file = path + '4.jpg'
    detector = dlib.get_frontal_face_detector()
    image = cv2.imread(image_file)
    image = resize(image, width=1200)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    for (i, rect) in enumerate(rects):
        (x, y, w, h) = rect_to_bb(rect)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image,
                    "Face： {}".format(i + 1),
                    (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Output", image)
    cv2.waitKey(0)


if __name__ == "__main__":
    detect()

