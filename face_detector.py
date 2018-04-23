#-*- coding:utf-8 -*-

import cv2


def detect_faces(image_pass):
    MODEL_PASS = './model/FaceDetector/haarcascade_frontalface_default.xml'

    image_BGR = cv2.imread(image_pass)
    model = cv2.CascadeClassifier(MODEL_PASS)

    image_gray = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2GRAY)

    facerect = model.detectMultiScale(image_gray, scaleFactor=1.11, minNeighbors=4, minSize=(100,100))

    if len(facerect) > 0:
        face_list = [image_BGR[y:y+h, x:x+w] for (x, y, w, h) in facerect]
        return face_list

    else:
        return []
