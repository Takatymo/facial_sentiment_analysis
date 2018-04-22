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


def cut_and_save_face_imgs():
    import os

    ORIGINAL_DATA_DIR_PATH = './data/original_images/face_dataset/'
    FACE_DATA_DIR_PATH = './data/face_images/'

    for dir_name in os.listdir(FACE_DATA_DIR_PATH):
        if dir_name not in ['angry', 'fear', 'happy', 'normal', 'sad']:
            continue

        for file_name in os.listdir(os.path.join(FACE_DATA_DIR_PATH, dir_name)):
            if os.path.splitext(file_name)[1] not in ['.tiff', '.jpg']:
                continue

            img_path = os.path.join(FACE_DATA_DIR_PATH, dir_name, file_name)
            image = cv2.imread(img_path)

            face_images = detect_faces(img_path)

            if len(face_images) != 1:
                continue

            result_path = os.path.join(FACE_DATA_DIR_PATH, dir_name, file_name)
            cv2.imwrite(result_path, face_images[0])
