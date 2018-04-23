#-*- coding:utf-8 -*-

import os

from face_detector import detect_faces

def cut_and_save_face_imgs():

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
