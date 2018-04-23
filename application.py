# -*- coding:utf-8 -*-
import cv2
import sys

from sentiment_classifier import SentimentClassifier
from face_detector import detect_faces, show_detection_result

if __name__ == '__main__':

    model = SentimentClassifier()

    #model.train()
    #model.save()

    model.load()

    img_path = input('\n Enter img path:')

    face_imgs = detect_faces(img_path)

    if len(face_imgs) != 1:
        print('Detect no faces.')
        sys.exit()

    result = model.predict(img_path)
    for item in result:
        print('{0} probability：{1}'.format(item['label'], item['prob']))
