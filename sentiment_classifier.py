# -*- coding:utf-8 -*-

from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import SGD
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np
from PIL import Image
import cv2

from face_detector import detect_faces


class SentimentClassifier(object):
    IMG_SIZE = 200
    NB_CLASSES = 5
    TRAIN_DATA_DIR = './data/face_images/train/'
    VALIDATION_DATA_DIR = './data/face_images/validation/'
    MODEL_PATH = './model/SentimentClassifier/model.h5'

    def __init__(self, img_size=IMG_SIZE, nb_classes=NB_CLASSES):
        self.img_size = img_size

        resnet = ResNet50(include_top=False,
                          input_shape=(img_size, img_size, 3),
                          weights="imagenet")
        h = Flatten()(resnet.output)
        model_output = Dense(nb_classes, activation="softmax")(h)
        self.model = Model(resnet.input, model_output)

    def train(self, train_data_dir=TRAIN_DATA_DIR, validation_data_dir=VALIDATION_DATA_DIR):
        train_datagen = ImageDataGenerator(channel_shift_range=100,
                                           horizontal_flip=True)
        validation_datagen = ImageDataGenerator()

        train_generator = train_datagen.flow_from_directory(
                                train_data_dir,
                                target_size=(self.img_size, self.img_size),
                                batch_size=32,
                                class_mode='categorical')

        validation_generator = validation_datagen.flow_from_directory(
                                validation_data_dir,
                                target_size=(self.img_size, self.img_size),
                                batch_size=32,
                                class_mode='categorical')

        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd)

        self.model.fit_generator(
                                train_generator,
                                steps_per_epoch=155,
                                epochs=4,
                                validation_data=validation_generator,
                                validation_steps=200)

    def predict(self, img_path):
        face_imgs = detect_faces(img_path)
        face_img = face_imgs[0]

        cv2.imshow('face', face_img)

        #convert OpenCV to PIL format
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = Image.fromarray(face_img)

        face_img = face_img.resize((self.img_size, self.img_size))

        x = img_to_array(face_img)
        x = np.expand_dims(x, axis=0)

        result = self.model.predict(x)

        return result

    def load(self, model_path=MODEL_PATH):
        print('Model Loaded.')
        self.model = load_model(model_path)

    def save(self, model_path=MODEL_PATH):
        print('Model Saved.')
        self.model.save(model_path)


if __name__ == '__main__':

    model = SentimentClassifier()

    #model.train()
    #model.save()

    model.load()

    while(True):
        img_path = input('Enter img path:')
        result_class = model.predict(img_path)
        print(result_class)
