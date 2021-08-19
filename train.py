"""
Author: Hemant Devidas Kshirsagar
Date: 28/03/2021
Objective: To impplement a Convolutional Neural Network(CNN) using Keras and Tensorflow to predict checked and unchecked images.
"""
import time
s = time.time()
import pandas as pd
import numpy as np
from processing_module.scripts.common import utils
from processing_module.scripts.common.Logger import Logger
from processing_module.scripts.common import keywords
from processing_module.scripts.common.libhelper import libHelper
import warnings
warnings.filterwarnings("ignore")

from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import mlflow
from sklearn.metrics import classification_report, confusion_matrix

import math
import seaborn as sns
import matplotlib.pyplot as plt

import typing

import random
import cv2

from PIL import Image
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import applications
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
import os
import fnmatch
import numpy as np
import time
from keras.preprocessing import image
from keras.models import load_model
import matplotlib.pyplot as plt
from keras.constraints import maxnorm
import splitfolders


class Train:
    def __init__(self):
        self.utility = utils.utils()
        self.Config = self.utility.getConfig()
        self.logger = Logger().getLoggerInstance(name=self.__class__.__name__)

        # self.lib_helper = libHelper()

        # Config Variables
        self.experiment_name = self.Config.get(keywords.TRAIN, keywords.EXPERIMENT_NAME)
        self.image_width = int(self.Config.get(keywords.TRAIN, keywords.IMAGE_WIDTH))
        self.image_height = int(self.Config.get(keywords.TRAIN, keywords.IMAGE_HEIGHT))
        self.data_dir = self.utility.getCompletepath(self.Config.get(keywords.TRAIN, keywords.DATA_DIR))
        self.raw_input_data_dir = self.utility.getCompletepath(self.Config.get(keywords.TRAIN, keywords.RAW_INPUT_DATA_DIR))
        self.train_data_dir = self.utility.getCompletepath(self.Config.get(keywords.TRAIN, keywords.TRAIN_DATA_DIR))
        self.validation_data_dir = self.utility.getCompletepath(self.Config.get(keywords.TRAIN, keywords.VALIDATION_DATA_DIR))

        self.nb_train_samples = int(self.Config.get(keywords.TRAIN, keywords.NB_TRAIN_SAMPLES))
        self.nb_validation_samples = int(self.Config.get(keywords.TRAIN, keywords.NB_VALIDATION_SAMPLES))
        
        self.batch_size = int(self.Config.get(keywords.TRAIN, keywords.BATCH_SIZE))
        self.epochs = int(self.Config.get(keywords.TRAIN, keywords.EPOCHS))

        self.success = True

    def save_conf(self, conf_matrix, categories):
        fig = plt.figure(figsize=(9, 6))
        df_cm = pd.DataFrame(
            conf_matrix,
            index=[i for i in categories],
            columns=[i for i in categories]
        )
        ax = sns.heatmap(df_cm, annot=True, cmap="Blues", fmt='g')
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        plt.ylabel('Actual/Real')
        plt.xlabel('Predicted')
        fig.savefig("confusion-matrix.png")

    def train_cnn(self):
        try:
            if K.image_data_format() == 'channels_first':
                input_shape = (3, self.image_width, self.image_height)
            else:
                input_shape = (self.image_width, self.image_height, 3)

            mlflow.log_param("batch_size", self.batch_size)
            mlflow.log_param("epochs", self.epochs)
            mlflow.log_param("input_shape", input_shape)

            model = Sequential()
            model.add(Conv2D(32, (3, 3), input_shape=input_shape))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Conv2D(32, (3, 3)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Flatten())

            model.add(Dense(128))
            model.add(Activation('relu'))

            model.add(Dropout(0.4))
            model.add(Dense(2))

            model.add(Activation('softmax'))

            model.compile(loss='categorical_crossentropy',
                          # optimizer='rmsprop',
                          optimizer='adam',
                          metrics=['accuracy'])

            modelcheckpoint = ModelCheckpoint('checkbox_model.hdf5', monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
            tbcheckpoint = TensorBoard(log_dir='logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

            train_datagen = ImageDataGenerator(
                rescale=1./255,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True)

            test_datagen = ImageDataGenerator(rescale=1./255)

            train_generator = train_datagen.flow_from_directory(
                self.train_data_dir,
                target_size=(self.image_width, self.image_height),
                batch_size=self.batch_size,
                class_mode='categorical')

            validation_generator = test_datagen.flow_from_directory(
                self.validation_data_dir,
                target_size=(self.image_width, self.image_height),
                batch_size=self.batch_size,
                class_mode='categorical')

            history = model.fit(
                train_generator,
                steps_per_epoch=self.nb_train_samples // self.batch_size,
                epochs=self.epochs,
                validation_data=validation_generator,
                validation_steps=self.nb_validation_samples // self.batch_size,
                callbacks=[modelcheckpoint, tbcheckpoint]) # uncomment this if you want to log into tensorboard

            # history.history['accuracy']
            model.save('model.h5')

            mlflow.log_metric("accuracy", history.history['accuracy'][self.epochs - 1])
            mlflow.log_metric("loss", history.history['loss'][self.epochs - 1])

            # confusion matrix & report
            Y_pred = model.predict_generator(validation_generator, self.nb_validation_samples // self.batch_size + 1)
            y_pred = np.argmax(Y_pred, axis=1)
            confusionmatrix = confusion_matrix(validation_generator.classes, y_pred)
            # print(confusionmatrix)
            # print(y_pred)

            target_names = list(validation_generator.class_indices.keys())
            classificationreport = classification_report(validation_generator.classes, y_pred,
                                                         target_names=target_names)
            with open("classification_report.txt", "w") as f:
                f.write(classificationreport)

            self.save_conf(confusionmatrix, target_names)

            mlflow.log_artifact('confusion-matrix.png')
            mlflow.log_artifact('classification_report.txt')

        except Exception as e:
            self.logger.error("Failed at train_cnn: " + str(e))

    def split_data(self):
        try:
            splitfolders.ratio(self.raw_input_data_dir, output=self.data_dir, seed=1337, ratio=(.8, .2), group_prefix=None)
        except Exception as e:
            self.logger.error("Exception at split_data: " + str(e))

    def transfer_learning_with_vgg19(self):
        try:
            if K.image_data_format() == 'channels_first':
                input_shape = (3, self.image_width, self.image_height)
            else:
                input_shape = (self.image_width, self.image_height, 3)

            mlflow.log_param("batch_size", self.batch_size)
            mlflow.log_param("epochs", self.epochs)
            mlflow.log_param("input_shape", input_shape)

            vgg_model = applications.VGG19(weights='imagenet',
                                           include_top=False,
                                           # input_shape=input_shape)
                                           input_shape=(160, 160, 3))

            # Creating dictionary that maps layer names to the layers
            layer_dict = dict([(layer.name, layer) for layer in vgg_model.layers])

            # Getting output tensor of the last VGG layer that we want to include
            x = layer_dict['block2_pool'].output

            # Stacking a new simple convolutional network on top of it
            x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
            x = Flatten()(x)
            x = Dense(128, activation='relu')(x)
            x = Dropout(0.5)(x)
            x = Dense(2, activation='softmax')(x)

            # Creating new model. Please note that this is NOT a Sequential() model.
            from keras.models import Model
            model = Model(inputs=vgg_model.input, outputs=x)

            print("test")

            # Make sure that the pre-trained bottom layers are not trainable
            for layer in model.layers[:7]:
                layer.trainable = False

            # Do not forget to compile it
            model.compile(loss='categorical_crossentropy',
                                 optimizer='adam',
                                 metrics=['accuracy'])

            modelcheckpoint = ModelCheckpoint('checkbox_model.hdf5', monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
            tbcheckpoint = TensorBoard(log_dir='logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

            train_datagen = ImageDataGenerator(
                rescale=1./255,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True)

            test_datagen = ImageDataGenerator(rescale=1./255)

            train_generator = train_datagen.flow_from_directory(
                self.train_data_dir,
                target_size=(self.image_width, self.image_height),
                batch_size=self.batch_size,
                class_mode='categorical')

            validation_generator = test_datagen.flow_from_directory(
                self.validation_data_dir,
                target_size=(self.image_width, self.image_height),
                batch_size=self.batch_size,
                class_mode='categorical')

            history = model.fit(
                train_generator,
                steps_per_epoch=self.nb_train_samples // self.batch_size,
                epochs=self.epochs,
                validation_data=validation_generator,
                validation_steps=self.nb_validation_samples // self.batch_size,
                callbacks=[modelcheckpoint, tbcheckpoint]) # uncomment this if you want to log into tensorboard

            # history.history['accuracy']
            model.save('model.h5')

            mlflow.log_metric("accuracy", history.history['accuracy'][self.epochs - 1])
            mlflow.log_metric("loss", history.history['loss'][self.epochs - 1])

            # confusion matrix & report
            Y_pred = model.predict_generator(validation_generator, self.nb_validation_samples // self.batch_size + 1)
            y_pred = np.argmax(Y_pred, axis=1)
            confusionmatrix = confusion_matrix(validation_generator.classes, y_pred)
            # print(confusionmatrix)
            # print(y_pred)

            target_names = list(validation_generator.class_indices.keys())
            classificationreport = classification_report(validation_generator.classes, y_pred,
                                                         target_names=target_names)
            with open("classification_report.txt", "w") as f:
                f.write(classificationreport)

            self.save_conf(confusionmatrix, target_names)

            mlflow.log_artifact('confusion-matrix.png')
            mlflow.log_artifact('classification_report.txt')
        except Exception as e:
            self.logger.error("Failed at transfer_learning_with_vgg19: " + str(e))

    def run(self):
        try:
            self.logger.info("start Training")

            mlflow.tensorflow.autolog()
            mlflow.set_experiment(experiment_name=self.experiment_name)
            start_time = time.time()

            # lib_helper = libHelper()
            # lib_helper.rename_images(self.raw_input_data_dir + "/checked", "JPG")
            # lib_helper.rename_images(self.raw_input_data_dir + "/unchecked", "JPG")

            # self.split_data()

            self.nb_train_samples = int(len(os.listdir(self.train_data_dir + "/checked"))) + int(len(os.listdir(self.train_data_dir + "/unchecked")))
            self.nb_validation_samples = int(len(os.listdir(self.validation_data_dir + "/checked"))) + int(len(os.listdir(self.validation_data_dir + "/unchecked")))

            print("Number of training samples: " + str(self.nb_train_samples))
            self.logger.info("Number of training samples: " + str(self.nb_train_samples))
            print("Number of validation samples: " + str(self.nb_validation_samples))
            self.logger.info("Number of validation samples: " + str(self.nb_validation_samples))

            # self.train_cnn()

            self.transfer_learning_with_vgg19()

            end_time = time.time()
            self.logger.info("Processed in: %s seconds" % (end_time - start_time))
            self.logger.info("ends Training")
        except Exception as e:
            self.success = False
            self.logger.error("Failed at Training step. Error is: " + str(e))

        return self.success
