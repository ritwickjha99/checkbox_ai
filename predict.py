import time

s = time.time()
import datetime
import pandas as pd
from processing_module.scripts.common import utils
from processing_module.scripts.common.Logger import Logger
from processing_module.scripts.common import keywords
import json
import ast
import requests
from keras.preprocessing import image
import numpy as np
from os import listdir
from os.path import isfile, join
import warnings
from PIL import Image

warnings.filterwarnings("ignore")
import os
from boxdetect import config
from boxdetect.pipelines import get_boxes
import cv2


class Predict:
    data = {}
    def __init__(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        self.utility = utils.utils()
        self.Config = self.utility.getConfig()
        self.logger = Logger().getLoggerInstance(name=self.__class__.__name__)

        # Config Variables
        self.url = str(self.Config.get(keywords.PREDICT, keywords.SERVE_URL))
        self.headers = {'Content-type': 'application/json'}
        self.classes = ast.literal_eval(self.Config.get(keywords.PREDICT, keywords.CLASSES))
        self.mypath = self.utility.getCompletepath(self.Config.get(keywords.PREDICT, keywords.PREDICT_IMAGE_PATH))
        self.filepath = self.utility.getCompletepath(self.Config.get(keywords.PREDICT, keywords.PREDICT_FILE_PATH))
        self.success = True

    def run(self):
        try:
            self.logger.info("start predicting")
            # Time-stamp needed to measure performance
            start_time = time.time()

            try:
                files = [fi for fi in listdir(self.filepath) if isfile(join(self.filepath, fi))]
                for f in files:
                    file_path = self.filepath + f
                    break
                image1 = cv2.imread(file_path)
                dimensions = image1.shape
                height = image1.shape[0]
                width = image1.shape[1]
                cfg = config.PipelinesConfig()
                cfg.width_range = (15, 70)
                cfg.height_range = (15, 70)
                cfg.scaling_factors = [0.7]
                cfg.wh_ratio_range = (0.5, 1.7)
                cfg.group_size_range = (1, 1)
                cfg.dilation_iterations = 0
                kernel1 = np.ones((2, 2), np.uint8)
                kernel2 = np.ones((1, 1), np.uint8)
                kernel3 = np.ones((1, 2), np.uint8)

                image2 = cv2.dilate(image1, kernel1)
                image3 = cv2.erode(image2, kernel2)
                image4 = cv2.dilate(image3, kernel3)
                rects, grouping_rects, image2, output_image = get_boxes(image4, cfg=cfg, plot=False)
                path = self.mypath
                log_date_fmt = "%Y-%m-%d %H:%M:%S.%f"
                time_s = datetime.datetime.now().strftime(log_date_fmt)
                print(str(time_s) + "hi")

                tim_stamp = int(start_time)
                directory = str(tim_stamp)
                path = path + directory
                os.makedirs(path)
                newpath = path + "_text"
                os.makedirs(newpath)

                count = 0;
                x_coord = [0]
                y_coord = [0]
                for x, y, w, h in rects[0:]:
                    cropped = image1[y - 8:y + h + 8, x - 8:x + w + 8]
                    path = path + "/"
                    x_coord.append(x)
                    y_coord.append(y)
                    x_coord.append(x + w)
                    y_coord.append(y + w)
                    cv2.imwrite(path + str(count) + ".png", cropped)
                    count = count + 1
                self.logger.info("start predicting from cropped images")
                x_coord.sort()
                y_coord.sort()
                x_fin = [x_coord[0]]
                y_mid = [y_coord[0]]
                for i in range(len(x_coord)):
                    if x_coord[i] not in range(x_fin[-1] - 5, x_fin[-1] + 5):
                        x_fin.append(x_coord[i])
                for i in range(len(y_coord)):
                    if y_coord[i] not in range(y_mid[-1] - 5, y_mid[-1] + 5):
                        y_mid.append(y_coord[i])
                y_fin = []
                for i in range(0, len(y_mid), 2):
                    if i == len(y_mid) - 1:
                        mid = (y_mid[i] + height) // 2
                    else:
                        mid = (y_mid[i] + y_mid[i + 1]) // 2
                    y_fin.append(mid)
                x_fin.append(width)
                if len(x_coord) > 3:
                    width = x_fin[3]
                if x_fin[1] - 0 < width - x_fin[0]:
                    for i in range(2,len(x_fin), 2):
                        for j in range(len(y_fin) - 1):
                            cropped = image1[y_fin[j]: y_fin[j + 1], x_fin[i] + 8:x_fin[i + 1] - 8]
                            newpath = newpath + "/"
                            cv2.imwrite(newpath + str(count) + ".png", cropped)
                            count += 1

                onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
                for file in onlyfiles:
                    img = image.load_img(path + file, target_size=(160, 160))
                    x = image.img_to_array(img)
                    x = np.expand_dims(x, axis=0)
                    data = json.dumps({"instances": x.tolist()})
                    json_response = requests.post(self.url, data=data, headers=self.headers)
                    predictions = json.loads(json_response.text)
                    filename = os.path.splitext(file)
                    # print(filename[0])
                    my_source = path + file
                    # print(my_source)
                    my_dest = filename[0] + "_" + self.classes[round(predictions[0][0])] + ".png"
                    my_dest = path + my_dest
                    # print(my_dest)
                    self.json_format('{}'.format(file), self.classes[round(predictions[0][0])])
                    os.rename(my_source, my_dest)
                    self.logger.info('{}'.format(file) + " - " + self.classes[round(predictions[0][0])])
                self.json_output()
                print("Check execution.log file for the prediction results.")
                '''
                onlyfiles = [f for f in listdir(self.mypath) if isfile(join(self.mypath, f))]


                for file in onlyfiles:

                    img = image.load_img(self.mypath + file, target_size=(20, 20))
                    x = image.img_to_array(img)
                    x = np.expand_dims(x, axis=0)
                    data = json.dumps({"instances": x.tolist()})
                    json_response = requests.post(self.url, data=data, headers=self.headers)
                    predictions = json.loads(json_response.text)
                    filename=os.path.splitext(file)
                    #print(filename[0])
                    my_source =self.mypath + file
                    #print(my_source)
                    my_dest =filename[0] +"_"+ self.classes[round(predictions[0][0])]+".png"
                    my_dest =self.mypath + my_dest
                    #print(my_dest)

                    os.rename(my_source, my_dest)
                    self.logger.info('{}'.format(file) + " - " + self.classes[round(predictions[0][0])])
                print("Check execution.log file for the prediction results.")

                '''
                '''
                for file in onlyfiles:
                    img = image.load_img(self.mypath + file, target_size=(160, 160))
                    x = image.img_to_array(img)
                    x = np.expand_dims(x, axis=0)
                    data = json.dumps({"instances": x.tolist()})
                    json_response = requests.post(self.url, data=data, headers=self.headers)
                    predictions = json.loads(json_response.text)
                    self.logger.info('{}'.format(file) + " - " + self.classes[round(predictions[0][0])])
                print("Check execution.log file for the prediction results.")
                '''
            except Exception as e:
                self.logger.error("Failed at predicting step. Error is: " + str(e))

            self.logger.info('Done predicting Processing')
            end_time = time.time()

            self.logger.info("Processed in: %s seconds" % (end_time - start_time))
            self.logger.info("ends predicting")
        except Exception as e:
            self.success = False
            self.logger.error("Failed at predicting step. Error is: " + str(e))

        return self.success


    def json_format(self, file, classification):
        self.data[file] = classification

    def json_output(self):
        with open('output.json', 'w') as outfile:
            json.dump(self.data, outfile)