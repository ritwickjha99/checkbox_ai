import cv2
import glob
from PIL import Image
import os
from os import makedirs
import sys 
# from __future__ import print_function
import pickle 
import os.path 
import io 
import shutil 
import requests
from googleapiclient.discovery import build 
from google_auth_oauthlib.flow import InstalledAppFlow 
from google.auth.transport.requests import Request 
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload

from processing_module.scripts.common import utils
from processing_module.scripts.common.Logger import Logger
from processing_module.scripts.common import keywords
import warnings
import io
warnings.filterwarnings("ignore")


class libHelper:

    def __int__(self):
        # self.logger.info("Initiated libHelper")
        self.utility = utils.utils()
        self.Config = self.utility.getConfig()
        # self.logger = Logger().getLoggerInstance(name=self.__class__.__name__)
        self.SCOPES = [self.Config.get(keywords.PREDICT, keywords.SERVE_URL)]
        self.creds = None
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                self.creds = pickle.load(token) 

        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token: 
                self.creds.refresh(Request()) 
            else: 
                flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES) 
                self.creds = flow.run_local_server(port=0) 

            with open('token.pickle', 'wb') as token: 
                pickle.dump(self.creds, token) 

        self.service = build('drive', 'v3', credentials=self.creds) 

        results = self.service.files().list(pageSize=100, fields="files(id, name)").execute() 
        items = results.get('files', [])
        print("items from libhelper: " + str(items))
        print("Completed libHelper initiation.")

    """
        created by- Achyuth Pasupuleti 
        email id- achyuth.pasupuleti@kizora.com
        purpose- python script to connect google drive
        input- needs credentials to login, file_id, file_name
        output- will print all files and download the file 
    """
    def FileDownload(self, file_id, file_name):
        print("Started FileDownload.")
        request = self.service.files().get_media(fileId=file_id) 
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request, chunksize=204800) 
        done = False
        try:
            while not done: 
                status, done = downloader.next_chunk()
            fh.seek(0)
            with open(file_name, 'wb') as f: 
                shutil.copyfileobj(fh, f)
            print("File Downloaded.")
            print().logger.info("Completed FileDownload.")
        except Exception as e:
            print("Failed at FileDownload. Error is: " + str(e))

    """
        Creator -Loesh Mogali
        Email - lokesh.mogali@kizora.com
        Purpose - To rename all the images in the input directory to the required format(ex:000001,000002,..)
        Input - it needs an input folder path and an input image format
        Output - all the images in the given directory will be renamed.
        Syntax - python rename_images.py input_images_directory_path format
    """
    def rename_images(self, imgdir, imgformat):
        try:
            print("Started rename_images.")
            if(imgformat=='JPEG'):
                n = 0
                for imfile in os.scandir(imgdir):
                    os.rename(imfile.path, os.path.join(imgdir, '{:06}.jpeg'.format(n)))
                    n += 1
            elif(imgformat=='PNG'):
                n = 0

                for imfile in os.scandir(imgdir):
                    #rename in the format of 000001.jpeg
                    os.rename(imfile.path, os.path.join(imgdir, '{:06}.PNg'.format(n)))
                    n += 1
            elif(imgformat=='JPG'):
                n = 0
                for imfile in os.scandir(imgdir):
                    os.rename(imfile.path, os.path.join(imgdir, '{:06}.jpg'.format(n)))
                    n += 1
            else:
                print("File format not supported.")
            print("Completed rename_images.")
        except Exception as e:
            print("Failed at rename_images. Error is: " + str(e))

    """
        Creator - Ritwick Jha
        Email - ritwick.jha@kizora.com
        Purpose - To resize all the images in the input directory to 20x20 and convert it to grayscale
        Input - it needs an input folder path
        Output - it resizes the images into 20x20 and converts the images to grayscale
        Syntax - python resize_images.py input_images_directory_path
    """
    def resize_images(self, path):
        try:
            print("Started resize_images.")
            # Destination Folder
            dstpath = path + "/output/"
            try:
                makedirs(dstpath)
            except:
                dstpath = path + "/output/"

            image_height = 20
            image_width = 20
            # array of images in the directory
            images = glob.glob(path + "/*")

            for i in images:
                img = Image.open(i)
                # To resize the image
                img = img.resize((image_height, image_width)) 
                # To convert the image to grayscale
                img = img.convert('L')  
                base = os.path.basename(i)
                img.save(dstpath + base)
            print("Completed resize_images.")
        except Exception as e:
            print("Failed at resize_images. Error: " + str(e))
