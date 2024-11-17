# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 18:48:31 2023

@author: Ritwik Ranjan Pathak
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import xml.etree.ElementTree as xet


from sklearn.model_selection  import train_test_split
'used for object detection model'
from tensorflow.keras.preprocessing.image import load_img, img_to_array

'reading the csv file'
df=pd.read_csv('labels.csv')
print(df.head())

'extracting the filename of the file like N1'

'''
filename=df['filepath'][0]
name=xet.parse(filename).getroot().find('filename').text
print(name)
'''
def getFileName(filename):
    filename_image=xet.parse(filename).getroot().find('filename').text
    print(filename_image)
    filepath_image=os.path.join('E:/NUMBER_PLATE_DETECTION_USING_YOLO/Cars_with_number_plate/',filename_image)
    return filepath_image





df['filepath'].apply(getFileName)
image_path=list(df['filepath'].apply(getFileName))
#print(images_path)


'verifying image and output'

file_path=image_path[0]
img=cv2.imread(file_path)

'making a named window whose size can be changed for our display'
'''
cv2.namedWindow('example',cv2.WINDOW_NORMAL)
cv2.imshow('example',img)
'if we press esc then the exit happen'
cv2.waitKey(0)
cv2.destroyAllWindow()
'''

'to show rectangular box in the image'
def call():
    cv2.rectangle(img,(1093,645),(1396,727),(0,255,0),5)
    cv2.namedWindow('example',cv2.WINDOW_NORMAL)
    cv2.imshow('example',img)
    'if we press esc then the exit happen'
    cv2.waitKey(0)
    cv2.destroyAllWindow()








