# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 10:53:40 2023

@author: Ritwik Ranjan Pathak
"""

''' limitations of object detection model
1.Low precision of object detection
2. Slow processing of the model

In YOLO model we have to use same data set but instread of xmin,xmax,ymin,ymax we have to use
center_x,center_y,width and height of the boundary box

We have to use data_image in which both train image and test image will be there
the train and test data both have a jpg file and a text file about the data
and the text file shoukd be named same ads the image file
'''

import numpy as np
import pandas as pd
from glob import glob
import xml.etree.ElementTree as xet
import cv2
from shutil import copy
import os

df=pd.read_csv('E:/NUMBER_PLATE_DETECTION_USING_YOLO/YOLO-5/labels.csv')
df.head()

#parsing
path="./images/N1.xml"
def parsing(path):
    parser=xet.parse(path).getroot()
    name=parser.find('filename').text
    filename='E:/NUMBER_PLATE_DETECTION_USING_YOLO/YOLO-5/data_images/test'
    parser_size=parser.find('size')
    width=int(parser_size.find('width').text)
    height=int(parser_size.find('height').text)
    return filename,width,height

df[['filename','width','height']]=df['filepath'].apply(parsing).apply(pd.Series)
df['center_x']=(df['xmax']+df['xmin'])/(2*df['width'])
df['center_y']=(df['ymax']+df['ymin'])/(2*df['height'])
df['bb_width']=(df['xmax']-df['xmin'])/df['width']
df['bb_height']=(df['ymax']-df['ymin'])/df['height']

df.head()
#print(width,height)



'''split data into train and testing'''
df_train=df.iloc[:200]
df_test=df.iloc[200:]

'''test file
class_id
center_x,center_y
bb_width,bb_height
'''

'''creating a text file'''
train_folder='E:/NUMBER_PLATE_DETECTION_USING_YOLO/YOLO-5/data_images/train'
values=df_train[['filename','center_x','center_y','bb_width','bb_height']].values

for fname, x,y, w, h in values:
    image_name=os.path.split(fname)[-1]
    txt_name=os.path.splitext(image_name)[0]
    
    dst_image_path=os.path.join(train_folder,image_name)
    dst_label_file=os.path.join(train_folder,txt_name+'.txt')
    '''copy each image to folder'''
   
    copy(fname,dst_image_path)


    '''
    generate text file with label inforation
     '''
    label_txt = f'0 {x} {y} {w} {h}'
    with open(dst_label_file,mode='w') as f:
        f.write(label_txt)
        
        f.close()
        
        
        
''' Difficukties in YOLO TRAINING '''
''' 1. hardware is major chalange for YOLO
2.YOLO REQUIRE FAST GPU '''


        