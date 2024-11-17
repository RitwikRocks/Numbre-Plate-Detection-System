# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 20:43:11 2023

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



from Object_detection import * 
from xml_to_csv import *

labels=df.iloc[:,1:].values
image=image_path[0]
img_arr=cv2.imread(image)
h,w,d=img_arr.shape

#preprocessing
'target size is important for object detection'
load_image=load_img(image,target_size=(224,224))
load_image_arr=img_to_array
#normalization to labels

data=[]
output=[]

for ind in range(len(image_path)):
    image=image_path[ind]
    img_arr=cv2.imread(image)
    h,w,d=img_arr.shape
    load_image=load_img(image,target_size=(224,224))
    load_image_arr=img_to_array(load_image)
    norm_load_image_arr=load_image_arr/255.0
    xmin,xmax,ymin,ymax= labels[ind]
    nxmin,nxmax=xmin/w,xmax/w
    nymin,nymax=ymin/h,ymax/h
    label_norm=(nxmin,nxmax,nymin,nymax)
    data.append(norm_load_image_arr)
    output.append(label_norm)
    
    
'testing and training the model(LINEAR REGRESSION)'
X=np.array(data,dtype=np.float32)
Y=np.array(output,dtype=np.float32)
#print(X.shape,Y.shape)
'training and testing (training 0.8 and testing 0.2'
x_train,x_test,y_train,y_test=train_test_split(X,Y,train_size=0.8,random_state=0)
#print(x_train.shape,x_test.shape)