# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 17:26:41 2023

@author: Ritwik Ranjan Pathak
"""

import numpy as np
import cv2
import matplotlib.pyplot  as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

model=tf.keras.models.load_model('E:/NUMBER_PLATE_DETECTION_USING_YOLO/Python_coding/models/object_detection.h5')
print("MODEL LOADED SUCESSFULLY")

def object_detection(path):    
    'create pipeline'
    'read image'
    image=load_img(path)
    image=np.array(image,dtype=np.uint8)  #8 bit array
    
    image1=load_img(path,target_size=(224,224))
    image_arr_224=img_to_array(image1)/225.0   #convert to image and get the normalized value
    h,w,d=image.shape
    test_arr=image_arr_224.reshape(1,224,224,3)
    test_arr.shape
    coords=model.predict(test_arr)
    
    'denormalize the output'
    denorm=np.array([w,w,h,h])
    coords=coords*denorm
    coords=coords.astype(np.int32)
    print(coords)
    
    
    
    'drawing bounding box on the image'
    xmin,xmax,ymin,ymax=coords[0]
    pt1=(xmin,ymin)
    pt2=(xmax,ymax)
    print("Boundary box of the image")
    print(pt1,pt2)
    cv2.rectangle(image,pt1,pt2,(0,255,0),3)
    #plt.figure(figsize=(10,8))
    #plt.imshow(image)
    #plt.show(image)
    #plt.show()
    return image,coords