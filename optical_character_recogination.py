# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 17:48:42 2023

@author: Ritwik Ranjan Pathak
"""
import pytesseract as pt
import numpy as np
import cv2
import matplotlib.pyplot  as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard


from creating_piping_model import *

path="E:/NUMBER_PLATE_DETECTION_USING_YOLO/Cars_with_number_plate/N1.jpeg"
image,cods=object_detection(path)
plt.figure(figsize=(10,8))
plt.imshow(image)
plt.show()

img=np.array(load_img(path))
xmin,xmax,ymin,ymax=cods[0]
'region of interest and cropping the image'
roi=img[ymin:ymax,xmin:xmax]
plt.imshow(roi)
plt.show()


'EXTRACTING TEXT FROM THE IMAGE'
text=pt.image_to_string(roi)
print(text)


'''limitations of pytesseract 
1. text should not be roatated or tilted at an angle. It gives wrong result
2. text should not be blur
3.their should not be any effect in the text
4.it should not work for handwritten text or cursive handwritting
5.resolution should be atleast 200dpi or width and height be 300px 
'''