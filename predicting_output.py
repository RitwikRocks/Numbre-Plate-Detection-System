# -*- coding: utf-8 -*-
"""
Created on Mon May  1 08:23:14 2023

@author: Ritwik Ranjan Pathak
"""

import cv2
import numpy as np
import os
import pytesseract as pt

#settings
INPUT_WIDTH=640
INPUT_HEIGHT=640

#loading the image
img=cv2.imread("E:/NUMBER_PLATE_DETECTION_USING_YOLO/YOLO-5/data_images/test/N66.jpeg")

#cv2.namedWindow("test_image",cv2.WINDOW_KEEPRATIO)

#Load YOLO MODEL
net = cv2.dnn.readNetFromONNX('E:/NUMBER_PLATE_DETECTION_USING_YOLO/YOLO-5/Model-20230501T025034Z-001/Model/weights/best.onnx')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

#converting image into YOLO FORMAT
image=img.copy()
row,col,d=image.shape
max_rc=max(row,col)
input_image=np.zeros((max_rc,max_rc,3),dtype=np.uint8)
input_image[0:row,0:col]=image

#cv2.imshow('test_image',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#get predictions from YOLO MODEL
blob = cv2.dnn.blobFromImage(input_image,1/255,(INPUT_WIDTH,INPUT_HEIGHT),swapRB=True,crop=False)
net.setInput(blob)
preds = net.forward()
detections = preds[0]

#detections.shape()

#Filter detection based on confidence and Probability Score
#center_x, center_y,width,height,confidence,probability
boxes=[]
confidences=[]

image_w, image_h = input_image.shape[:2]
x_factor = image_w/INPUT_WIDTH
y_factor = image_h/INPUT_HEIGHT
    
for i in range(len(detections)):
    row=detections[i]
    confidence=row[4]
    if confidence>0.4:
        class_score=row[5]
        cx, cy , w, h = row[0:4]

        left = int((cx - 0.5*w)*x_factor)
        top = int((cy-0.5*h)*y_factor)
        width = int(w*x_factor)
        height = int(h*y_factor)
        box = np.array([left,top,width,height])
        
        confidences.append(confidence)
        boxes.append(box)
        
# clean
boxes_np = np.array(boxes).tolist()
confidences_np = np.array(confidences).tolist()
# NMS
index = cv2.dnn.NMSBoxes(boxes_np,confidences_np,0.25,0.45).flatten()
    
# drawings
for ind in index:
    x,y,w,h =  boxes_np[ind]
    bb_conf = confidences_np[ind]
    conf_text = 'plate: {:.0f}%'.format(bb_conf*100)
    #license_text = extract_text(image,boxes_np[ind])


cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,255),2)
cv2.rectangle(image,(x,y-30),(x+w,y),(255,0,255),-1)
cv2.rectangle(image,(x,y+h),(x+w,y+h+30),(0,0,0),-1)


cv2.putText(image,conf_text,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),1)
#cv2.putText(image,license_text,(x,y+h+27),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),1)

cv2.imshow('res',image)
cv2.waitKey(0)
cv2.destroyAllWindows()