# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 17:41:37 2023

@author: Ritwik Ranjan Pathak
"""

import pandas as pd
import xml.etree.ElementTree as xet

'it is used to file path from the pc'
from glob import glob

path=glob('E:/NUMBER_PLATE_DETECTION_USING_YOLO/Cars_with_number_plate/*.xml')

'parsing the first element from xml to csv by extracting the data'
#filename=path[0]


'function to parse'
labels_dict=dict(filepath=[],xmin=[],xmax=[],ymin=[],ymax=[])

for filename in path:
    info=xet.parse(filename)
    root=info.getroot()
    member_object=root.find('object')
    labels_info=member_object.find('bndbox')
    xmin=int(labels_info.find('xmin').text)   #typecasting string to int'
    xmax=int(labels_info.find('xmax').text)
    ymin=int(labels_info.find('ymin').text)
    ymax=int(labels_info.find('ymax').text)
    labels_dict['filepath'].append(filename)
    labels_dict['xmin'].append(xmin)
    labels_dict['xmax'].append(xmax)
    labels_dict['ymin'].append(ymin)
    labels_dict['ymax'].append(ymax)
    
    
'converting dictionary to pandas dataframe for good readability'
df=pd.DataFrame(labels_dict)

'saving all the DataFram of Pandas in CSV'
df.to_csv('labels.csv',index=False)
    

#print(xmin,xmax,ymin,ymax)
