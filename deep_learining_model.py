# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 23:42:00 2023

@author: Ritwik Ranjan Pathak
"""

from tensorflow.keras.applications import MobileNetV2, InceptionV3, InceptionResNetV2
from tensorflow.keras.layers import Dense,Dropout,Flatten,Input
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard


from Object_detection import * 
from xml_to_csv import *
from data_preprocessing import *

'using the most advance deep learning model'
inception_resnet= InceptionResNetV2(weights="imagenet",include_top=False,
                                    input_tensor=Input(shape=(224,224,3)))
inception_resnet.trainable=False

headmodel=inception_resnet.output
headmodel=Flatten()(headmodel)
headmodel=Dense(500,activation="relu")(headmodel)
headmodel=Dense(250,activation="relu")(headmodel)


'using sigmoid activation function'
headmodel=Dense(4,activation='sigmoid')(headmodel)

'making the input and output model'
model=Model(inputs=inception_resnet.input,outputs=headmodel)

'compiling the model'
model.compile(loss='mse',optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5))
print(model.summary)

'testing the value'

tfb=TensorBoard('object_detection')
history=model.fit(x=x_train,y=y_train,batch_size=10,epochs=200,
                  validation_data=(x_test,y_test),callbacks=[tfb],initial_epoch=100)

'saving the model'
model.save('./models/object_detection.h5')
