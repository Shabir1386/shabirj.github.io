sha# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Created on Tue Jan 15 18:41:57 2019

@author: shabirjameel
"""

# Part 1 - Building the CNN 
# To train the model with 1000 images

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import theano
import tensorflow as tf

# Importing the Keras libraries and packages  
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

# Initialising the CNN
classifier = Sequential()
N = 3

# Step 1 - Convolution
classifier.add(Convolution2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

for i in range(1, N):

    # Adding multiple Convolution Layers
    classifier.add(Convolution2D(32, (3, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full Connection
classifier.add(Dense(activation = 'relu', units=128))
classifier.add(Dense(activation = 'sigmoid', units=1))

# Compiling the CNN binary_crossentropy -- categorical_crossentropy
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator
                                   
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                        steps_per_epoch=25,
                        validation_data=test_set,
                        validation_steps=4)
                        #epochs=25,
                        
# Part 3 - Making new predictions

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/IDRiD_143.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
#training_set.class_indices
'''if result[0][0] >= 0.995 and result[0][0] <= 0.996:
    prediction = 'Proliferative Diabetic Retinopathy (PDR)'
elif result[0][0] > 0.96 and result[0][0] <= 0.97:
    prediction = 'Severe non-proliferative Retinopathy'
elif result[0][0] > 0.998 and result[0][0] <= 0.999:
    prediction = 'Moderate non-proliferative Retinopathy'''
if result[0][0] >= 0.9:
    prediction = 'Diabetic Retinopathy Traces Observed'
elif result[0][0] > 0 and result[0][0] < 0.99:
    prediction = 'Diabetic Retinopathy Traces Not Found'