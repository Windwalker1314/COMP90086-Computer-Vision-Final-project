# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 07:33:51 2021

@author: Windwalker
"""
import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D,Flatten, Dense, Dropout
from tensorflow.keras.applications.vgg16 import VGG16

def CellClassifier():
    vgg_model = VGG16(input_shape=(224,224,3),include_top=False,weights='imagenet')
    x = Flatten()(vgg_model.output)
    x=Dense(1024,activation='relu')(x)
    x=Dense(120, activation='softmax')(x)
    model = Model(inputs=vgg_model.input, outputs=x)
    model.compile(optimizer='Adam',loss=SparseCategoricalCrossentropy(),metrics=["accuracy"])
    return model

def CellClassifier_v1(input_size = (224,224,3), classes = 120):
    # VGG with fewer parameters
    img_input = Input(shape=input_size)
    x = Conv2D(32, (5, 5), activation='relu', padding='valid', name='block1_conv1')(img_input)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((3, 3), strides=(3, 3), name='block1_pool')(x)
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2pool')(x)
    
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    
    x = Flatten()(x)
    x = Dense(512,activation='relu')(x)
    x = Dense(classes, activation ='softmax')(x)
    model = Model(img_input,x)
    model.compile(optimizer='Adam',loss=SparseCategoricalCrossentropy(),metrics=["accuracy"])
    return model
