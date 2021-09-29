# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 08:06:45 2021

@author: Windwalker
"""
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from cellclassifier import CellClassifier
data_gen_args = dict(rescale=1./255,
                     width_shift_range=0,
                     height_shift_range=0,
                     zoom_range=0,
                     horizontal_flip=False,
                     vertical_flip=False)

seed = 1
train_dir = './data_cells/train_dir'
val_dir = './data_cells/val_dir'

train_datagen = ImageDataGenerator(**data_gen_args)
val_datagen = ImageDataGenerator(**data_gen_args)
train_generator = train_datagen.flow_from_directory(
                        train_dir,
                        color_mode="rgb",
                        seed=seed,
                        target_size=(224,224),
                        class_mode="sparse",
                        batch_size=32
                    )
val_generator = val_datagen.flow_from_directory(
                        val_dir,
                        color_mode="rgb",
                        seed=seed,
                        target_size=(224,224),
                        class_mode="sparse",
                        batch_size=32
                    )


model = CellClassifier()

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
r=model.fit(train_generator, validation_data=val_generator,
            validation_steps=20,steps_per_epoch=211,epochs=100,callbacks=[callback])



