# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 08:06:45 2021

@author: Windwalker
"""
import tensorflow as tf
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from cellClassifier import CellClassifier_Xception
data_gen_args = dict(rescale=1./255,
                     width_shift_range=0,
                     height_shift_range=0,
                     zoom_range=0,
                     horizontal_flip=False,
                     vertical_flip=False,
                     validation_split=0.1)

seed = 1
df = pd.read_csv("train_with_cells.csv")

train_datagen = ImageDataGenerator(**data_gen_args)
val_datagen = ImageDataGenerator(**data_gen_args)
df = pd.read_csv("train_with_cells.csv")
df = df.astype({'cells1':'str'})
train_generator = train_datagen.flow_from_dataframe(dataframe=df,
                                                    directory='./data/train/',
                                                    x_col='filename',
                                                    y_col='cells1',
                                                    subset="training",
                                                    batch_size = 32,
                                                    seed=seed,
                                                    shuffle=True,
                                                    class_mode='sparse',
                                                    target_size=(224,224))
val_generator = val_datagen.flow_from_dataframe(dataframe=df,
                                                directory='./data/train/',
                                                x_col='filename',
                                                y_col='cells1',
                                                subset="validation",
                                                batch_size = 32,
                                                seed=seed,
                                                shuffle=True,
                                                class_mode='sparse',
                                                target_size=(224,224))


model = CellClassifier_Xception()

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
r=model.fit(train_generator, validation_data=val_generator,
            validation_steps=20,steps_per_epoch=211,epochs=100,callbacks=[callback])



