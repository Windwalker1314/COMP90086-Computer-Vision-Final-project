# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 07:11:33 2021

@author: Windwalker
"""
import os
import pandas as pd
import numpy as np
import shutil
from sklearn.model_selection import train_test_split

base_dir = "data_cells"
#base_dir = "data_locations"
train_dir = os.path.join(base_dir, 'train_dir')
val_dir = os.path.join(base_dir, 'val_dir')
try:
    os.mkdir(base_dir)
except FileExistsError:
    print("base_dir created")

try:
    os.mkdir(train_dir)
except FileExistsError:
    pass

try:
    os.mkdir(val_dir)
except FileExistsError:
    pass

df_data = pd.read_csv('train_with_cells.csv')
labels=np.unique(df_data.cells1)
#labels=np.unique(df_data['locationID'])
for l in labels:
    try:
        os.mkdir(os.path.join(train_dir,str(l)))
    except FileExistsError:
        pass
    try:    
        os.mkdir(os.path.join(val_dir,str(l)))
    except FileExistsError:
        pass

y = df_data.cells1
#y = df_data['locationID']
X = df_data.id

X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.1,random_state=2020)

origin_img_dir = 'data/train'
img_data = os.listdir(origin_img_dir)


for img_id, label in zip(X_train.values, y_train.values):
    imgName = img_id+".jpg"
    if imgName in img_data:
        src = os.path.join(origin_img_dir,imgName)
        dst = os.path.join(train_dir,str(label),imgName)
        shutil.copyfile(src, dst)

for img_id, label in zip(X_val.values, y_val.values):
    imgName = img_id+".jpg"
    if imgName in img_data:
        src = os.path.join(origin_img_dir,imgName)
        dst = os.path.join(val_dir,str(label),imgName)
        shutil.copyfile(src, dst)


