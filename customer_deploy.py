#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 16:31:35 2022

@author: anne
"""

import os
import pickle
import numpy as np


MODEL_SAVE_PATH = os.path.join(os.getcwd(),'model','model.h5')

LABEL_ENCODER_PATH = os.path.join(os.getcwd(),'model','label_encoder.pkl')

with open(MODEL_SAVE_PATH,'rb') as file:
    model = pickle.load(file)
    
with open(LABEL_ENCODER_PATH,'rb') as file:
    le = pickle.load(file)
    
#%%
# data cleaning
# Features Selection - accepted_features

new_data = []
new_data = np.expand_dims(new_data,axis=0)

print(model.predict(new_data))

print(le.inverse_transform(model.predict(new_data)))