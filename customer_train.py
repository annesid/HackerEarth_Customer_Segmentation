#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 09:19:58 2022

@author: anne
"""

import os
import pickle
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from customer_module import EDA,ModelDevelopment,ModelEvaluation

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder, LabelEncoder
from sklearn.metrics import classification_report,confusion_matrix,ConfusionMatrixDisplay

from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import TensorBoard,EarlyStopping 

#%%PATH 
CSV_PATH_TRAIN = os.path.join(os.getcwd(),'Dataset','Train.csv')
CSV_PATH_TEST = os.path.join(os.getcwd(),'Dataset','Test.csv')


LOGS_PATH = os.path.join(os.getcwd(),
                          'logs',
                          datetime.datetime.now().
                          strftime('%Y%m%d-%H%M%S'))

LABEL_ENCODER_PATH = os.path.join(os.getcwd(),'model','label_encoder.pkl')
MMS_PATH = os.path.join(os.getcwd(),'model', 'mms.pkl')
OHE_PATH = os.path.join(os.getcwd(),'model','ohe_encoder.pkl')
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'model','model.h5')

#%% Step 1) data loading

df = pd.read_csv(CSV_PATH_TRAIN)
df_test = pd.read_csv(CSV_PATH_TEST)

#%% Step 2) Data inspection

df.info()
df.isna().sum()
df.describe().T

df = df.drop(labels=['id','days_since_prev_campaign_contact'],axis=1)

cat = ['job_type','marital','education','default','housing_loan',
       'personal_loan','communication_type','month','prev_campaign_outcome',
       'term_deposit_subscribed','day_of_month']

cont = ['customer_age','balance','last_contact_duration',
        'num_contacts_in_campaign','num_contacts_prev_campaign']

eda = EDA()
eda.displot_graph(cont,df)
eda.countplot_graph(cat,df)

df.info()
df.describe().T
df.isna().sum()
df.boxplot()
df.duplicated().sum()

#correlation
plt.figure() 
_ = sns.heatmap(df[cont].corr(), annot=True)

df.groupby(['term_deposit_subscribed', 'prev_campaign_outcome']).agg({'prev_campaign_outcome':'count'}).plot(kind='bar')

#%% Step 3) data Cleaning

for i in cat:
    if i == 'term_deposit_subscribed':
        continue
    else:
        le = LabelEncoder()
        temp = df[i]
        temp[temp.notnull()] = le.fit_transform(temp[df[i].notnull()])
        df[i] = pd.to_numeric(df[i],errors='coerce')
        PICKLE_SAVE_PATH = os.path.join(os.getcwd(),'mmodel',i+'_encoder.pkl')
        with open(PICKLE_SAVE_PATH,'wb') as file:
            pickle.dump(le,file)
            
# drop NaNs - simple imputer

for i in cont:
    df[i] = df[i].fillna(df[i].median())

for i in cat:
    df[i] = df[i].fillna(df[i].mode()[0])

df.isna().sum()

#%% Step 4) Features selection
# Cont vs cat

X = df.drop(labels='term_deposit_subscribed',axis=1)
y = df['term_deposit_subscribed'].astype(int)

selected_features = []

for i in cont:
    lr = LogisticRegression()
    lr.fit(np.expand_dims(X[i],axis=-1),y)
    print(i)
    print(lr.score(np.expand_dims(X[i],axis=-1),y))
    if lr.score(np.expand_dims(X[i],axis=-1),y) > 0.5:
        selected_features.append(i)

print(selected_features)

#Cat vs cat

for i in cat:
    print(i)
    matrix = pd.crosstab(df[i],y).to_numpy()
    print(eda.cramers_corrected_stat(matrix))
    if eda.cramers_corrected_stat(matrix) > 0.3:
        selected_features.append(i)
        
print(selected_features)


#%% Step 5) model preprocessing

df = df.loc[:,selected_features]
# X = df.drop(labels='term_deposit_subscribed',axis=1)
# y = df['term_deposit_subscribed'].astype(int)

# model evaluation

# MMS
mms = MinMaxScaler()
X = mms.fit_transform(X)
with open(MMS_PATH,'wb') as file:
    pickle.dump(mms,file)

#OHE
ohe = OneHotEncoder(sparse=False)
y = ohe.fit_transform(np.expand_dims(y,axis=-1))
with open(OHE_PATH,'wb') as file:
    pickle.dump(ohe,file)
    
#Train test Split
X_train,X_test,y_train,y_test = train_test_split(X,y,
                                                 test_size=0.3,
                                                 random_state=123)

#%%

input_shape = np.shape(X_train)[1:]
nb_class = len(np.unique(y,axis=0))

md = ModelDevelopment()
model = md.simple_dl_model(input_shape,nb_class,nb_node=128,dropout_rate=0.3)

plot_model(model,show_shapes=False,show_layer_names=True)

#%%
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['acc'])

tensorboard_callback = TensorBoard(log_dir=LOGS_PATH,histogram_freq=1) #improving model line

early_callback = EarlyStopping(monitor = 'val_acc',patience=5)

hist = model.fit(X_train,y_train,
                      epochs=100,
                      callbacks=[tensorboard_callback,early_callback],
                      validation_data=(X_test,y_test))

#%% model evaluation

print(hist.history.keys())

me = ModelEvaluation()
me.plot_hist_graph(hist)

#%%
y_pred = np.argmax(model.predict(X_test),axis=1)
y_test = np.argmax(y_test,axis=1)

print(classification_report(y_test,y_pred))

#%% Confusion Matrix

cm=confusion_matrix(y_test,y_pred)

disp=ConfusionMatrixDisplay(confusion_matrix=cm)#,display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.show()

#%% model saving

model.save(MODEL_SAVE_PATH)


