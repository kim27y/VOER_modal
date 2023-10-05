from shutil import copyfile, copy
import matplotlib.pyplot as plt
import IPython.display as ipd
import os
import pandas as pd

import numpy as np
import itertools

# Audio processing
import librosa
import librosa.display
import audiomentations

# Sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, top_k_accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

from keras.utils import np_utils

X_train = []
y_train = []

root_path = 'data/feature/wav/'

def switch(emotion):
    if emotion == 2:
        return 'calm'
    elif emotion == 3:
        return 'happy'
    elif emotion == 4:
        return 'sad'
    elif emotion == 5:
        return 'angry'
    elif emotion == 6:
        return 'fear'
    elif emotion == 7:
        return 'disgust'
    elif emotion == 8:
        return 'surprise'


for i in os.listdir(root_path):
    for j in os.listdir(os.path.join(root_path,i)):
        file_path = os.path.join(root_path, i, j)
        label = os.path.basename(file_path)[:-4].split('-')
        label = switch(int(label[2]))
        X_train.append(np.load(file_path,allow_pickle=True))
        y_train.append(label)

X_test = X_train[1920:2304]
y_test = y_train[1920:2304]

X_train = X_train[:1920]
y_train = y_train[:1920]


for i in range(len(X_train)):
    X_train[i] = X_train[i].astype(np.float64)
for i in range(len(X_test)):
    X_test[i] = X_test[i].astype(np.float64)
    
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

scaler_train = StandardScaler()

X_train = scaler_train.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
X_test = scaler_train.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

lb = LabelEncoder()
y_train = np_utils.to_categorical(lb.fit_transform(y_train))
y_test = np_utils.to_categorical(lb.fit_transform(y_test))

X_train = np.expand_dims(X_train, axis = 3)
X_test = np.expand_dims(X_test, axis = 3)

np.save('data/X_train', X_train)
np.save('data/y_train', y_train)
np.save('data/X_test', X_test)
np.save('data/y_test', y_test)