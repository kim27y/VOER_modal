# Utility
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

# Keras
import keras
from keras import regularizers
from keras.regularizers import l2
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Dense, Activation, LeakyReLU, Conv2D, MaxPooling2D, GlobalMaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils, to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint

X_train = np.load('./data/X_train_basic.npy',allow_pickle=True).astype(float)
y_train = np.load('./data/y_train_basic.npy',allow_pickle=True).astype(float)
X_test = np.load('./data/X_test_basic.npy',allow_pickle=True).astype(float)
y_test = np.load('./data/y_test_basic.npy',allow_pickle=True).astype(float)

num_classes = 7

model0 = Sequential()

model0.add(Conv2D(32, (3, 3), activation = 'relu', input_shape = (128, 282, 1)))
model0.add(MaxPooling2D((2, 2)))

model0.add(Conv2D(64, (3, 3), activation = 'relu'))
model0.add(MaxPooling2D((2, 2)))

model0.add(Conv2D(64, (3, 3), activation = 'relu'))
model0.add(GlobalMaxPooling2D())

model0.add(Dense(64, activation = 'relu'))

model0.add(Dense(num_classes, activation = 'softmax'))

stop_early = EarlyStopping(monitor = 'val_loss', patience = 10, restore_best_weights = True)
model0.compile(loss = 'categorical_crossentropy', optimizer = Adam(learning_rate = 0.001), metrics = ['accuracy'])
history0 = model0.fit(X_train, y_train, batch_size = 32, epochs = 100, validation_data = (X_test, y_test), callbacks = [stop_early])
