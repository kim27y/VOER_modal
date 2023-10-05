import matplotlib.pyplot as plt
import IPython.display as ipd
import os
import pandas as pd

import numpy as np

# Audio processing
import librosa
import librosa.display
import audiomentations
import moviepy.editor as mp

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
 


dir_list = os.listdir('./data/features/')

data = []
label = []
for i in dir_list:
  filename = os.listdir('./data/features/' + i)
  for f in filename:
    # Remove wav extension
    id = f[:-4].split('_')[0].split('-')
    if(id[2] != '01'):
      # Dividing according to emotions
      data.append(np.load(os.path.join('./data/features/' + i, f)))
      label.append(switch(int(id[2])))


X_train = np.array(data)
y_train = np.array(label)

lb = LabelEncoder()
y_train = np_utils.to_categorical(lb.fit_transform(y_train))

np.save('./data/X_train.npy', X_train)
np.save('./data/y_train.npy', y_train)

print(y_train)
print(X_train.shape)
print(y_train.shape)


