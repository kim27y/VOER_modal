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

# Switch label function
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

# mel spectrogram
def feature_extractor(input, sr = 48000):
    return librosa.power_to_db(librosa.feature.melspectrogram(y=input*1.0, sr = sr, n_fft = 1024, n_mels = 128, fmin = 50, fmax = 24000)) 


transform = audiomentations.Compose([
    audiomentations.AddGaussianNoise(p = 1),
    audiomentations.PitchShift(p = 1),
    audiomentations.TimeStretch(p = 1),
    audiomentations.Shift(min_shift = 0.25, max_shift = 0.25, rollover = False, p = 1)
])

# Video dataset
dir_list = os.listdir('./data/wav/')

emotion = []
path = []

for i in dir_list:
  filename = os.listdir('./data/wav/' + i)
  for f in filename:
    # Remove wav extension
    id = f[:-4].split('-')
    if(id[2] != '01'):
      # Dividing according to emotions
      emotion.append(switch(int(id[2])))
      path.append('./data/wav/' + i + '/' + f)
      
# Make DataFrame value to save data's path and label
df = pd.concat([pd.DataFrame(emotion), pd.DataFrame(path)], axis = 1)
df.columns = ['emotion', 'path']

audio = []
for filename in df['path']:
  data, sampling_rate = librosa.load(filename, sr = 48000, duration = 3, offset = 0.5) # We want the native sr
  audio.append(data)
df = pd.DataFrame(np.column_stack([df, audio]))
df.columns = ['emotion', 'path', 'data']

for i in range(len(df)):
  if(len(df['data'][i]) != 144000):
    start_pad = (144000 - len(df['data'][i]))//2
    end_pad = 144000 - len(df['data'][i]) - start_pad
    df['data'][i] = np.pad(df['data'][i], (start_pad, end_pad), mode = 'constant')

df['features'] = [0] * 2304
for i in range(len(df)):
  mel = feature_extractor(df['data'][i], 'mel')
  df['features'][i] = np.array(mel, dtype = object)

X_test = df['features'][1920:2304].tolist()
y_test = df['emotion'][1920:2304].tolist()
X_train = df['features'][:1920].tolist()
y_train = df['emotion'][:1920].tolist()

for i in range(1920):
  augmented_samples = transform(df['data'][i], 48000)
  mel = feature_extractor(augmented_samples, 'mel')
  X_train.append(np.array(mel, dtype = object))
  y_train.append(df['emotion'][i])

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

scaler_train = StandardScaler()

X_train[:1920] = scaler_train.fit_transform(X_train[:1920].reshape(-1, X_train.shape[-1])).reshape(X_train[:1920].shape)
X_train[1920:] = scaler_train.transform(X_train[1920:].reshape(-1, X_train.shape[-1])).reshape(X_train[1920:].shape)
X_test = scaler_train.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

lb = LabelEncoder()
y_train = np_utils.to_categorical(lb.fit_transform(y_train))
y_test = np_utils.to_categorical(lb.fit_transform(y_test))

X_train = np.expand_dims(X_train, axis = 3)
X_test = np.expand_dims(X_test, axis = 3)


# X_train = np.load('./data/X_train.npy')
# X_test = np.load('./data/X_test.npy')
X_train_vanilla = X_train[:896]
y_train_vanilla = y_train[:896]
X_val = X_train[896:1120]
y_val = y_train[896:1120]

print(X_train_vanilla.shape, y_train_vanilla.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)

np.save('./data/X_train_basic.npy', X_train)
np.save('./data/X_test_basic.npy', X_test)
np.save('./data/y_train_basic.npy', y_train)
np.save('./data/y_test_basic.npy', y_test)