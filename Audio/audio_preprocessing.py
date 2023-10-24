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
save_list = './data/features'
emotion = []
path = []

for i in dir_list:
    # Remove wav extension
    id = i[:-4].split('-')
    if(id[2] != '01'):
      # Dividing according to emotions
      emotion.append(switch(int(id[2])))
      path.append('./data/wav/' + i)
      
# Make DataFrame value to save data's path and label
df = pd.concat([pd.DataFrame(emotion), pd.DataFrame(path)], axis = 1)
df.columns = ['emotion', 'path']
audio = []
for filename in df['path']:
  data, sampling_rate = librosa.load(filename, sr = 48000, duration = 3, offset = 0.5) # We want the native sr
  audio.append(data)
# df = pd.DataFrame(np.column_stack([df, audio]))
df['data'] = audio

data_len = len(df)
train_len = int(len(df)*0.7)

print("dataframe 생성 완료")

for i in range(data_len):
  if(len(df['data'][i]) != 144000):
    start_pad = (144000 - len(df['data'][i]))//2
    end_pad = 144000 - len(df['data'][i]) - start_pad
    df['data'][i] = np.pad(df['data'][i], (start_pad, end_pad), mode = 'constant')

df['features'] = [0] * data_len
for i in range(data_len):
  mel = feature_extractor(df['data'][i])
  df['features'][i] = np.array(mel, dtype = object)

X_test = df['features'][train_len:data_len].tolist()
y_test = df['emotion'][train_len:data_len].tolist()
X_train = df['features'][:train_len].tolist()
y_train = df['emotion'][:train_len].tolist()

"데이터셋 생성 완료"

for i in range(train_len):
  augmented_samples = transform(df['data'][i], 48000)
  mel = feature_extractor(augmented_samples)
  X_train.append(np.array(mel, dtype = object))
  y_train.append(df['emotion'][i])

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

scaler_train = StandardScaler()

X_train[:train_len] = scaler_train.fit_transform(X_train[:train_len].reshape(-1, X_train.shape[-1])).reshape(X_train[:train_len].shape)
X_train[train_len:] = scaler_train.transform(X_train[train_len:].reshape(-1, X_train.shape[-1])).reshape(X_train[train_len:].shape)
X_test = scaler_train.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

lb = LabelEncoder()
y_train = np_utils.to_categorical(lb.fit_transform(y_train))
y_test = np_utils.to_categorical(lb.fit_transform(y_test))

X_train = np.expand_dims(X_train, axis = 3)
X_test = np.expand_dims(X_test, axis = 3)

"완료"

np.save('./data/X_train.npy', X_train)
np.save('./data/X_test.npy', X_test)
np.save('./data/y_train.npy', y_train)
np.save('./data/y_test.npy', y_test)

"저장 완료"