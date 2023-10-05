import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

from keras import regularizers
from keras.regularizers import l2
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Dense, Activation, LeakyReLU, Conv2D, MaxPooling2D, GlobalMaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils, to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report

X_train = np.load('./data/X_train2.npy')
y_train = np.load('./data/y_train2.npy')

X_train = X_train.reshape(-1,X_train.shape[2])

X_test = X_train[int(X_train.shape[0]*0.85):]
y_test = y_train[int(y_train.shape[0]*0.85):]
X_train = X_train[:int(X_train.shape[0]*0.85)]
y_train = y_train[:int(y_train.shape[0]*0.85)]
tmp = [0,0,0,0,0,0,0]
for i in range(y_train.shape[0]):
    tmp[np.argmax(y_train[i])] += 1
print(tmp)

num_samples = 400
# num_classes = 7

# model = Sequential([
#     Dense(512, activation='relu', input_shape=(400,)),
#     Dropout(0.5),
#     Dense(256, activation='relu'),
#     Dropout(0.5),
#     Dense(128, activation='relu'),
#     Dropout(0.5),
#     Dense(7, activation='softmax')
# ])

# # 모델을 컴파일합니다.
# optimizer = Adam(learning_rate=0.0005)
# model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# # 모델을 학습합니다.
# history = model.fit(X_train, y_train, validation_split=0.2, epochs=10, batch_size=32, verbose=1)

# # 모델을 평가합니다.
# score = model.evaluate(X_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])

# # 클래스별 평가 지표를 출력합니다.
# y_pred = model.predict(X_test)
# y_pred_classes = np.argmax(y_pred, axis=1)
# y_true_classes = np.argmax(y_test, axis=1)
# print(classification_report(y_true_classes, y_pred_classes))


# model = keras.Sequential([
#     layers.Input(shape=num_samples),
#     layers.Flatten(),
#     layers.Dense(1024, activation='relu'),
#     layers.Dense(512, activation='relu'),
#     layers.Dense(128, activation='relu'),
#     layers.Dense(num_classes, activation='softmax')
# ])

# # 모델 컴파일

 
# batch_size = 32
# epochs = 10


# stop_early = EarlyStopping(monitor = 'val_loss', patience = 10, restore_best_weights = True)
# model.compile(loss = 'categorical_crossentropy', optimizer = Adam(learning_rate = 0.01), metrics = ['accuracy'])
# history0 = model.fit(X_train, y_train, batch_size = batch_size, epochs = epochs, validation_data = (X_test, y_test), callbacks = [stop_early])