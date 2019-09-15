import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


with open('Train/train_all_data.csv', 'r') as f:
    raw_data = pd.read_csv(f)

input_raw_data = raw_data[(raw_data.regYear == 2016) | (raw_data.regMonth < 9)]
output_raw_data = raw_data[(raw_data.regYear == 2017) & (raw_data.regMonth >= 9)]

X_list = []
Y_list = []

for model in set(input_raw_data.model):
    X_i = input_raw_data[input_raw_data.model == model]
    bodyType = X_i.iloc[0].bodyType
    for adcode in set(X_i.adcode):
        # print('model:', model)
        # print('bodyType:', bodyType)
        # print('adcode:', adcode)
        X_ir = X_i[X_i.adcode == adcode]
        X_ir = X_ir[['regYear', 'regMonth', 'salesVolume', 'popularity', 'carCommentVolum', 'newsReplyVolum']]
        X_ir = X_ir.values.reshape(-1, 20, 6, 1)
        Y_ir = output_raw_data[(output_raw_data.model == model) & (output_raw_data.adcode == adcode)]['salesVolume']
        Y_ir = Y_ir.values.reshape(-1, 4)
        # print('X_ir:\n', X_ir)
        # print('Y_ir:\n', Y_ir)
        X_list.append(X_ir)
        Y_list.append(Y_ir)

x_train = np.vstack(X_list)
y_train = np.vstack(Y_list)
print(x_train.shape, y_train.shape)

model = keras.Sequential()

model.add(layers.Conv2D(input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3]),
    filters=32, kernel_size=(3,3), strides=(1,1), padding='valid',
    activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(32, use_bias=True, activation='relu'))
model.add(layers.Dense(4, use_bias=True, activation='softmax'))

model.compile(optimizer=keras.optimizers.SGD(0.1),
    loss=keras.losses.MSE,
    metrics=['mse'])
model.summary()

history = model.fit(x_train, y_train, batch_size=64, epochs=5, validation_split=0.1)

res = model.evaluate(x_train, y_train)
