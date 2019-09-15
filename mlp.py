import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import pandas as pd
import numpy as np


with open('Train/train_all_data.csv', 'r') as f:
    raw_data = pd.read_csv(f)

input_raw_data = raw_data[(raw_data.regYear == 2016) | (raw_data.regMonth < 12)]
output_raw_data = raw_data[(raw_data.regYear == 2017) & (raw_data.regMonth >= 12)]

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
        X_ir = X_ir[['salesVolume']]
        X_ir = X_ir.values.reshape(-1, 23)
        Y_ir = output_raw_data[(output_raw_data.model == model) & (output_raw_data.adcode == adcode)]['salesVolume']
        Y_ir = Y_ir.values.reshape(-1, 1)
        # print('X_ir:\n', X_ir)
        # print('Y_ir:\n', Y_ir)
        X_list.append(X_ir)
        Y_list.append(Y_ir)

x_train = np.vstack(X_list)
y_train = np.vstack(Y_list)
print(x_train.shape, y_train.shape)

# 构建模型

model = keras.Sequential([
    layers.Dense(32, activation='sigmoid', kernel_initializer='he_normal', input_shape=(23,)),
    layers.Dense(64, activation='sigmoid', kernel_initializer='he_normal'),
    layers.Dense(32, activation='sigmoid', kernel_initializer='he_normal'),
    layers.Dense(1)
])

# 配置模型
model.compile(optimizer=keras.optimizers.Adam(),
             loss='mean_squared_error',
             metrics=['mse'])
model.summary()

# 训练
model.fit(x_train, y_train, batch_size=64, epochs=500, validation_split=0.1, verbose=1)

# result = model.evaluate(x_test, y_test)

# print(model.metrics_names)
# print(result)