import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import pandas as pd
import numpy as np


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma, mu, sigma

with open('Train/train_all_data.csv', 'r') as f:
    raw_data = pd.read_csv(f)

input_raw_data = raw_data[(raw_data.regYear == 2016) | (raw_data.regMonth < 9)]
output_raw_data = raw_data[(raw_data.regYear == 2017) & (raw_data.regMonth >= 9)]

X_train_list = []
Y_train_list = []
X_test_list = []
Y_test_list = []

for model in set(input_raw_data.model):
    X_i = input_raw_data[input_raw_data.model == model]
    bodyType = X_i.iloc[0].bodyType
    for adcode in set(X_i.adcode):
        # print('model:', model)
        # print('bodyType:', bodyType)
        # print('adcode:', adcode)
        X_ir = X_i[X_i.adcode == adcode]
        X_ir = X_ir[['salesVolume', 'popularity', 'carCommentVolum', 'newsReplyVolum']]
        X_ir = X_ir.values.reshape(-1, 20 * 4)
        Y_ir = output_raw_data[(output_raw_data.model == model) & (output_raw_data.adcode == adcode)]['salesVolume']
        Y_ir = Y_ir.values.reshape(-1, 4)
        # print('X_ir:\n', X_ir)
        # print('Y_ir:\n', Y_ir)
        if np.random.rand(1)[0] < 0.1:
            X_test_list.append(X_ir)
            Y_test_list.append(Y_ir)
        else:
            X_train_list.append(X_ir)
            Y_train_list.append(Y_ir)

x_all = np.vstack(X_train_list + X_test_list)
x_all, _, _ = standardization(x_all)
x_train = x_all[:len(X_train_list)]
x_test = x_all[len(X_train_list):]

y_all = np.vstack(Y_train_list + Y_test_list)
y_all, mu, sigma = standardization(y_all)
y_train = y_all[:len(Y_train_list)]
y_test = y_all[len(Y_train_list):]
print(x_train.shape, y_train.shape)

# 构建模型

model = keras.Sequential([
    layers.Dense(32, activation='sigmoid', kernel_initializer='he_normal', input_shape=(x_train.shape[1], )),
    layers.Dense(32, activation='sigmoid', kernel_initializer='he_normal'),
    layers.Dense(4)
])

# 配置模型
model.compile(optimizer=keras.optimizers.Adam(0.001),
             loss='mean_squared_error',
             metrics=['mse'])
model.summary()

# 训练
model.fit(x_train, y_train, batch_size=256, epochs=500)

result = model.predict(x_test)
# result = result * sigma + mu
# y_test = y_test * sigma + mu

# print(model.metrics_names)
# print(y_test)
# print(result)
print(((y_test - result) ** 2).mean() ** 0.5)
