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
    raw_data['regDate'] = raw_data['regYear'] * 100 + raw_data['regMonth']

X_train_list = []
Y_train_list = []
X_test_list = []
Y_test_list = []
label_test_list = []

for m in range(1, 10):
    regDate_start = 201600 + m
    regDate_train_end = 201600 + 100 * ((m + 6) // 12) + (m + 6) % 12 + 1
    regDate_test_end = 201600 + 100 * ((m + 10) // 12) + (m + 10) % 12 + 1
    # regDate_train_end = 201600 + 100 * ((m + 10) // 12) + (m + 10) % 12 + 1
    # regDate_test_end = 201600 + 100 * ((m + 14) // 12) + (m + 14) % 12 + 1
    input_raw_data = raw_data[(raw_data.regDate >= regDate_start) & (raw_data.regDate <= regDate_train_end)]
    output_raw_data = raw_data[(raw_data.regDate > regDate_train_end) & (raw_data.regDate <= regDate_test_end)]

    for model in set(input_raw_data.model):
        X_i = input_raw_data[input_raw_data.model == model]
        # bodyType = X_i.iloc[0].bodyType
        for adcode in set(X_i.adcode):
            # print('model:', model)
            # print('bodyType:', bodyType)
            # print('adcode:', adcode)
            X_ir = X_i[X_i.adcode == adcode]
            X_ir = X_ir[['salesVolume', 'popularity', 'carCommentVolum', 'newsReplyVolum']]
            X_ir = X_ir.values.reshape(-1, 8 * 4)
            Y_ir = output_raw_data[(output_raw_data.model == model) & (output_raw_data.adcode == adcode)]['salesVolume']
            Y_ir = Y_ir.values.reshape(-1, 4)
            # print('X_ir:\n', X_ir)
            # print('Y_ir:\n', Y_ir)
            X_train_list.append(X_ir)
            Y_train_list.append(Y_ir)

regDate_start = 201701
# regDate_train_end = 201712
regDate_train_end = 201708
regDate_test_end = 201712
input_raw_data = raw_data[(raw_data.regDate >= regDate_start) & (raw_data.regDate <= regDate_train_end)]
output_raw_data = raw_data[(raw_data.regDate > regDate_train_end) & (raw_data.regDate <= regDate_test_end)]

for model in set(input_raw_data.model):
    X_i = input_raw_data[input_raw_data.model == model]
    # bodyType = X_i.iloc[0].bodyType
    for adcode in set(X_i.adcode):
        # print('model:', model)
        # print('bodyType:', bodyType)
        # print('adcode:', adcode)
        X_ir = X_i[X_i.adcode == adcode]
        X_ir = X_ir[['salesVolume', 'popularity', 'carCommentVolum', 'newsReplyVolum']]
        X_ir = X_ir.values.reshape(-1, 8 * 4)
        Y_ir = output_raw_data[(output_raw_data.model == model) & (output_raw_data.adcode == adcode)]['salesVolume']
        Y_ir = Y_ir.values.reshape(-1, 4)
        # print('X_ir:\n', X_ir)
        # print('Y_ir:\n', Y_ir)
        X_test_list.append(X_ir)
        Y_test_list.append(Y_ir)
        label_test_list.append((model, adcode))

x_all = np.vstack(X_train_list + X_test_list)
x_all, _, _ = standardization(x_all)
x_train = x_all[:len(X_train_list)]
x_test = x_all[len(X_train_list):]

y_train = np.vstack(Y_train_list)
y_train, mu, sigma = standardization(y_train)
y_test = np.vstack(Y_test_list)
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
model.fit(x_train, y_train, batch_size=256, epochs=100)

result = model.predict(x_test)
result = result * sigma + mu
# y_test = y_test * sigma + mu

# print(model.metrics_names)
# print(y_test)
print(y_test.mean())
# print(result)
print(((y_test - result) ** 2).mean() ** 0.5)
# input()

# with open('Forecast/evaluation_public.csv', 'r') as f:
#     forecast_data = pd.read_csv(f)

# for i in range(len(label_test_list)):
#     model, adcode = label_test_list[i]
#     salesVolume = result[i]
#     forecast_data.forecastVolum[(forecast_data.model == model) & (forecast_data.adcode == adcode) & (forecast_data.regMonth == 1)] = int(salesVolume[0])
#     forecast_data.forecastVolum[(forecast_data.model == model) & (forecast_data.adcode == adcode) & (forecast_data.regMonth == 2)] = int(salesVolume[1])
#     forecast_data.forecastVolum[(forecast_data.model == model) & (forecast_data.adcode == adcode) & (forecast_data.regMonth == 3)] = int(salesVolume[2])
#     forecast_data.forecastVolum[(forecast_data.model == model) & (forecast_data.adcode == adcode) & (forecast_data.regMonth == 4)] = int(salesVolume[3])

# del forecast_data['province']
# del forecast_data['adcode']
# del forecast_data['model']
# del forecast_data['regYear']
# del forecast_data['regMonth']

# forecast_data.to_csv('Example/0915.csv', index=False)