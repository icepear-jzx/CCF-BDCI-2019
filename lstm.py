import keras
from keras import layers
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '7'

train_file = 'Train/train_extra_data.csv'
eval_file = 'Forecast/evaluation_public.csv'


def preprocess_train_data():
    df = pd.read_csv(train_file)
    train_list = []
    for model in df['model'].unique():
        for adcode in df['adcode'].unique():
            index = (df['model'] == model) & (df['adcode'] == adcode)
            train_list.append(df[['salesVolume', 'popularity']][index].values)
    return np.array(train_list)
    

def scale(x):
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    return (x - mu)/sigma, mu, sigma


def re_scale(x, mu, sigma):
    return x*sigma[0] + mu[0]


def my_metric(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def get_score(y_true, y_pred):
    return 1 - np.sum(np.abs(y_pred-y_true)/y_true)/60


def build_lstm():
    lstm = keras.Sequential()
    lstm.add(layers.LSTM(32, input_dim=2, return_sequences=False))
    # lstm.add(layers.LSTM(32))
    lstm.add(layers.Dense(32))
    lstm.add(layers.Dense(1))
    lstm.compile(keras.optimizers.Adam(1e-2), loss=keras.losses.mse)
    return lstm


def build_mlp():
    mlp = keras.Sequential()
    mlp.add(layers.Reshape([1*22], input_shape=[22]))
    mlp.add(layers.Dense(32, activation='sigmoid'))
    mlp.add(layers.Dense(32, activation='sigmoid'))
    mlp.add(layers.Dense(32, activation='sigmoid'))
    mlp.add(layers.Dense(1))
    mlp.compile(keras.optimizers.Adam(1e-2), loss=keras.losses.mse)
    return mlp


def main():
    x = preprocess_train_data()
    print('The shape of input data is ', x.shape)
    xs, mu, sigma = scale(x)
    model = build_lstm()
    model.summary()

    # model.fit(xs[:, :-2, :1], xs[:, -2, 0], batch_size=100, epochs=200, validation_split=0.1, verbose=2)
    for t in range(1, xs.shape[1]-1):
        model.fit(x[:, :t, :], x[:, t, 0], batch_size=32, epochs=t, validation_split=0.1, verbose=2)
    x_pred = model.predict(x[:, 1:-1, :])
    # x_pred = re_scale(x_pred, mu, sigma)
    print('rmse: %.3f'%my_metric(x[:, -1, 0], x_pred))
    print('score: %.3f'%get_score(x[:, -1, 0], x_pred))


if __name__ == '__main__':
    main()
