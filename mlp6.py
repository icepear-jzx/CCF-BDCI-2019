import keras
from keras import layers
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import os
from dataparser import write_results
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

train_file = 'Train/train_extra_data.csv'
eval_file = 'Forecast/evaluation_public.csv'


def preprocess_train_data():
    df = pd.read_csv(train_file)
    train_list = []
    for model in df['model'].unique():
        for adcode in df['adcode'].unique():
            index = (df['model'] == model) & (df['adcode'] == adcode)
            train_list.append(df[['salesVolume']][index].values)
    return np.array(train_list) # 1320 * 24 * 1


def build_mlp(input_dim):
    input = layers.Input(shape=(input_dim, ))
    dense = layers.Dense(32, activation='sigmoid', kernel_initializer='he_normal')(input)
    dense = layers.Dense(input_dim, kernel_initializer='he_normal')(dense)
    dense = layers.Add()([input, dense])
    dense = layers.Activation('sigmoid')(dense)
    dense = layers.Dense(32, activation='sigmoid', kernel_initializer='he_normal')(dense)
    output = layers.Dense(input_dim)(dense)
    model = keras.Model(input, output)
    model.compile(keras.optimizers.Adam(0.01), loss=keras.losses.mse)
    return model


def my_metric(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def get_score(y_true, y_pred):
    return 1 - np.sum(np.abs(y_pred-y_true)/y_true)/60


def scale_fit(x):
    mu = np.mean(x)
    sigma = np.std(x)
    return mu, sigma


def scale_to(x, mu, sigma, month_range):
    xs = (x - mu) / sigma
    return xs


def scale_back(xs, mu, sigma, month_range):
    x = xs * sigma + mu
    return x


def smooth(x):
    mean_2016 = np.mean(x[:, :12], axis=1)
    mean_2017 = np.mean(x[:, 12:], axis=1)
    # (5.5, mean_2016) (17.5, mean_2017)
    k = (mean_2017 - mean_2016) / 12
    b = mean_2016 - k * 5.5
    base = np.array([k * i + b for i in range(36)]).T
    xs = x - base[:, :24]
    return xs, base


def main():
    x = preprocess_train_data()
    x = np.reshape(x, (-1, 24))
    x, base = smooth(x)
    
    mu, sigma = scale_fit(x)
    xs_train = scale_to(x[:, :12], mu, sigma, range(0, 12))
    ys_train = scale_to(x[:, 12:], mu, sigma, range(0, 12))
    xs_test = scale_to(x[:, :12], mu, sigma, range(0, 12))
    y_true = x[:, 12:]

    # xs_train = np.vstack([xs_train[:, 0:4], xs_train[:, 4:8], xs_train[:, 8:12]])
    # ys_train = np.vstack([ys_train[:, 0:4], ys_train[:, 4:8], ys_train[:, 8:12]])

    print('The shape of input data is ', x.shape)

    for i in range(10):
        model = build_mlp(input_dim=12)
        model.summary()

        model.fit(xs_train, ys_train, batch_size=32, epochs=300, validation_split=0, verbose=2)
        ys_pred = model.predict(xs_test)
        y_pred = scale_back(ys_pred, mu, sigma, range(0, 12))
        rmse = my_metric(y_true, y_pred)
        print('rmse: %.3f'%rmse)

        # for i in range(0,1320,100):
        #     plt.plot(y_true[i], label="true", color='green')
        #     plt.plot(y_pred[i], label='predicted', color='red')
        #     # plt.plot(x[0][:12], label='origin', color='blue')
        #     plt.legend(loc='upper left')
        #     plt.show()

        xs_eval = scale_to(x[:, 12:], mu, sigma, range(0, 12))
        ys_eval = model.predict(xs_eval)
        y_eval = scale_back(ys_eval, mu, sigma, range(0, 12))
        y_eval = y_eval + base[:, 24:]
        for j in range(1320):
            top = x[j][23] + base[j][23]
            bottom = np.min(y_eval[j][:4])
            if bottom < 0:
                y_eval[j][:4] = top - (top - y_eval[j][:4]) * top / (top - bottom)
        y_result = np.reshape(y_eval[:, :4], (1320*4), order='F')
        write_results('Results/mlp6-No.{}'.format(i), y_result)


if __name__ == '__main__':
    main()
