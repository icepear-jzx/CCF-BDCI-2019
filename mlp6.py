import keras
from keras import layers
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import os
from dataparser import write_results
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


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


def line_func(x, k, b):
    return k * x + b


def exp_func(x, lam, a, b):
    return a * lam * np.exp(-lam * (x + b))


def smooth(x):
    tend_x = np.arange(5.5, 18.5, 1)
    base_x = np.arange(0, 36, 1)

    tend_y = []
    for i in range(13):
        tend_y.append(np.mean(x[:, i:12+i], axis=1))
    tend_y = np.array(tend_y).T

    base = np.zeros(shape=(1320, 36))
    base_revise = np.zeros(shape=(1320, 36))
    for i in range(0,1320):
        # para, _ = curve_fit(line_func, tend_x, tend_y[i], p0=[1, 100])
        # k = para[0]
        # b = para[1]
        k = (tend_y[i][12] - tend_y[i][0]) / 12
        b = tend_y[i][0] - k * 5.5
        print(k,b)
        base_y = line_func(base_x, k, b)
        base[i] = base_y[:]
        if k < 0:
            para, _ = curve_fit(exp_func, tend_x, tend_y[i], p0=[1, 10000, 0], maxfev = 1000000)
            lam = para[0]
            a = para[1]
            b = para[2]
            base_y = exp_func(base_x, lam, a, b)
        base_revise[i] = base_y[:]
        
        print('fit:', i)
        
        # plt.plot(range(24), x[i])
        # plt.plot(tend_x, tend_y[i])
        # plt.plot(base_x, base_y)
        # plt.show()
    
    xs = x - base[:, :24]
    return xs, base, base_revise


def main():
    x = preprocess_train_data()
    x = np.reshape(x, (-1, 24))
    x, base, base_revise = smooth(x)
    
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
        # x = x + base[:, :24]
        y_eval = y_eval + base[:, 24:]

        # deal with long tail
        for j in range(1320):
            print('zoom:', j)
            # top = base[j][12] # > bottom bottom_revise
            top = x[j][23] + base[j][23]
            bottom = np.min(y_eval[j][:4])
            bottom_revise = base_revise[j][24:28] # >= 0
            print(y_eval[j][:4])
            if bottom < 0:
                # y_eval[j][:4] = top - (top - y_eval[j][:4]) * top / (top - bottom)
                if (base_revise[j][24:28] == base[j][24:28]).all(): # k > 0
                    y_eval[j][:4] = top - (top - y_eval[j][:4]) * (top - 1) / (top - bottom)
                else: # k < 0
                    y_eval[j][:4] = top - (top - y_eval[j][:4]) * (top - bottom_revise) / (top - bottom)

            # if y_eval[j][:4].min() < 0:
            #     print(top)
            #     print(y_eval[j][:4])
            #     print(base_revise[j][24:28])
            #     print(base[j][24:28])
            #     input()

        if y_eval[:, :4].min() < 0:
            print('Have negative number!')

        y_result = np.reshape(y_eval[:, :4], (1320*4), order='F')
        write_results('Results/mlp6-No.{}'.format(i), y_result)


if __name__ == '__main__':
    main()
