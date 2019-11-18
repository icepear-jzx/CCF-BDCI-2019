import keras
from keras import layers
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
    salesVolume_list = []
    popularity_list = []
    comment_list = []
    reply_list = []
    car_feature_list = []
    adcode_feature_list = []

    models = df['model'].unique()
    adcodes = df['adcode'].unique()
    bodyTypes = df['bodyType'].unique()
    model_list = list(set(df['model']))
    adcode_list = list(set(df['adcode']))
    bodyType_list = list(set(df['bodyType']))
    model_bodyType_df = df[['model','bodyType']].groupby(['model'], as_index=False).first()
    model_bodyType_map = {}
    for i in range(len(models)):
        model = model_bodyType_df.iat[i, 0]
        bodyType = model_bodyType_df.iat[i, 1]
        model_bodyType_map[model] = bodyType

    model_onehot_map = {model: [0] * 60 for model in models}
    bodyType_onehot_map = {bodyType: [0] * 4 for bodyType in bodyTypes}
    adcode_onehot_map = {adcode: [0] * 22 for adcode in adcodes}
    for model in models:
        model_onehot_map[model][model_list.index(model)] = 1
    for bodyType in bodyTypes:
        bodyType_onehot_map[bodyType][bodyType_list.index(bodyType)] = 1
    for adcode in adcodes:
        adcode_onehot_map[adcode][adcode_list.index(adcode)] = 1

    for model in models:
        for adcode in adcodes:
            mask = (df['model'] == model) & (df['adcode'] == adcode)
            df_mask = df[mask]
            salesVolume_list.append(df_mask['salesVolume'].values)
            popularity_list.append(df_mask['popularity'].values)
            comment_list.append(df_mask['carCommentVolum'].values)
            reply_list.append(df_mask['newsReplyVolum'].values)

            model_onehot = model_onehot_map[model]
            bodyType_onehot = bodyType_onehot_map[model_bodyType_map[model]]
            car_feature = np.hstack((model_onehot, bodyType_onehot))
            car_feature_list.append(car_feature)

            adcode_onehot = adcode_onehot_map[adcode]
            adcode_feature_list.append(adcode_onehot)
    
    salesVolume_list = np.reshape(np.array(salesVolume_list), (-1, 24)) # 1320 * 24
    popularity_list = np.reshape(np.array(popularity_list), (-1, 24)) # 1320 *24
    comment_list = np.reshape(np.array(comment_list), (-1, 24)) # 1320 * 24
    reply_list = np.reshape(np.array(reply_list), (-1, 24)) # 1320 * 24
    car_feature_list = np.reshape(np.array(car_feature_list), (-1, 64)) # 1320 * (60 + 4)
    # car_feature_list = np.reshape(np.array(car_feature_list), (-1, 4))
    adcode_feature_list = np.reshape(np.array(adcode_feature_list), (-1, 22)) # 1320 * 22

    return salesVolume_list, popularity_list, comment_list, reply_list, car_feature_list, adcode_feature_list


def build_mlp():
    sales = layers.Input(shape=(12, ))
    popularity = layers.Input(shape=(12, ))
    comment = layers.Input(shape=(12, ))
    reply = layers.Input(shape=(12, ))
    car_onehot = layers.Input(shape=(64, ))
    # car_onehot = layers.Input(shape=(4, ))
    adcode_onehot = layers.Input(shape=(22, ))

    # car_embed = layers.Dense(12, activation='sigmoid', kernel_initializer='he_normal')(car_onehot)
    # car_embed = layers.Dense(12, activation='sigmoid', kernel_initializer='he_normal')(car_embed)
    # car_embed = layers.concatenate([sales, car_embed])
    # car_embed = layers.Dense(24, activation='sigmoid', kernel_initializer='he_normal')(car_embed)

    # adcode_embed = layers.Dense(12, activation='sigmoid', kernel_initializer='he_normal')(adcode_onehot)
    # adcode_embed = layers.Dense(12, activation='sigmoid', kernel_initializer='he_normal')(adcode_embed)

    # extra = layers.concatenate([popularity, comment, reply])
    # extra_embed = layers.Dense(12, activation='sigmoid', kernel_initializer='he_normal')(extra)
    # extra_embed = layers.Dense(12, activation='sigmoid', kernel_initializer='he_normal')(extra_embed)
    
    feature = layers.concatenate([sales, popularity])
    feature = layers.Dense(24, activation='sigmoid', kernel_initializer='he_normal')(feature)
    feature = layers.Dense(12, activation='sigmoid', kernel_initializer='he_normal')(feature)

    dense = layers.Add()([sales, feature])
    dense = layers.Dense(24, activation='sigmoid', kernel_initializer='he_normal')(dense)
    # dense = layers.concatenate([car_embed, adcode_embed])
    # dense = layers.Dense(24, activation='sigmoid', kernel_initializer='he_normal')(dense)
    # dense = layers.Add()([sales, dense])
    # dense = layers.Dense(12, activation='sigmoid', kernel_initializer='he_normal')(dense)
    output = layers.Dense(12)(dense)

    model = keras.Model([sales, popularity, comment, reply, car_onehot, adcode_onehot], output)
    # model = keras.Model([sales, car_onehot, adcode_onehot], output)
    model.compile(keras.optimizers.Adam(1e-2), loss=keras.losses.mse)

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
    salesVolume_list, popularity_list, comment_list, reply_list, \
        car_feature_list, adcode_feature_list = preprocess_train_data()

    # salesVolume_list, base = smooth(salesVolume_list)
    s_mu, s_sigma = scale_fit(salesVolume_list)
    s_train = scale_to(salesVolume_list[:, :12], s_mu, s_sigma, range(12))
    y_train = scale_to(salesVolume_list[:, 12:], s_mu, s_sigma, range(12))

    p_mu, p_sigma = scale_fit(popularity_list)
    p_train = scale_to(popularity_list[:, :12], p_mu, p_sigma, range(12))

    c_mu, c_sigma = scale_fit(comment_list)
    c_train = scale_to(comment_list[:, :12], c_mu, c_sigma, range(12))

    r_mu, r_sigma = scale_fit(reply_list)
    r_train = scale_to(reply_list[:, :12], r_mu, r_sigma, range(12))

    car_onehot = car_feature_list[:, :]
    adcode_onehot = adcode_feature_list[:, :]

    for i in range(0, 1320, 100):
        plt.plot(range(1, 25), scale_to(salesVolume_list, s_mu, s_sigma, range(12))[i])
        plt.plot(range(1, 25), scale_to(reply_list, p_mu, p_sigma, range(12))[i])
        plt.show()

    # for i in range(1, 6):
    #     s_train = np.vstack((s_train, scale_to(salesVolume_list[:, i:12+i], s_mu, s_sigma, range(i,12+i))))
    #     y_train = np.vstack((y_train, scale_to(salesVolume_list[:, 12+i:18+i], s_mu, s_sigma, range(i,6+i))))
    #     p_train = np.vstack((p_train, scale_to(popularity_list[:, i:12+i], p_mu, p_sigma, range(i,12+i))))
    #     c_train = np.vstack((c_train, scale_to(comment_list[:, i:12+i], c_mu, c_sigma, range(i,12+i))))
    #     r_train = np.vstack((r_train, scale_to(reply_list[:, i:12+i], r_mu, r_sigma, range(i,12+i))))
    #     car_onehot = np.vstack((car_onehot, car_feature_list[:, :]))
    #     adcode_onehot = np.vstack((adcode_onehot, adcode_feature_list[:, :]))

    s_test = scale_to(salesVolume_list[:, :12], s_mu, s_sigma, range(12))
    p_test = scale_to(popularity_list[:, :12], p_mu, p_sigma, range(12))
    c_test = scale_to(comment_list[:, :12], c_mu, c_sigma, range(12))
    r_test = scale_to(reply_list[:, :12], r_mu, r_sigma, range(12))
    y_true = salesVolume_list[:, 12:]

    for i in range(10):
        model = build_mlp()
        model.summary()

        model.fit([s_train, p_train, c_train, r_train, car_onehot, adcode_onehot],
            y_train, batch_size=32, epochs=300, validation_split=0, verbose=2)
        ys_pred = model.predict([s_test, p_test, c_test, r_test, car_onehot, adcode_onehot])
        y_pred = scale_back(ys_pred, s_mu, s_sigma, range(12))
        rmse = my_metric(y_true, y_pred)
        print('rmse: %.3f'%rmse)

        # for i in range(10):
        #     plt.plot(y_true[i], label="true", color='green')
        #     plt.plot(y_pred[i], label='predicted', color='red')
        #     # plt.plot(x[0][:12], label='origin', color='blue')
        #     plt.legend(loc='upper left')
        #     plt.show()

        s_eval = scale_to(salesVolume_list[:, 12:], s_mu, s_sigma, range(12))
        p_eval = scale_to(popularity_list[:, 12:], p_mu, p_sigma, range(12))
        c_eval = scale_to(comment_list[:, 12:], c_mu, c_sigma, range(12))
        r_eval = scale_to(reply_list[:, 12:], r_mu, r_sigma, range(12))
        ys_eval = model.predict([s_eval, p_eval, c_eval, r_eval, car_onehot, adcode_onehot])
        y_eval = scale_back(ys_eval, s_mu, s_sigma, range(12))
        y_eval = y_eval + base[:, 24:]

        # deal with long tail
        for j in range(1320):
            print('zoom:', j)
            # top = base[j][12] # > bottom bottom_revise
            top = popularity_list[j][23] + base[j][23]
            bottom = np.min(y_eval[j][:4])
            # bottom_revise = base_revise[j][24:28] # >= 0
            print(y_eval[j][:4])
            if bottom < 0:
                # y_eval[j][:4] = top - (top - y_eval[j][:4]) * top / (top - bottom)
                # if (base_revise[j][24:28] == base[j][24:28]).all(): # k > 0
                y_eval[j][:4] = top - (top - y_eval[j][:4]) * (top - 1) / (top - bottom)

        y_result = np.reshape(y_eval[:, :4], (1320*4), order='F')
        write_results('Results/mlp5-No.{}'.format(i), y_result)


if __name__ == '__main__':
    main()
