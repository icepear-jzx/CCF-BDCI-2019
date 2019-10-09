import keras
from keras import layers
import pandas as pd
import numpy as np
import os
from dataparser import write_results

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

            # model_onehot = model_onehot_map[model]
            bodyType_onehot = bodyType_onehot_map[model_bodyType_map[model]]
            # car_feature = np.hstack((model_onehot, bodyType_onehot))
            car_feature_list.append(bodyType_onehot)

            adcode_onehot = adcode_onehot_map[adcode]
            adcode_feature_list.append(adcode_onehot)
    
    salesVolume_list = np.reshape(np.array(salesVolume_list), (-1, 24)) # 1320 * 24
    popularity_list = np.reshape(np.array(popularity_list), (-1, 24)) # 1320 *24
    comment_list = np.reshape(np.array(comment_list), (-1, 24)) # 1320 * 24
    reply_list = np.reshape(np.array(reply_list), (-1, 24)) # 1320 * 24
    # car_feature_list = np.reshape(np.array(car_feature_list), (-1, 64)) # 1320 * (60 + 4)
    car_feature_list = np.reshape(np.array(car_feature_list), (-1, 4))
    adcode_feature_list = np.reshape(np.array(adcode_feature_list), (-1, 22)) # 1320 * 22

    return salesVolume_list, popularity_list, comment_list, reply_list, car_feature_list, adcode_feature_list


def build_mlp():
    sales = layers.Input(shape=(12, ))
    popularity = layers.Input(shape=(12, ))
    comment = layers.Input(shape=(12, ))
    reply = layers.Input(shape=(12, ))
    # car_onehot = layers.Input(shape=(64, ))
    car_onehot = layers.Input(shape=(4, ))
    adcode_onehot = layers.Input(shape=(22, ))

    car_embed = layers.Dense(4, activation='sigmoid', kernel_initializer='he_normal')(car_onehot)
    # car_embed = layers.Dense(12, activation='sigmoid', kernel_initializer='he_normal')(car_embed)

    adcode_embed = layers.Dense(12, activation='sigmoid', kernel_initializer='he_normal')(adcode_onehot)
    adcode_embed = layers.Dense(6, activation='sigmoid', kernel_initializer='he_normal')(adcode_embed)
    
    feature = layers.concatenate([sales, car_embed, adcode_embed])
    feature = layers.Dense(24, activation='sigmoid', kernel_initializer='he_normal')(feature)
    feature = layers.Dense(12, activation='sigmoid', kernel_initializer='he_normal')(feature)

    dense = layers.concatenate([sales, feature])
    dense = layers.Dense(24, activation='sigmoid', kernel_initializer='he_normal')(dense)
    dense = layers.Dense(24, activation='sigmoid', kernel_initializer='he_normal')(dense)
    output = layers.Dense(12)(dense)

    model = keras.Model([sales, popularity, comment, reply, car_onehot, adcode_onehot], output)
    model.compile(keras.optimizers.Adam(1e-2), loss=keras.losses.mse)

    return model


def my_metric(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def get_score(y_true, y_pred):
    return 1 - np.sum(np.abs(y_pred-y_true)/y_true)/60


def scale_fit(x):
    assert x.shape[1] % 12 == 0
    mu = np.zeros(shape=(12, ), dtype=np.float)
    sigma = np.zeros(shape=(12, ), dtype=np.float)
    for i in range(12):
        mu[i] = np.mean(x[:, [i + 12 * j for j in range(x.shape[1]//12)]])
        sigma[i] = np.mean(x[:, [i + 12 * j for j in range(x.shape[1]//12)]])
    return mu, sigma


def scale_to(x, mu, sigma, month_range):
    assert x.shape[1] == len(month_range)
    xs = np.zeros_like(x, dtype=np.float)
    for i in range(x.shape[1]):
        xs[:, i] = (x[:, i] - mu[month_range[i]])/sigma[month_range[i]]
    return xs


def scale_back(xs, mu, sigma, month_range):
    assert xs.shape[1] == len(month_range)
    x = np.zeros_like(xs, dtype=np.float)
    for i in range(xs.shape[1]):
        x[:, i] = xs[:, i]*sigma[month_range[i]] + mu[month_range[i]]
    return x


def main():
    salesVolume_list, popularity_list, comment_list, reply_list, \
        car_feature_list, adcode_feature_list = preprocess_train_data()

    s_mu, s_sigma = scale_fit(salesVolume_list)
    s_train = scale_to(salesVolume_list[:, :12], s_mu, s_sigma, range(12))
    y_train = scale_to(salesVolume_list[:, 12:], s_mu, s_sigma, range(12))

    p_mu, p_sigma = scale_fit(popularity_list)
    p_train = scale_to(popularity_list[:, :12], p_mu, p_sigma, range(12))

    c_mu, c_sigma = scale_fit(comment_list)
    c_train = scale_to(comment_list[:, :12], c_mu, c_sigma, range(12))

    r_mu, r_sigma = scale_fit(reply_list)
    r_train = scale_to(reply_list[:, :12], r_mu, r_sigma, range(12))

    s_test = scale_to(salesVolume_list[:, :12], s_mu, s_sigma, range(12))
    p_test = scale_to(popularity_list[:, :12], p_mu, p_sigma, range(12))
    c_test = scale_to(comment_list[:, :12], c_mu, c_sigma, range(12))
    r_test = scale_to(reply_list[:, :12], r_mu, r_sigma, range(12))
    y_true = salesVolume_list[:, 12:]

    model = build_mlp()
    model.summary()

    model.fit([s_train, p_train, c_train, r_train, car_feature_list, adcode_feature_list],
        y_train, batch_size=32, epochs=300, validation_split=0, verbose=2)
    ys_pred = model.predict([s_test, p_test, c_test, r_test, car_feature_list, adcode_feature_list])
    y_pred = scale_back(ys_pred, s_mu, s_sigma, range(12))
    rmse = my_metric(y_true, y_pred)
    print('rmse: %.3f'%rmse)

    s_eval = scale_to(salesVolume_list[:, 12:], s_mu, s_sigma, range(12))
    p_eval = scale_to(popularity_list[:, 12:], p_mu, p_sigma, range(12))
    c_eval = scale_to(comment_list[:, 12:], c_mu, c_sigma, range(12))
    r_eval = scale_to(reply_list[:, 12:], r_mu, r_sigma, range(12))
    ys_eval = model.predict([s_eval, p_eval, c_eval, r_eval, car_feature_list, adcode_feature_list])
    y_eval = scale_back(ys_eval, s_mu, s_sigma, range(12))
    y_result = np.reshape(y_eval[:, :4], (1320*4), order='F')
    write_results('Results/rmse-%d-all-data-mlp'%rmse, y_result)


if __name__ == '__main__':
    main()
