import keras
from keras import layers
import pandas as pd
import numpy
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
            train_list.append(df['salesVolume', 'popularity', 'carCommentVolum', 'newsReplyVolum'][index].values)

