import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')


class BaseModel:
    """
    This model uses LightGBM to regress and forcast.
    It's the most basic model.
    """

    def __init__(self, params):
        self.params = params
        self.load_path = './Train/train_extra_data.csv'
        self.forecast_path = './Forecast/evaluation_public.csv'
        self.save_path = './Results/lgb-unknown.csv'
        self.raw_data = None        # raw data in train_extra_data.csv which will be split into trainset and testset
        self.raw_forecast = None     # raw data in evaluation_public.csv
        self.train = None         # data for train
        self.train_y = None
        self.test = None          # data for test
        self.test_y = None
        self.forecast = None       # data for forecasting
        self.features = None
        self.cat_feats = None


    def load_data(self):
        """
        Load data.
        """
        self.raw_data = pd.read_csv(self.load_path)
        self.raw_forecast = pd.read_csv(self.forecast_path)
        model_bodyType = self.raw_data[['model','bodyType']].groupby(['model'],as_index=False).first()
        self.raw_forecast = pd.merge(self.raw_forecast, model_bodyType, on='model', how='left')


    def gen_feat(self):
        """
        Generate features.
        """
        all_data = pd.concat([self.raw_data, self.raw_forecast]).reset_index(drop=True)
        train_test_len = self.raw_data.shape[0]

        all_data['monthNum'] = (all_data['regYear'] - 2016) * 12 + all_data['regMonth']
        

        mask = ['province','bodyType','model','monthNum','salesVolume', 'popularity', 'carCommentVolum',
            'newsReplyVolum', 'salesVolume_model_in_all_adcode', 'salesVolume_bodyType_in_all_adcode',
            'salesVolume_adcode_in_all_model', 'salesVolume_bodyType_in_all_model']
        df = all_data[mask].copy()
        for i in range(1, 13):
            history = df.copy()
            history['monthNum'] += i
            rename_list = ['salesVolume', 'popularity', 'carCommentVolum',
                'newsReplyVolum', 'salesVolume_model_in_all_adcode', 'salesVolume_bodyType_in_all_adcode',
                'salesVolume_adcode_in_all_model', 'salesVolume_bodyType_in_all_model']
            for name in rename_list:
                history.rename(columns={name: name + '_{}m_ago'.format(i)}, inplace=True)
            all_data = pd.merge(all_data, history, on=['province','bodyType','model','monthNum'], how='left')
        
        day_map = {1:31,2:28,3:31,4:30,5:31,6:30,7:31,8:31,9:30,10:31,11:30,12:31}
        all_data['dayCount'] = all_data['regMonth'].map(day_map)
        all_data.loc[(all_data.regMonth==2) & (all_data.regYear==2016), 'dayCount'] = 29
        
        train_test = all_data.iloc[:train_test_len]
        forecast = all_data.iloc[train_test_len:]
        test_mask = (train_test['regYear']==2017) & (train_test['regMonth'].isin([9,10,11,12]))
        test = train_test[test_mask].copy()
        train = train_test[~test_mask].copy()
        
        drop_l = ['id', 'adcode','forecastVolum', 'popularity', 'carCommentVolum',
                'newsReplyVolum', 'salesVolume_model_in_all_adcode', 'salesVolume_bodyType_in_all_adcode',
                'salesVolume_adcode_in_all_model', 'salesVolume_bodyType_in_all_model']
        train.drop(drop_l,axis=1,inplace=True)
        test.drop(drop_l,axis=1,inplace=True)
        forecast.drop(drop_l,axis=1,inplace=True)

        features = [_ for _ in train.columns if _ not in ['salesVolume']]
        cat_feats = ['model', 'province', 'bodyType', 'regYear', 'regMonth']

        label = 'salesVolume'
        train_y = train[label].copy()
        test_y = test[label].copy()

        for f in cat_feats:
            for df in [train, test, forecast]:
                df[f] = df[f].astype('category')

        self.train = train
        self.test = test
        self.forecast = forecast
        self.train_y = train_y
        self.test_y = test_y
        self.features = features
        self.cat_feats = cat_feats


    def lgb_train(self, verbose=10, num_round=20000):
        """
        Start training.
        """
        train = self.train.copy()
        test = self.test.copy()
        forecast = self.forecast.copy()
        train_y = self.train_y.copy()
        test_y = self.test_y.copy()
        features = self.features.copy()
        cat_feats = self.cat_feats.copy()

        train_data = lgb.Dataset(train[features], label=train_y.values)
        test_data = lgb.Dataset(test[features], label=test_y.values)

        model = lgb.train(
            self.params, 
            train_data, 
            num_round, 
            valid_sets = [train_data, test_data], 
            verbose_eval=verbose, 
            early_stopping_rounds = 200,
            categorical_feature=cat_feats,
            )


    def forcast(self):
        """
        Forcast salesVolume in 2018.1 ~ 2018.4.
        """
        pass


    def save_data(self):
        """
        Save the predicted data.
        """
        pass


if __name__ == "__main__":
    params = {
        'bagging_freq': 1,
        'bagging_fraction': 0.9,
        'learning_rate': 0.01,
        'max_depth': -1,
        "lambda_l1": 0.1,
        "lambda_l2": 1.2,
        'min_data_in_leaf': 15,
        "metric": 'mae',
        'num_leaves': 31,
        'num_threads': -1,
        'objective': 'regression',
        }
    model = BaseModel(params)
    model.load_data()
    model.gen_feat()
    model.lgb_train()
    model.save_data()