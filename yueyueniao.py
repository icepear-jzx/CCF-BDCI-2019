import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

import gc
from tqdm import tqdm as tqdm # 如果使用.py文件的话可以把tqdm_notebook改成tqdm

import warnings
warnings.filterwarnings('ignore')

class BaseModel:
    def __init__(self):
        self.unstack_data = {}
        
        self.train = None
        self.test = None
        self.valid = None
        self.train_target = None
        self.valid_target = None
        self.features = None
        self.cat_feats = None
        self.weight = None
        
        self.model = None
    # 读取数据
    def readData(self,path='./Train/'):
        """
        读取数据集
        """
        train_sales = pd.read_csv(path+'train_sales_data.csv')
        train_search = pd.read_csv(path+'train_search_data.csv')
        test = pd.read_csv('./Forecast/'+'evaluation_public.csv')

        train = pd.merge(train_sales, train_search, on=['province','adcode','model','regYear','regMonth'],how='left')

        model_bodyType = train[['model','bodyType']].groupby(['model'],as_index=False).first()
        test = pd.merge(test, model_bodyType, on='model', how='left')
        return train, test
    # 特征工程
    def transStrFeats(self, df, l):
        """
        字符特征转换
        :param: l:要转码的特征列表
        """
        for f in l:
            map_dict = {i:j+1 for j,i in enumerate(df[f].unique())}
            df[f] = df[f].map(map_dict)
        return df
    def getHistoryIncrease(self, dataset, step=1, wind=1, col='salesVolume'):
        """
        计算历史涨幅
        :param: step:月份跨度
        :param: wind:计算涨幅的月份区间
        :param: col:计算涨幅的目标列
        例：step=1,wind=2,计算当月 前第1月 较 前第3月 的涨幅）
        """
        if col not in self.unstack_data.keys():
            res = []
            bar = tqdm(dataset['province'].unique(), desc='history increase')
            for i in bar:
                for j in dataset['model'].unique():
                    msk = (dataset['province']==i) & (dataset['model']==j)
                    df = dataset[msk].copy().reset_index(drop=True)
                    df = df[['mt',col]].set_index('mt').T
                    df['province'] = i
                    df['model'] = j
                    res.append(df)
            res = pd.concat(res).reset_index(drop=True)
            self.unstack_data[col] = res.copy()
            
        res = self.unstack_data[col].copy()
        res_ = res.copy()
        for i in range(step+wind+1,29):
            res_[i] = (res[i-step] - res[i-(step+wind)]) / res[i-(step+wind)]
        for i in range(1,step+wind+1):
            res_[i]=np.NaN
        res = res_.set_index(['province','model']).stack().reset_index()
        res.rename(columns={0:'{}_last{}_{}_increase'.format(col,step,wind)},inplace=True)
        dataset = pd.merge(dataset, res, on=['province','model','mt'], how='left')

        return dataset
    
    def genDataset(self, pred={}, month=1):
        """
        生成做好特征工程的数据集
        :param: pred: 测试集预测数据字典 {月份:预测结果}
        :param: month: 当前预测的月份
        """
        train, test = self.readData()
        trainset = pd.concat([train, test]).reset_index(drop=True)
        train_len = train.shape[0]

        
        trainset = self.transStrFeats(trainset, ['model'])
        trainset['mt'] = (trainset['regYear'] - 2016) * 12 + trainset['regMonth']

        if len(pred)>0:
            for m in pred.keys():
                test['salesVolume'] = pred[m]
                msk = test['regMonth']==m
                trainset['salesVolume'][trainset['mt']==(24+m)] = test['salesVolume'][msk].values
        
        #############################特征工程#############################
        
        df = trainset[['province','bodyType','model','mt','salesVolume']].copy()
        for i in [1,2,3,4,5,6,7,8,9,10,11,12]:
            history = df.copy()
            history['mt'] += i
            history.rename(columns={'salesVolume':'Label_{}m_ago'.format(i)},inplace=True)
            trainset = pd.merge(trainset, history, on=['province','bodyType','model','mt'], how='left')
        
        df = trainset[['province','bodyType','model','mt','popularity']].copy()
        for i in [4,5,6]:
            history = df.copy()
            history['mt'] += i
            history.rename(columns={'popularity':'popularity_{}m_ago'.format(i)},inplace=True)
            trainset = pd.merge(trainset, history, on=['province','bodyType','model','mt'], how='left')
        
        day_map = {1:31,2:28,3:31,4:30,5:31,6:30,7:31,8:31,9:30,10:31,11:30,12:31}
        trainset['dayCount']=trainset['regMonth'].map(day_map)
        trainset.loc[(trainset.regMonth==2)&(trainset.regYear==2016),'dayCount']=29
        trainset['salesVolume']/=trainset['dayCount']
        trainset['popularity']/=trainset['dayCount']
        
        base_step = month-1 if month-1>0 else 1
        trainset = self.getHistoryIncrease(trainset, step=base_step)
        trainset = self.getHistoryIncrease(trainset, step=base_step+1)
        trainset = self.getHistoryIncrease(trainset, step=base_step+2)
        trainset = self.getHistoryIncrease(trainset, step=base_step, wind=2)
        trainset = self.getHistoryIncrease(trainset, step=base_step+1, wind=2)
        trainset = self.getHistoryIncrease(trainset, step=base_step+2, wind=2)
        trainset = self.getHistoryIncrease(trainset, step=base_step, wind=12)
        
        trainset = self.getHistoryIncrease(trainset, step=month, col='popularity')
        trainset = self.getHistoryIncrease(trainset, step=month+1, col='popularity')
        trainset = self.getHistoryIncrease(trainset, step=month+2, col='popularity')
        trainset = self.getHistoryIncrease(trainset, step=month, wind=2, col='popularity')
        trainset = self.getHistoryIncrease(trainset, step=month+1, wind=2, col='popularity')
        trainset = self.getHistoryIncrease(trainset, step=month+2, wind=2, col='popularity')
        
        trainset['salesVolume']*=trainset['dayCount']
        trainset['popularity']*=trainset['dayCount']
        # 划分训练、验证集
        train = trainset.iloc[:train_len]
        test = trainset.iloc[train_len:]
        valid_mask = (train['regYear']==2017) & (train['regMonth'].isin([9,10,11,12]))
        valid = train[valid_mask].copy()
        train = train[~valid_mask].copy()
        
        # 去掉无效特征
        drop_l = ['adcode','forecastVolum','id']
        train.drop(drop_l,axis=1,inplace=True)
        test.drop(drop_l,axis=1,inplace=True)
        valid.drop(drop_l,axis=1,inplace=True)

        # 生成特征列表，训练标签
        features = [_ for _ in train.columns if _ not in ['salesVolume']]
        cat_feats = ['model', 'province', 'bodyType', 'regMonth']

        label = 'salesVolume'
        train_target = train[label].copy()
        valid_target = valid[label].copy()

        for f in cat_feats:
            for df in [train, test, valid]:
                df[f] = df[f].astype('category')

        self.train = train
        self.test = test
        self.valid = valid
        self.train_target = train_target
        self.valid_target = valid_target
        self.features = features
        self.cat_feats = cat_feats
    
    # 模型训练
    def getScore(self, df, oof):
        score = 0
        for f in df['model'].unique():
            msk = df['model']==f
            tmp = df[msk]
            score += np.sqrt(mean_squared_error(tmp['salesVolume'],oof[msk]))/(tmp['salesVolume']).mean()
        score = 1-score/df['model'].nunique()
        return score
    def lgb_train(self, param, data=None, verbose=10, num_round=20000):
        if data is None:
            train = self.train.copy()
            test = self.test.copy()
            valid = self.valid.copy()
            train_target = self.train_target.copy()
            valid_target = self.valid_target.copy()
            features = self.features.copy()
            cat_feats = self.cat_feats.copy()
        else:
            train = data['train'].copy()
            test = data['test'].copy()
            valid = data['valid'].copy()
            train_target = data['train_target'].copy()
            valid_target = data['valid_target'].copy()
            features = data['features'].copy()
            cat_feats = data['cat_features'].copy()


        trn_data = lgb.Dataset(train[features], label=train_target.values)
        val_data = lgb.Dataset(valid[features], label=valid_target.values)
        feature_importance_df = pd.DataFrame()

        model = lgb.train(
            param, 
            trn_data, 
            num_round, 
            valid_sets = [trn_data, val_data], 
            verbose_eval=verbose, 
            early_stopping_rounds = 200,
            categorical_feature=cat_feats,
            )

        feature_importance_df["Feature"] = features
        feature_importance_df["importance"] = model.feature_importance()
        oof = model.predict(valid[features], num_iteration=model.best_iteration)
        print("CV score: {:<8.5f}".format(self.getScore(valid, oof)))
        
        train_all = pd.concat([train, valid]).reset_index(drop=True)
        for f in cat_feats:
            train_all[f] = train_all[f].astype('category')
        target_all = train_target.append(valid_target)
        model = model.refit(train_all[features], target_all, categorical_feature=cat_feats)
        pred = model.predict(test[features], num_iteration=model.best_iteration)
        self.model = model
        
        gc.collect()
        return oof, pred, feature_importance_df


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


baseModel = BaseModel()
pred_dict = {}
for i in [1,2,3,4]:
    baseModel.genDataset(pred=pred_dict, month=i)
    oof, pred, feat_importance = baseModel.lgb_train(params)
    pred_dict[i] = pred


test = baseModel.test.copy()
test['pred1'] = pred_dict[1]
test['pred2'] = pred_dict[2]
test['pred3'] = pred_dict[3]
test['pred4'] = pred_dict[4]

msk1 = test['regMonth']==1
msk2 = test['regMonth']==2
msk3 = test['regMonth']==3
msk4 = test['regMonth']==4

test['pred'] = 0.
test['pred'][msk1] = test['pred1'][msk1]
test['pred'][msk2] = test['pred2'][msk2]
test['pred'][msk3] = test['pred3'][msk3]
test['pred'][msk4] = test['pred4'][msk4]

PRED = test['pred'].values
submit = pd.read_csv('./Forecast/evaluation_public.csv')
submit['forecastVolum'] = PRED
submit['forecastVolum'][submit['forecastVolum']<0] = 0
submit[['id','forecastVolum']].round().astype(int).to_csv('submit.csv',encoding='utf8',index=False)
