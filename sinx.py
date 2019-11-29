import sys
# import shap
import numpy as np
import pandas as pd
import os 
import gc
from tqdm import tqdm, tqdm_notebook 
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import LabelEncoder
import datetime
import time
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.cluster import KMeans
import xgboost as xgb
from sklearn.externals import joblib
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
train_sales  = pd.read_csv('Train/train_sales_data.csv')
train_search = pd.read_csv('Train/train_search_data.csv')
train_user   = pd.read_csv('Train/train_user_reply_data.csv')
evaluation_public = pd.read_csv('Forecast/evaluation_public.csv')
submit_example    = pd.read_csv('Example/submit_example.csv')
data = pd.concat([train_sales, evaluation_public], ignore_index=True)
data = data.merge(train_search, 'left', on=['province', 'adcode', 'model', 'regYear', 'regMonth'])
data = data.merge(train_user, 'left', on=['model', 'regYear', 'regMonth'])
# data=pd.concat([data, k_mean_data], axis=1)
data['label'] = data['salesVolume']
data['id'] = data['id'].fillna(0).astype(int)
data['bodyType'] = data['model'].map(train_sales.drop_duplicates('model').set_index('model')['bodyType'])
#LabelEncoder
for i in ['bodyType', 'model']:
    data[i] = data[i].map(dict(zip(data[i].unique(), range(data[i].nunique()))))
data['mt'] = (data['regYear'] - 2016) * 12 + data['regMonth']

def get_stat_feature(df_,): 
    df = df_.copy()
    stat_feat = []
    stat_feat_2=[]
    stat_feat_3 = []
    stat_feat_4 = []
    df['model_adcode'] = df['adcode'] + df['model']
    df['model_adcode_mt'] = df['model_adcode'] * 100 + df['mt']
    for col in ['label']:
        # 历史销量数据特征
        for i in [1,2,3,4,5,6,8,9,10,11,12,13,14,15,16]:
            stat_feat.append('shift_model_adcode_mt_{}_{}'.format(col,i))
            stat_feat_2.append('shift_model_adcode_mt_{}_{}'.format(col,i))
            df['model_adcode_mt_{}_{}'.format(col,i)] = df['model_adcode_mt'] + i#新加一列值，等于车型*省*时间+i，寻求i个月前的值，将model_adcode_mt_作为索引
            df_last = df[~df[col].isnull()].set_index('model_adcode_mt_{}_{}'.format(col,i))
            df['shift_model_adcode_mt_{}_{}'.format(col,i)] = df['model_adcode_mt'].map(df_last[col])#后者索引是31000002开始，前者少i，取前面的匹配后面索引成功，就取值
    for col in ['popularity']:
        # 历史销量数据特征
        for i in [1,2,3,10,11,12]:#popularity只取一部分
            stat_feat.append('shift_model_adcode_mt_{}_{}'.format(col,i))
            stat_feat_2.append('shift_model_adcode_mt_{}_{}'.format(col,i))
            df['model_adcode_mt_{}_{}'.format(col,i)] = df['model_adcode_mt'] + i#新加一列值，等于车型*省*时间+i，寻求i个月前的值，将model_adcode_mt_作为索引
            df_last = df[~df[col].isnull()].set_index('model_adcode_mt_{}_{}'.format(col,i))
            df['shift_model_adcode_mt_{}_{}'.format(col,i)] = df['model_adcode_mt'].map(df_last[col])#后者索引是31000002开始，前者少i，取前面的匹配后面索引成功，就取值
    df["increase16_4"]=(df["shift_model_adcode_mt_label_16"]-df["shift_model_adcode_mt_label_4"])/df["shift_model_adcode_mt_label_16"]#同比一年前的增长
    mean=pd.DataFrame(df.groupby(["model","mt"]).shift_model_adcode_mt_label_12.agg({"mean_province":"mean",
                                                                          "min_province":"min",}))
    df=pd.merge(df,mean,on=["model","mt"],how="left")
    mean=pd.DataFrame(df.groupby(["model","mt"]).shift_model_adcode_mt_label_15.agg({"mean_province_15":"mean",}))
    df=pd.merge(df,mean,on=["model","mt"],how="left")
    mean=pd.DataFrame(df.groupby(["model","mt"]).shift_model_adcode_mt_label_3.agg({"mean_province_3":"mean",}))
    df=pd.merge(df,mean,on=["model","mt"],how="left")
    mean=pd.DataFrame(df.groupby(["model","mt"]).shift_model_adcode_mt_label_16.agg({"mean_province_16":"mean",}))
    df=pd.merge(df,mean,on=["model","mt"],how="left")
    mean=pd.DataFrame(df.groupby(["model","mt"]).shift_model_adcode_mt_label_4.agg({"mean_province_4":"mean",}))
    df=pd.merge(df,mean,on=["model","mt"],how="left")
	#另一种统计方式
    mean=pd.DataFrame(df.groupby(["adcode","mt"]).shift_model_adcode_mt_label_15.agg({"mean_Month_15":"mean"}))
    df=pd.merge(df,mean,on=["adcode","mt"],how="left") 
    mean=pd.DataFrame(df.groupby(["adcode","mt"]).shift_model_adcode_mt_label_3.agg({"mean_Month_3":"mean"}))
    df=pd.merge(df,mean,on=["adcode","mt"],how="left") 
    mean=pd.DataFrame(df.groupby(["adcode","mt"]).shift_model_adcode_mt_label_16.agg({"mean_Month_16":"mean"}))
    df=pd.merge(df,mean,on=["adcode","mt"],how="left") 
    mean=pd.DataFrame(df.groupby(["adcode","mt"]).shift_model_adcode_mt_label_4.agg({"mean_Month_4":"mean"}))
    df=pd.merge(df,mean,on=["adcode","mt"],how="left") 
    #基于统计特征的increase，强特
    df["increase_mean_province_16_4"]=(df["mean_province_16"]-df["mean_province_4"])/df["mean_province_16"]
    df["increase_mean_province_15_3"]=(df["mean_province_15"]-df["mean_province_3"])/df["mean_province_15"]
    df["increase_mean_Month_15_3"]=(df["mean_Month_15"]-df["mean_Month_3"])/df["mean_Month_15"]
    df["increase_mean_Month_16_4"]=(df["mean_Month_16"]-df["mean_Month_4"])/df["mean_Month_16"]
    mean=pd.DataFrame(df.groupby(["adcode","mt"]).shift_model_adcode_mt_label_12.agg({"mean_Month":"mean",}))
    df=pd.merge(df,mean,on=["adcode","mt"],how="left")
 	#几个月sum
    df["sum_1"]=df["shift_model_adcode_mt_label_11"].values+df["shift_model_adcode_mt_label_12"].values+df["shift_model_adcode_mt_label_1"].values+df["shift_model_adcode_mt_label_2"].values
    df["sum_2"]=df["shift_model_adcode_mt_label_12"].values+df["shift_model_adcode_mt_label_1"].values
    df["sum_3"]=df["shift_model_adcode_mt_label_3"].values+df["shift_model_adcode_mt_label_2"].values+df["shift_model_adcode_mt_label_1"].values
    stat_feat_4 = ["mean_province","min_province","mean_Month","sum_1","sum_2","sum_3","increase16_4",
                   "increase_mean_province_15_3","increase_mean_Month_15_3","increase_mean_province_16_4","increase_mean_Month_16_4"]#所有统计特征
    stat_feat.remove("shift_model_adcode_mt_label_15")#删掉两个特征
    stat_feat.remove("shift_model_adcode_mt_label_16")
    return df,stat_feat+stat_feat_3+stat_feat_4
	#下面基本和鱼佬一样

def score(data, pred='pred_label', label='label', group='model'):
    data['pred_label'] = data['pred_label'].apply(lambda x: 0 if x < 0 else x).round().astype(int)
    data_agg = data.groupby('model').agg({
        pred:  list,
        label: [list, 'mean']
    }).reset_index()
    data_agg.columns = ['_'.join(col).strip() for col in data_agg.columns]
    nrmse_score = []
    for raw in data_agg[['{0}_list'.format(pred), '{0}_list'. format(label), '{0}_mean'.format(label)]].values:
        nrmse_score.append(mse(raw[0], raw[1]) ** 0.5 / raw[2] )
    print(1 - np.mean(nrmse_score))
    return 1 - np.mean(nrmse_score)

def get_model_type(train_x,train_y,valid_x,valid_y,m_type='lgb',i=0):   
    if m_type == 'lgb':
        model = lgb.LGBMRegressor(
                                num_leaves=2**5-1, reg_alpha=0.25, reg_lambda=0.25, objective='mse',
                                max_depth=-1, learning_rate=0.05, min_child_samples=10, random_state=2019,
                                n_estimators=2000, subsample=0.9, colsample_bytree=0.7,num_threads= -1,
                                )
        model.fit(train_x, train_y, 
              eval_set=[(train_x, train_y),(valid_x, valid_y)], 
              categorical_feature=cate_feat, 
              early_stopping_rounds=100, verbose=100)
        joblib.dump(model, "lgbm_"+str(i)+".m")
        print("lgb_model_%d has saved"%i)
    elif m_type == 'xgb':
        model = xgb.XGBRegressor(
                                max_depth=5 , learning_rate=0.05, n_estimators=2000, 
                                objective='reg:gamma', tree_method = 'hist',subsample=0.9, 
                                colsample_bytree=0.7, min_child_samples=5,eval_metric = 'rmse' 
                                )
        model.fit(train_x, train_y, 
              eval_set=[(train_x, train_y),(valid_x, valid_y)], 
              early_stopping_rounds=100, verbose=100)   
        joblib.dump(model, "xgbm_"+str(i)+".m")
        print("xgb_model_%d has saved"%i)
    return model

def get_train_model(df_, m, m_type='lgb',i=0):
    df = df_.copy()
    # 数据集划分m=25,26,27,28,
    st = 13#start time 
    all_idx   = (df['mt'].between(st , m-1))#原版
    train_idx = (df['mt'].between(st , m-5))
    valid_idx = (df['mt'].between(m-4, m-4))
    test_idx  = (df['mt'].between(m  , m  ))
    # 最终确认
    train_x = df[train_idx][features]
    train_y = df[train_idx]['n_label']
    valid_x = df[valid_idx][features]
    valid_y = df[valid_idx]['n_label']   
    # get model
    model = get_model_type(train_x,train_y,valid_x,valid_y,m_type,i)  
    # offline
    df['pred_label'] = np.expm1(model.predict(df[features]))
    best_score = score(df[valid_idx]) 
    # online
    if m_type == 'lgb':
        model.n_estimators = model.best_iteration_ + 100
        model.fit(df[all_idx][features], df[all_idx]['n_label'], categorical_feature=cate_feat)
    elif m_type == 'xgb':
        model.n_estimators = model.best_iteration + 100
        model.fit(df[all_idx][features], df[all_idx]['n_label'])
    df['forecastVolum'] = np.expm1(model.predict(df[features]))
    print('valid mean:',df[valid_idx]['pred_label'].mean())
    print('true  mean:',df[valid_idx]['label'].mean())
    print('test  mean:',df[test_idx]['forecastVolum'].mean())
    # 阶段结果
    sub = df[test_idx][['id']]
    sub['forecastVolum'] = df[test_idx]['forecastVolum'].apply(lambda x: 0 if x < 0 else x).round().astype(int) 
    print(sub.shape)
    return sub,df[valid_idx]['pred_label']

for month in [25,26,27,28]: 
    m_type = 'lgb' 
    data['n_label'] = np.log1p(data['label'])
    data_df, stat_feat = get_stat_feature(data)#每次都要更新下特征
    num_feat = ['regYear'] + stat_feat
    cate_feat = ['adcode','bodyType','model','regMonth',]#,'k_mean_1','k_mean'
    if m_type == 'lgb':
        for i in cate_feat:
            data_df[i] = data_df[i].astype('category')
    elif m_type == 'xgb':
        lbl = LabelEncoder()  
        for i in tqdm(cate_feat):
            data_df[i] = lbl.fit_transform(data_df[i].astype(str)) 
    features = num_feat + cate_feat
    print(len(features), len(set(features)))   
    sub,val_pred = get_train_model(data_df, month, m_type,month-24)   
    data.loc[(data.regMonth==(month-24))&(data.regYear==2018), 'salesVolume'] = sub['forecastVolum'].values
    data.loc[(data.regMonth==(month-24))&(data.regYear==2018), 'label'      ] = sub['forecastVolum'].values
sub = data.loc[(data.regMonth>=1)&(data.regYear==2018), ['id','salesVolume']]
sub.columns = ['id','forecastVolum']
sub[['id','forecastVolum']].round().astype(int).to_csv('B_res.csv', index=False)
#结果基于规则纠正
my_data=pd.read_csv('B_res.csv')
my_data["forecastVolum"]=my_data["forecastVolum"]*0.79-5
my_data["forecastVolum"]=(my_data["forecastVolum"]).astype(int)
my_data.loc[my_data[my_data["forecastVolum"] < 4].index,"forecastVolum"]=4
my_data.loc[my_data[my_data["forecastVolum"] >9000].index,"forecastVolum"]=9000
my_data.to_csv('submit.csv',index=0)


