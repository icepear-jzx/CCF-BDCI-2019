import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error as mse
from tqdm import tqdm, tqdm_notebook
import warnings
warnings.filterwarnings('ignore')

path='./Train/'
train_sales_data = pd.read_csv(path+'train_sales_data.csv')
train_search_data = pd.read_csv(path+'train_search_data.csv')
train_user_reply_data = pd.read_csv(path+'train_user_reply_data.csv')
test = pd.read_csv('./Forecast/evaluation_public.csv')

# train_sales_data\train_search_data\train_user_reply_data  拼接
data = pd.merge(train_sales_data, train_search_data, 'left', on=['province', 'adcode', 'model', 'regYear', 'regMonth'])
data = pd.merge(data, train_user_reply_data, 'left', on=['model', 'regYear', 'regMonth'])
 
# col, col2, col3 中 ，设1.5倍四分位距之外的数据为异常值，用上下四分位数的均值填充
col, col2, col3 = ['popularity', 'carCommentVolum', 'newsReplyVolum']
col_per = np.percentile(data[col],(25,75))
diff = 1.5*(col_per[1] - col_per[0])
col_per_in = (data[col] >= col_per[0] - diff) & (data[col] <= col_per[1] + diff)
 
col_per2 = np.percentile(data[col2],(25,75))
diff2 = 1.5*(col_per2[1] - col_per2[0])
col_per_in2 = (data[col2] >= col_per2[0] - diff2) & (data[col2] <= col_per2[1] + diff2)
 
col_per3 = np.percentile(data[col3],(25,75))
diff3 = 1.5*(col_per3[1] - col_per3[0])
col_per_in3 = (data[col3] >= col_per3[0] - diff3) & (data[col3] <= col_per3[1] + diff3)
 
data.loc[~col_per_in, col] = col_per.mean()
data.loc[~col_per_in2, col2] = col_per2.mean()
data.loc[~col_per_in3, col3] = col_per3.mean()
 
# 统计销量
data['bt_ry_mean'] = data.groupby(['bodyType','regYear'],as_index = False)['salesVolume'].transform('mean')
data['ad_ry_mean'] = data.groupby(['adcode','regYear'],as_index = False)['salesVolume'].transform('mean')
data['md_ry_mean'] = data.groupby(['model','regYear'],as_index = False)['salesVolume'].transform('mean')

data['bt_ry_rm_sum'] = data.groupby(['bodyType','regYear','regMonth'], as_index = False)['salesVolume'].transform('sum')
data['ad_ry_rm_sum'] = data.groupby(['adcode','regYear','regMonth'],as_index = False)['salesVolume'].transform('sum')
data['md_ry_rm_sum'] = data.groupby(['model','regYear','regMonth'],as_index = False)['salesVolume'].transform('sum')

'''
一、lgb预测
'''

# 测试集并入
data = pd.concat([data, test], ignore_index=True)
data['label'] = data['salesVolume']
data['id'] = data['id'].fillna(0).astype(int)
del data['salesVolume'], data['forecastVolum']
# 填补测试集的车身类型
data['bodyType'] = data['model'].map(train_sales_data.drop_duplicates('model').set_index('model')['bodyType'])
# 编码 bodyType、model  
for i in ['bodyType', 'model']:
    data[i] = data[i].map(dict(zip(data[i].unique(), range(data[i].nunique()))))
# 距离2016年的时间间隔，月数
data['mt'] = (data['regYear'] - 2016) * 12 + data['regMonth']
 
shift_feat = []
data['model_adcode'] = data['adcode'] + data['model']
data['model_adcode_mt'] = data['model_adcode'] * 100 + data['mt']

#print(data.columns) 

# 填充测试集特征值
for col in ['carCommentVolum','newsReplyVolum','popularity','bt_ry_mean','ad_ry_mean', 'md_ry_mean','bt_ry_rm_sum','ad_ry_rm_sum','md_ry_rm_sum']:
    lgb_col_na = pd.isnull(data[col])
    data[col] = data[col].replace(0,1)
    data.loc[pd.isnull(data[col]),col] = ((((data.loc[(data['regYear'].isin([2017]))&(data['regMonth'].isin([1,2,3,4])), col].values /
    data.loc[(data['regYear'].isin([2016]))&(data['regMonth'].isin([1,2,3,4])), col].values)))*
    data.loc[(data['regYear'].isin([2017]))&(data['regMonth'].isin([1,2,3,4])), col].values * 1.03).round()

# 每年的新年在第几月份
data['happyNY'] = 0
data.loc[(data['regYear'].isin([2016,2018])&data['regMonth'].isin([2])),'happyNY'] = 1
data.loc[(data['regYear'].isin([2017])&data['regMonth'].isin([1])),'happyNY'] = 1
 

# label 下移4月，则测试集填充上了label
for month in [4]:
    for col in ['label','carCommentVolum','newsReplyVolum','popularity','bt_ry_mean','ad_ry_mean', 'md_ry_mean','bt_ry_rm_sum','ad_ry_rm_sum','md_ry_rm_sum']:
        shift_feat.append('shift_model_adcode_mt_{}_{}'.format(col,month))
        data['model_adcode_mt_{}'.format(month)] = data['model_adcode_mt'] + month
        data_last = data.loc[~pd.isnull(data['label'])].set_index('model_adcode_mt_{}'.format(month))
        data['shift_model_adcode_mt_{}_{}'.format(col,month)] = data['model_adcode_mt'].map(data_last[col])

        data.loc[pd.isnull(data['shift_model_adcode_mt_{}_{}'.format(col,month)]),'shift_model_adcode_mt_{}_{}'.format(col,month)] = (((data.loc[(data.regMonth> 12-month) & (data.regMonth <= 12) & data.regYear.isin([2016]),i].values)/
 data.loc[(data.regMonth> 12 - month) & (data.regMonth <= 12) & data.regYear.isin([2017]),i].values)*
data.loc[(data.regMonth> 12- month) & (data.regMonth <= 12) & data.regYear.isin([2016]),i].values).round()

################ adpated from yueyueniao  more history information ###################################
'''
for month in [1]:
    for col in ['label','carCommentVolum','newsReplyVolum','popularity']:
        df = data[['province','bodyType','model','mt',col]].copy()
        for i in list(range(1, month+4)) + [12]:
            history = df.copy()
            history['mt'] += i
            history.rename(columns={col:'{}_{}m_ago'.format(col,i)},inplace=True)
            shift_feat.append('{}_{}m_ago'.format(col,i))
            data = pd.merge(data, history, on=['province','bodyType','model','mt'], how='left')
            #print(data.columns.values)
            #data.loc[pd.isnull(data['{}_{}m_ago'.format(col,i)]),'{}_{}m_ago'.format(col,i)] = data.groupby(['province','bodyType','model','regYear','regMonth'],as_index = False)[col].transform('mean')

          
def getHistoryIncrease(step=1, wind=1, col='salesVolume'):
        """
        计算历史涨幅
        :param: step:月份跨度
        :param: wind:计算涨幅的月份区间
        :param: col:计算涨幅的目标列
        例：step=1,wind=2,计算当月 前第1月 较 前第3月 的涨幅）
        """
        if col not in self.unstack_data.keys():
            res = []
            bar = tqdm(data['province'].unique(), desc='history increase')
            for i in bar:
                for j in data['model'].unique():
                    msk = (data['province']==i) & (data['model']==j)
                    df = data[msk].copy().reset_index(drop=True)
                    df = df[['mt',col]].set_index('mt').T
                    df['province'] = i
                    df['model'] = j
                    res.append(df)
            res = pd.concat(res).reset_index(drop=True)
            self.unstack_data[col] = res.copy()

for month in [1,2,3,4]:
    base_step = month-1 if month-1>0 else 1
    data = getHistoryIncrease(step=base_step)
    data = getHistoryIncrease(step=base_step+1)
    trainset = getHistoryIncrease(step=base_step+2)

    trainset = getHistoryIncrease(step=base_step, wind=2)
    trainset = getHistoryIncrease(step=base_step+1, wind=2)
    trainset = getHistoryIncrease(step=base_step+2, wind=2)
    trainset = getHistoryIncrease(step=base_step, wind=12)
        
    trainset = getHistoryIncrease(trainset, step=month, col='popularity')
    trainset = getHistoryIncrease(trainset, step=month+1, col='popularity')
    trainset = getHistoryIncrease(trainset, step=month+2, col='popularity')

    trainset = getHistoryIncrease(trainset, step=month, wind=2, col='popularity')
    trainset = getHistoryIncrease(trainset, step=month+1, wind=2, col='popularity')
    trainset = getHistoryIncrease(trainset, step=month+2, wind=2, col='popularity')
'''
################################## end ##################################################

# 根据月份添加权重值
a = 6; b = 4
data['weightMonth'] = data['regMonth'].map({1:a, 2:a, 3:a, 4:a,
                                            5:b, 6:b, 7:b, 8:b, 9:b, 10:b, 11:b, 12:b,})
 

  
def score(data):
    pred = data.groupby(['adcode', 'model'])['pred_label'].agg(lambda x: list(x))
    label = data.groupby(['adcode', 'model'])['label'].agg(lambda x: list(x))
    label_mean = data.groupby(['adcode', 'model'])['label'].agg(lambda x: np.mean(x))
    data_agg = pd.DataFrame()
    data_agg['pred_label'] = pred
    data_agg['label'] = label
    data_agg['label_mean'] = label_mean
    nrmse_score = []
    for raw in data_agg.values:
        nrmse_score.append(mse(raw[0], raw[1]) ** 0.5 / raw[2])
    return 1 - np.mean(nrmse_score)
 
 
df_lgb = pd.DataFrame({'id': test['id']})
for col_add in ['ad_ry_mean', 'md_ry_mean', 'bt_ry_mean']:
    # 取用的字段，用于训练模型
    num_feat = shift_feat
    cate_feat = ['adcode', 'bodyType', 'model', 'regYear', 'regMonth', 'happyNY']
    #features = num_feat + cate_feat + ['weightMonth','ad_ry_rm_sum','md_ry_rm_sum','bt_ry_rm_sum'] + [col_add]  # [ad_ry_mean, md_ry_mean, bt_ry_mean]
    features = num_feat + cate_feat+['weightMonth']

    train_idx = (data['mt'] <= 20) # 小于等于20月以内的数据作为训练集
    valid_idx = (data['mt'].between(21, 24)) # 21到24个月的数据作为验证集
    test_idx = (data['mt'] > 24) # 大于24个月的是测试集
 
    # label
    data['n_label'] = np.log(data['label'])
 
    train_x = data[train_idx][features]
    train_y = data[train_idx]['n_label']
 
    valid_x = data[valid_idx][features]
    valid_y = data[valid_idx]['n_label']
 
    ############################### lgb ###################################
    lgb_model = lgb.LGBMRegressor(
        num_leaves=40, reg_alpha=1, reg_lambda=0.1, objective='mse',
        max_depth=-1, learning_rate=0.05, min_child_samples=5, random_state=2019,
        n_estimators=8000, subsample=0.8, colsample_bytree=0.8)
 
    lgb_model.fit(train_x, train_y, eval_set=[(valid_x, valid_y)],
                  categorical_feature=cate_feat, early_stopping_rounds=100, verbose=300)
    data['pred_label'] = np.e ** lgb_model.predict(data[features])
    model = lgb_model

    # 特征重要程度
    print ('lgb Features:',sorted(dict(zip(train_x.columns,model.feature_importances_)).items(),key=lambda x: x[1], reverse=True))
    print('AVE NRMSE:',score(data = data[valid_idx]))
    model.n_estimators = model.best_iteration_
    model.fit(data[~test_idx][features], data[~test_idx]['n_label'], categorical_feature=cate_feat)
    data['forecastVolum'] = np.e ** model.predict(data[features])
    sub = data[test_idx][['id']]
    sub['forecastVolum'] = data[test_idx]['forecastVolum'].apply(lambda x: 0 if x < 0 else x).round().astype(int)
    sub_lgb = sub.reset_index(drop=True)
    sub_lgb = sub_lgb[['id','forecastVolum']]
    #print('lgb中forecastVolmn:',(sub_lgb['forecastVolum']==0).sum())
    df_lgb[col_add] = sub_lgb['forecastVolum']
    
df_lgb.to_csv("Results/df_lgb.csv", index=False) 
sub = pd.DataFrame(columns= ['id', 'forecastVolum'])
sub['id'] = df_lgb['id']
sub['forecastVolum']= ((df_lgb['ad_ry_mean']+df_lgb['md_ry_mean'] + df_lgb['bt_ry_mean'])/3).astype(int)
#sub.set_index('id')
sub.to_csv('Results/sub_fe7.csv',index = False)