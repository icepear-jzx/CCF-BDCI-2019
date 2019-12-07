import pandas as pd
import numpy as np


# tend1 = pd.read_csv('Results/tend-0.5405.csv')
# tend2 = pd.read_csv('Results/tend2-0.5377.csv')

# fuse = 0.5 * tend1 + 0.5 * tend2
# fuse[['id','forecastVolum']].round().astype(int).to_csv('tend-fuse.csv',encoding='utf8',index=False)

# lgb1 = pd.read_csv('Results/lgb[1]-0.5651.csv')
# lgb2 = pd.read_csv('Results/lgb[2]-0.5708.csv')
# lgb3 = pd.read_csv('Results/lgb[3]-0.5699.csv')

# fuse = (lgb1 + lgb2 + lgb3) / 3
# fuse[['id','forecastVolum']].round().astype(int).to_csv('lgb-fuse.csv',encoding='utf8',index=False)

lgb = pd.read_csv('Results/lgb-0.6254.csv')
lgb_fuse = pd.read_csv('Results/lgb-fuse-0.5718.csv')
mlp6 = pd.read_csv('Results/mlp6-fuse-0.5664.csv')
xgb = pd.read_csv('Results/xgb-0.5976.csv')
fuse6005 = pd.read_csv('Results/fuse-0.6005.csv')
lgbxgb = pd.read_csv('Results/lgb-xgb-0.6104.csv')
model1 = pd.read_csv('model_1.csv')
model2 = pd.read_csv('model_2.csv')

fuse = 0.3 * model1 + 0.7 * model2

fuse[['id','forecastVolum']].round().astype(int).to_csv('fuse.csv',encoding='utf8',index=False)

# mul_fuse = (lgb * xgb).apply(np.sqrt)

# mul_fuse[['id','forecastVolum']].round().astype(int).to_csv('mul_fuse.csv',encoding='utf8',index=False)

# mlps = []
# for i in range(10):
#     mlps.append(pd.read_csv('Results/mlp7-No.{}.csv'.format(i)))
# fuse = 0
# for i in range(10):
#     fuse += mlps[i] * 0.1
# fuse[['id','forecastVolum']].round().astype(int).to_csv('mlp7-fuse.csv',encoding='utf8',index=False)


# lgb = pd.read_csv('lgb.csv')

# # noise = pd.DataFrame(np.random.choice([-1, 0, 1], 5280), columns=['noise'])

# # xgb['forecastVolum'] = xgb['forecastVolum'] + noise['noise']
# lgb['forecastVolum'] = lgb['forecastVolum'].apply(lambda x: 0 if x < 0 else x)

# lgb[['id','forecastVolum']].round().astype(int).to_csv('lgb.csv',encoding='utf8',index=False)
