import pandas as pd


lgb = pd.read_csv('Results/lgb-0.524.csv')
mlp = pd.read_csv('Results/ywmlp-0.527.csv')
tend = pd.read_csv('Results/tend-0.54.csv')

fuse = 0.2 * lgb + 0.4 * tend + 0.4 * mlp

fuse[['id','forecastVolum']].round().astype(int).to_csv('fuse.csv',encoding='utf8',index=False)
