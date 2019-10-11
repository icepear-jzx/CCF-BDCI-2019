import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math
from dataparser import write_results


def normal_func(x, a, u, sig):
    return a*np.exp(-(x - u) ** 2 / (2 * sig ** 2)) / (sig * math.sqrt(2 * math.pi))


def line_func(x, k, b):
    return k * x + b


train_file = 'Train/train_extra_data.csv'
eval_file = 'Forecast/evaluation_public.csv'

df = pd.read_csv(train_file)
train_list = []

for model in df['model'].unique():
    for adcode in df['adcode'].unique():
        index = (df['model'] == model) & (df['adcode'] == adcode)
        train_list.append(df[['salesVolume']][index].values)

y =  np.array(train_list)
y = np.reshape(y, (-1, 24))

# fit
tend_x = np.arange(5.5, 18.5, 1)
base_x = np.arange(0, 36, 1)

tend_y = []
for i in range(13):
    tend_y.append(np.mean(y[:, i:12+i], axis=1))
tend_y = np.array(tend_y).T

y_pred = np.zeros(shape=(1320, 12), dtype=np.float)
y = np.hstack([y, y_pred])
# print(y.shape)
for i in range(0,1320):
    print('fit:', i)
    para, _ = curve_fit(line_func, tend_x, tend_y[i], p0=[1, 100])
    k = para[0]
    b = para[1]
    base_y = line_func(base_x, k, b)
    if k < 0:
        para, _ = curve_fit(normal_func, tend_x, tend_y[i], p0=[10000, -10, 16], maxfev = 1000000)
        a = para[0]
        u = para[1]
        sig = para[2]
        base_y = normal_func(base_x, a, u, sig)
    
    for j in range(12):
        y[i][24+j] = (0.25*y[i][11+j] + 0.5*y[i][12+j] + 0.25*y[i][13+j]) * base_y[24+j] / base_y[12+j]
        # y[i][24+j] = (y[i][12+j]) * base_y[24+j] / base_y[12+j]

    # plt.axvline(23)
    # plt.plot(range(36), y[i])
    # plt.plot(tend_x, tend_y[i])
    # plt.plot(base_x, base_y)
    # plt.show()

y_result = np.reshape(y[:, 24:28], (1320*4), order='F')
write_results('Results/tend2', y_result)





