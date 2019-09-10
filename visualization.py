import matplotlib.pyplot as plt
import pandas as pd


def draw(data, ylabel=None, filter={}, ifshow=False):
    
    if ylabel == None:
        # use the column named *Volume as ylabel
        for col in data.columns:
            if 'Volume' in col:
                ylabel = col
                break
    
    # filter
    for key, value in filter.items():
        data = data[data[key] == value]

    # new column called regDate
    data['regDate'] = data['regYear'] * 100 + data['regMonth']

    # sort according to regDate
    data.index = data['regDate']
    data = data.sort_index()

    # sum the data with same regDate
    data = data.groupby(data['regDate']).sum()

    # regDates are mapped to 1 ~ len(column)
    data.index = range(1, data.shape[0] + 1)

    plt.plot(data[ylabel])

    if ifshow:
        plt.show()


def show_salesVolume_model():

    # read data
    path = 'Train/train_sales_data.csv'
    with open(path, 'r') as f:
        data = pd.read_csv(f)
    
    for model in set(data['model']):
        draw(data, filter={'model': model})
    
    plt.show()


show_salesVolume_model()
