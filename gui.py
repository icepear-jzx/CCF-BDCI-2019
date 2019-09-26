import matplotlib.pyplot as plt
import pandas as pd
# don't show warning
pd.set_option('mode.chained_assignment', None)


def draw(data, ylabel=None, filter={}, ifshow=False):

    if ylabel == None:
        # use the lastcolumn as ylabel
        ylabel = data.columns[-1]

    # filter
    label = ''
    for key, value in filter.items():
        data = data[data[key] == value]
        label += str(value)

    # new column called regDate
    data['regDate'] = data['regYear'] * 100 + data['regMonth']

    # sort according to regDate
    data.index = data['regDate']
    data = data.sort_index()

    # sum the data with same regDate
    data = data.groupby(data['regDate']).sum()

    # regDates are mapped to 1 ~ len(column)
    data.index = range(1, data.shape[0] + 1)

    plt.plot(data[ylabel], label=label)

    if ifshow:
        plt.show()


def show_all_salesVolume(col, ifshowlabel=False):

    # read data
    path = 'Train/train_sales_data.csv'
    with open(path, 'r') as f:
        data = pd.read_csv(f)

    for item in set(data[col]):
        draw(data.copy(), filter={col: item})

    plt.title('salesVolume of all {}s'.format(col))
    if ifshowlabel:
        plt.legend()
    plt.show()


def show_all_popularity(col, ifshowlabel=False):

    # read data
    path = 'Train/train_search_data.csv'
    with open(path, 'r') as f:
        data = pd.read_csv(f)

    for item in set(data[col]):
        draw(data.copy(), filter={col: item})

    plt.title('popularity of all {}s'.format(col))
    if ifshowlabel:
        plt.legend()
    plt.show()


def show_all_carCommentVolum(col, ifshowlabel=False):

    # read data
    path = 'Train/train_user_reply_data.csv'
    with open(path, 'r') as f:
        data = pd.read_csv(f)

    for item in set(data[col]):
        draw(data.copy(), filter={col: item})

    plt.title('carCommentVolum of all {}s'.format(col))
    if ifshowlabel:
        plt.legend()
    plt.show()


def show_all_newsReplyVolum(col, ifshowlabel=False):

    # read data
    path = 'Train/train_user_reply_data.csv'
    with open(path, 'r') as f:
        data = pd.read_csv(f)

    for item in set(data[col]):
        draw(data.copy(), filter={col: item})

    plt.title('newsReplyVolum of all {}s'.format(col))
    if ifshowlabel:
        plt.legend()
    plt.show()


def show_forecast(col, path, step_by_step=False):

    # read data
    with open(path, 'r') as f:
        result_data = pd.read_csv(f)
    with open('Train/train_sales_data.csv', 'r') as f:
        train_data = pd .read_csv(f)
    with open('Forecast/evaluation_public.csv', 'r') as f:
        forecast_data = pd.read_csv(f)
    
    # merge data
    del forecast_data['forecastVolum']
    data = pd.merge(forecast_data, result_data, on=['id'])
    data.rename(columns={'forecastVolum': 'salesVolume'}, inplace=True)
    del data['id']
    model_bodyType = train_data[['model','bodyType']].groupby(['model'], as_index=False).first()
    data = pd.merge(data, model_bodyType, on='model', how='left')
    data = pd.concat([train_data, data], sort=False)

    # draw
    plt.title('forecasting of all {}s'.format(col))
    if step_by_step:
        for item in set(data[col]):
            draw(data.copy(), filter={col: item})
            plt.axvline(24)
            # show
            plt.show()
    else:
        for item in set(data[col]):
            draw(data.copy(), filter={col: item})
        # show
        plt.axvline(24)
        plt.show()
