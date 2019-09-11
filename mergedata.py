import pandas as pd


def merge_data():
    """
    Merge train_sales_data.csv, train_search_data.csv and train_user_reply_data.csv together.
    Output: Train/train_all_data.csv
    """

    with open('Train/train_sales_data.csv', 'r') as f:
        sales_data = pd.read_csv(f)

    with open('Train/train_search_data.csv', 'r') as f:
        search_data = pd.read_csv(f)

    with open('Train/train_user_reply_data.csv', 'r') as f:
        user_reply_data = pd.read_csv(f)

    all_data = sales_data

    match_col = ['province', 'adcode', 'model', 'regYear', 'regMonth']
    all_data = all_data.merge(search_data, on=match_col)

    match_col = ['model','regYear','regMonth']
    all_data = all_data.merge(user_reply_data, on=match_col)

    all_data.to_csv('Train/train_all_data.csv', index=False)


def test_merge_data():
    """
    To test if the output of merge_data() is right.
    """

    with open('Train/train_all_data.csv', 'r') as f:
        all_data = pd.read_csv(f)

    with open('Train/train_sales_data.csv', 'r') as f:
        sales_data = pd.read_csv(f)

    with open('Train/train_search_data.csv', 'r') as f:
        search_data = pd.read_csv(f)

    with open('Train/train_user_reply_data.csv', 'r') as f:
        user_reply_data = pd.read_csv(f)
    
    print()
