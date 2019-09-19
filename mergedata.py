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


def gen_extra_data():
    """
    Add extra data into train_extra_data.csv.
    For example, salesVolume in a province etc.
    """

    with open('Train/train_all_data.csv', 'r') as f:
        all_data = pd.read_csv(f)
    
    model_set = set(all_data.model)
    adcode_set = set(all_data.adcode)

    for regYear in range(2016, 2018):
        filter_regYear = (all_data.regYear == regYear)
        for regMonth in range(1, 13):
            filter_regMonth = (all_data.regMonth == regMonth)
            for model in model_set:
                filter_model = (all_data.model == model)
                bodyType = all_data[filter_model].iloc[0].bodyType
                filter_bodyType = (all_data.bodyType == bodyType)
                # salesVolume of single model in all provinces
                filter_model_regYear_regMonth = filter_model & filter_regYear & filter_regMonth
                salesVolume_model_in_all_adcode = all_data.salesVolume[filter_model_regYear_regMonth].sum()
                all_data.loc[filter_model_regYear_regMonth, 'salesVolume_model_in_all_adcode'] = salesVolume_model_in_all_adcode
                # salesVolume of single bodyType in all provinces
                filter_bodyType_regYear_regMonth = filter_bodyType & filter_regYear & filter_regMonth
                salesVolume_bodyType_in_all_adcode = all_data.salesVolume[filter_bodyType_regYear_regMonth]
                all_data.loc[filter_bodyType_regYear_regMonth, 'salesVolume_bodyType_in_all_adcode'] = salesVolume_bodyType_in_all_adcode
                
                for adcode in adcode_set:
                    filter_adcode = (all_data.adcode == adcode)
                    # salesVolume of all models in single province
                    filter_adcode_regYear_regMonth = filter_adcode & filter_regYear & filter_regMonth
                    salesVolume_adcode_in_all_model = all_data.salesVolume[filter_adcode_regYear_regMonth].sum()
                    all_data.loc[filter_adcode_regYear_regMonth, 'salesVolume_adcode_in_all_model'] = salesVolume_adcode_in_all_model
                    # salesVolume of all models with the same bodyType in single provimce
                    filter_adcode_bodyType_regYear_regMonth = filter_adcode & filter_bodyType_regYear_regMonth
                    salesVolume_bodyType_in_all_model = all_data.salesVolume[filter_adcode_bodyType_regYear_regMonth].sum()
                    all_data.loc[filter_adcode_bodyType_regYear_regMonth, 'salesVolume_bodyType_in_all_model'] = salesVolume_bodyType_in_all_model
    
    all_data.to_csv('Train/train_extra_data.csv', index=False)
    

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
    
    print('Start test...')

    for index, row in all_data.iterrows():

        print('Test:', index)

        match_col = ['province', 'adcode', 'model', 'regYear', 'regMonth']
        test_col = ['popularity']
        match_row = search_data
        for col in match_col:
            match_row = match_row[match_row[col] == row[col]]
        if (match_row[test_col] != row[test_col]).any().any():
            print('Error!!!')
            return False

        match_col = ['model','regYear','regMonth']
        test_col = ['carCommentVolum', 'newsReplyVolum']
        match_row = user_reply_data
        for col in match_col:
            match_row = match_row[match_row[col] == row[col]]
        if (match_row[test_col] != row[test_col]).any().any():
            print('Error!!!')
            return False
    
    print('Success.')
    return True


gen_extra_data()