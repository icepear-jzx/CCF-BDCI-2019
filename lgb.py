import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')


class LGBModel:
    """
    This model uses LightGBM to regress and forcast.
    It's the most basic model.
    """

    def __init__(self, params):
        self.params = params
        self.load_path = './Train/train_extra_data.csv'
        self.forcast_path = './Forcast/evaluation_pulic.csv'
        self.save_path = './Results/lgb-unknown.csv'
        self.raw_data = None
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None


    def load_data(self):
        """
        Load data.
        """
        self.raw_data = pd.load_csv(self.load_path)


    def gen_feat(self):
        """
        Generate features.
        """
        pass


    def train(self):
        """
        Start training.
        """
        pass


    def forcast(self):
        """
        Forcast salesVolume in 2018.1 ~ 2018.4.
        """
        pass


    def save_data(self):
        """
        Save the predicted data.
        """
        pass


if __name__ == "__main__":
    params = {}
    model = LGBModel(params)
    model.load_data()
    model.gen_feat()
    model.train()
    model.save_data()