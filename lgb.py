import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')


class LGBModel:
    """
    This model uses LightGBM to regress and predict.
    """

    def __init__(self, params):
        self.params = params


    def load_data(self):
        """
        Load data.
        """
        pass


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


    def predict(self):
        """
        Predict salesVolume in 2018.1 ~ 2018.4.
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