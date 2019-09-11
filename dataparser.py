import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix

class DataParser:
    """
    This class pre-processes the original row data. It mainly has the following functions:
    1)  Reads the .csv file and returns the vectorized representation of all data, 
        as well as some statistic features of the data.

    2)  Divides the whole data set into training set and testing set.
        The training and testing set will be identically destributed.
    """


    def __init__(self, filename, numeric_cols=[], ignore_cols=[], dense=False, label_name=None):
        """
        : param filename:       The file to be processed.
        : param numeric_cols:   A list of names of numeric features. Other features will be treated as catalogic.
        : param ignore_cols:    A list of names of features to be ignored.
        : param dense:          Determine generating a dense matrix or sparse a one.
        : param label_name:     If the label column is in the file, specify it.
        """
        self.filename = filename
        self.numeric_cols = numeric_cols
        self.ignore_cols = ignore_cols
        self.dense = dense
        self.label_name = label_name

        self.record_num = 0
        self.feat_dict = None   # feature dictionary as discribed below
        self.feat_dim = 0       # dimention of the embedded vector representation
        self.field_dim = 0      # number of feature filed
                                # for example, 'province' is a feature field and 'shanghai' is a feature
        self.sparse_data = None
        self.dense_data = None
        self.labels = None


    def gen_feat_dict(self):
        """
        Generates a feature dictionary mapping each feature into a position in the vector.
        Used for one-hot embedding.

        : return: A dictionary:
            {catalogic_feature1:{value1:position1, ...}, ..., numeric_feature1:position_n, ...}
        """
        self.feat_dict = {}
        df = pd.read_csv(self.filename)
        feat_count = 0
        field_count = 0

        for col in df.columns:
            if col in self.ignore_cols or col == self.label_name:
                continue

            # Arrange a single position for numeric features
            if col in self.numeric_cols:
                self.feat_dict[col] = feat_count
                feat_count += 1
                field_count += 1
            # Arrange a series of positions for catalogic features, which will be encoded as one-hot
            else:
                values = df[col].unique()
                self.feat_dict[col] = dict(zip(values, range(feat_count, feat_count + len(values))))
                feat_count += len(values)
                field_count += 1

        self.feat_dim = feat_count
        self.field_dim = field_count
        return self.feat_dict
        

    def gen_vectors(self):
        """
        Generate vector representation of .csv files.
        A feature dict must be generated before.

        : return: if dense == True:
                    A matrix concated with one-hot vector of catalogic features 
                    and numeric vector of numeric features.
                  else:
                    (Xi, Xv)
                    Xi is a 2d array of feature indices of each sample in the dataset.
                    Xv is a 2d array of feature values of each sample in the dataset.
        """
        assert not self.feat_dict == None, 'A feature dict must be generated before using this function.'

        dfi = pd.read_csv(self.filename)
        if self.label_name is not None:
            y = dfi[self.label_name].values
            self.labels = y
        
        dfv = dfi.copy()
        for col in dfi.columns:
            if col in self.ignore_cols or col == self.label_name:
                dfi.drop(col, axis=1, inplace=True)
                dfv.drop(col, axis=1, inplace=True)
                continue
            if col in self.numeric_cols:
                dfi[col] = self.feat_dict[col]
            else:
                dfi[col] = dfi[col].map(self.feat_dict[col])
                dfv[col] = 1
            
        # 2d array of feature indices of each sample in the dataset
        Xi = dfi.values
        # 2d array of feature values of each sample in the dataset
        Xv = dfv.values
        self.sparse_data = (Xi, Xv)
        self.record_num = len(Xi)
        
        if self.dense:
            col_ix = np.reshape(Xi, [-1])
            row_ix = np.repeat(np.arange(0, self.record_num), self.field_dim)
            data = np.reshape(Xv, [-1])
            self.dense_data = coo_matrix((data, (row_ix, col_ix)), [self.record_num, self.feat_dim]).toarray()
            if self.label_name != None:
                return self.dense_data, self.labels
            else:
                return self.dense_data
        else:
            if self.label_name != None:
                return Xi, Xv, y
            else:
                return Xi, Xv

    
    def gen_train_test(self, train_ratio=0.95):
        """
        Generate itentically distributed training set and testing set,
        for the fucking competition doesn't offer a testing set.
        """
        assert not self.feat_dict == None, 'A feature dictionary must be generated before using this function.'
        assert not self.sparse_data == None, 'gen_vectors() must be called before using this function.'

        train_num = int(self.record_num * train_ratio)
        # Create a random permutation, and re-arrange the data with same order.
        permu = np.random.permutation(self.record_num)
        self.sparse_data = (self.sparse_data[0][permu], self.sparse_data[1][permu])
        self.labels = self.labels[permu]

        self.Xi_train, self.Xv_train, self.y_train = \
            self.sparse_data[0][:train_num], self.sparse_data[1][:train_num], self.labels[:train_num]
        self.Xi_test, self.Xv_test, self.y_test = \
            self.sparse_data[0][train_num:], self.sparse_data[1][train_num:], self.labels[train_num:]

        if self.dense:
            self.dense_data = self.dense_data[permu]
            self.data_train, self.data_test = self.dense_data[:train_num], self.dense_data[train_num:]
            return self.data_train, self.y_train, self.data_test, self.y_test
        
        return self.Xi_train, self.Xv_train, self.y_train, self.Xi_test, self.Xv_test, self.y_test