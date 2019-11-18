import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
import time

class DataParser:
    """
    This class pre-processes the original row data. It mainly has the following functions:
    1)  Reads the .csv file and returns the vectorized representation of all data, 
        as well as some statistic features of the data.

    2)  Divides the whole data set into training set and testing set.
        The training and testing set will be identically destributed.
    """


    def __init__(self, filename, pred_filename=None,
                 numeric_cols=[], ignore_cols=[], dense=False, label_name=None):
        """
        : param filename:       The file to be processed(training set).
        : param pred_filename:  The file to be predicted.
        : param numeric_cols:   A list of names of numeric features. Other features will be treated as catalogic.
        : param ignore_cols:    A list of names of features to be ignored.
        : param dense:          Determine generating a dense matrix or sparse a one.
        : param label_name:     If the label column is in the file, specify it.
        """
        self.filename = filename
        self.pred_filename = pred_filename
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
        
        # If evaluating data set is set, combine it with training set
        if self.pred_filename != None:
            pred_df = pd.read_csv(self.pred_filename)
            common_features = set(df.columns) & set(pred_df.columns)
            for col in df.columns:
                if col not in common_features:
                    df.drop(col, axis=1, inplace=True)
            for col in pred_df.columns:
                if col not in common_features:
                    pred_df.drop(col, axis=1, inplace=True)
            df = pd.concat([df, pred_df])
            
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


    def set_feat_dict(self, fd):
        self.feat_dict = fd
        

    def gen_vectors(self, filename=None):
        """
        Generate vector representation of .csv files.
        A feature dict must be generated before.

        : param filename: if filename is set, process this file;
                          else process self.filename.
        : return: if dense == True:
                    A matrix concated with one-hot vector of catalogic features 
                    and numeric vector of numeric features.
                  else:
                    (Xi, Xv)
                    Xi is a 2d array of feature indices of each sample in the dataset.
                    Xv is a 2d array of feature values of each sample in the dataset.
        """
        assert(not self.feat_dict == None, 'gen_feat_dict() or set_feat_dict() must be called before using this function.')

        if filename == None:
            dfi = pd.read_csv(self.filename)
        else:
            dfi = pd.read_csv(filename)
        if (self.label_name is not None) and (self.label_name in dfi.columns):
            y = dfi[self.label_name].values
            self.labels = y
        else:
            y = None
        
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
                return self.dense_data, y
            else:
                return self.dense_data
        else:
            if self.label_name != None:
                return Xi, Xv, y
            else:
                return Xi, Xv
                
    
    def gen_train_test_random(self, train_ratio=0.95):
        """
        Generate itentically distributed training set and testing set,
        for the fucking competition doesn't offer a testing set.
        """
        assert not self.feat_dict == None, 'gen_feat_dict() must be called before using this function.'
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


    def gen_fine_grained_test(self, partial_cols=[], all_test=False):
        """
        For every feature combination in partial_cols, generate a separate test set.
        """
        # assert self.dense, 'This function only for dense representation yet. Sparse version will be updated soon.'
        # assert not self.sparse_data == None, 'gen_test_data() must be called before using this function.'
        for col in partial_cols:
            assert not col in self.numeric_cols, 'Partial features should not be numeric.'

        if all_test:
            if not self.dense:
                self.Xi_test = self.sparse_data[0]
                self.Xv_test = self.sparse_data[1]
                self.y_test = self.labels

        data_indices = []
        def helper(vec_pos, cols):
            if len(cols) == 0:
                if self.dense:
                    indices = [self.data_test[:, vec_pos[i]] == 1 for i in range(len(vec_pos))]
                else:
                    indices = [np.array([vec_pos[i] in self.Xi_test[j] for j in range(len(self.Xi_test))])
                                                                       for i in range(len(vec_pos))]
                index = indices[0]
                for i in range(1, len(indices)):
                    index *= indices[i]
                data_indices.append(np.where(index))
                return
            
            for pos in self.feat_dict[cols[0]].values():
                helper(vec_pos + [pos], cols[1:])
        
        helper([], partial_cols)
        if self.dense:
            return [(self.data_test[index], self.y_test[index]) for index in data_indices]
        else:
            return [(self.Xi_test[index], self.Xv_test[index], self.y_test[index]) for index in data_indices]


def write_results(filename, y):
    time_str = time.strftime('%m-%d-%H:%M',time.localtime(time.time()))
    filename = '%s.csv'%(filename)

    with open('Forecast/evaluation_public.csv', 'r') as f:
        forecast_data = pd.read_csv(f)

    y = np.reshape(y, [-1])
    fr = open(filename, 'w')
    fr.write('id,forecastVolum\n')
    for i in range(len(y)):
        fr.write('%d,%d\n'%(forecast_data.iloc[i].id, int(max(y[i], 0))))
    fr.close()


def process_negative(y):
    y = np.reshape(y, [-1])
    mean = np.mean(y)
    for i in range(len(y)):
        if y[i] < 0: y[i] = mean
    return y