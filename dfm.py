import numpy as np
import tensorflow as tf
from dataparser import *
import os

# Comment this line to use gpu.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class DFM:
    def __init__(self, feature_size, field_size, 
                 embedding_size=40, dropout_fm=[1.0, 1.0],
                 deep_layers=[32, 32], dropout_deep=[0.5, 0.5, 0.5], 
                 deep_layers_activation=tf.nn.relu,
                 epochs=100, batch_size=1000, 
                 learning_rate=0.1,
                 use_fm=True, use_deep=True, l2_reg=0.01):
        assert use_fm or use_deep, 'At least one of use_fm and use_deep should be True.'

        self.feature_size = feature_size        # number of all features, denoted as M
        self.field_size = field_size            # size of feature feilds, denoted as F
        self.embedding_size = embedding_size    # denoted as K

        self.dropout_fm = dropout_fm
        self.deep_layers = deep_layers
        self.dropout_deep = dropout_deep
        self.deep_layers_activation = deep_layers_activation
        self.use_fm = use_fm
        self.use_deep = use_deep
        self.l2_reg = l2_reg
        self.learning_rate = learning_rate

        self.epochs = epochs
        self.batch_size = batch_size
        self.train_result, self.valid_result = [], []

        self._init_graph()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())


    def _init_graph(self):
        self.feat_index = tf.placeholder(tf.int32, [None, None], name='feat_index')
        self.feat_value = tf.placeholder(tf.float32, [None, None], name='feat_value')
        self.label = tf.placeholder(tf.float32, [None, 1], name='label')
        self.dropout_keep_fm = tf.placeholder(tf.float32, [None], name='dropout_keep_fm')
        self.dropout_keep_deep = tf.placeholder(tf.float32, [None], name='dropout_keep_deep')
        self.train_phase = tf.placeholder(tf.bool, [], name='train_phase')
        self.weights = self._initialize_weights()

        # 'feature_embeddings' is M * K
        self.embeddings = tf.nn.embedding_lookup(self.weights['feature_embeddings'], self.feat_index)    # None * F * K
        feat_value = tf.reshape(self.feat_value, [-1, self.field_size, 1])  # None * F * 1
        self.embeddings = tf.multiply(self.embeddings, feat_value)          # None * F * K

        # --------------- first order term --------------
        self.y_first_order = tf.nn.embedding_lookup(self.weights['feature_bias'], self.feat_index)  # None * F * 1
        self.y_first_order = tf.reduce_sum(tf.multiply(self.y_first_order, feat_value), -1)         # None * F
        self.y_first_order = tf.nn.dropout(self.y_first_order, self.dropout_keep_fm[0])             # None * F
        
        # --------------- second order term -------------
        # sum_square part
        self.summed_features_emb = tf.reduce_sum(self.embeddings, 1)            # None * K
        self.summed_features_emb_square = tf.square(self.summed_features_emb)   # None * K
        # spuare_sum part
        self.squared_features_emb = tf.square(self.embeddings)  # None * F * K
        self.squared_sum_features_emb = tf.reduce_sum(self.squared_features_emb, 1) # None * K
        # second order
        self.y_second_order = 0.5*tf.subtract(self.summed_features_emb_square, self.squared_sum_features_emb) # None * K
        self.y_second_order = tf.nn.dropout(self.y_second_order, self.dropout_keep_fm[1])
        
        # --------------- deep componet -----------------
        self.y_deep = tf.reshape(self.embeddings, shape=[-1, self.field_size * self.embedding_size])    # None * (F*K)
        self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[0])
        for i in range(0, len(self.deep_layers)):
            self.y_deep = tf.add(tf.matmul(self.y_deep, self.weights['layer_%d'%i]), self.weights['bias_%d'%i]) # None * layer[i] * 1
            self.y_deep = self.deep_layers_activation(self.y_deep)
            self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[i + 1])
        
        # --------------- DeepFM ------------------------
        if self.use_fm and self.use_deep:
            concat_input = tf.concat([self.y_first_order, self.y_second_order, self.y_deep], -1)
        elif self.use_fm:
            concat_input = tf.concat([self.y_first_order, self.y_second_order], -1)
        elif self.use_deep:
            concat_input = self.y_deep
        self.out = tf.add(tf.matmul(concat_input, self.weights['concat_projection']), self.weights['concat_bias'])
        
        # loss
        self.loss = tf.nn.l2_loss(tf.subtract(self.label, self.out))
        self.rmse = tf.reduce_mean((self.label - self.out)**2)**0.5
        # l2 regulation on weights
        if self.l2_reg > 0:
            self.loss += tf.contrib.layers.l2_regularizer(
                self.l2_reg)(self.weights['concat_projection'])
            if self.use_deep:
                for i in range(len(self.deep_layers)):
                    self.loss += tf.contrib.layers.l2_regularizer(
                        self.l2_reg)(self.weights['layer_%d'%i])
        # train operation
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                epsilon=1e-8).minimize(self.loss)


    def _initialize_weights(self):
        weights = dict()

        # embeddings
        weights['feature_embeddings'] = tf.Variable(
            tf.random_normal([self.feature_size, self.embedding_size], 0.0, 0.01),
            name='feature_embeddings'
        )
        weights['feature_bias'] = tf.Variable(
            tf.random_uniform([self.feature_size, 1], 0.0, 1.0), name='feature_bias'
        )

        # deep layers
        num_layer = len(self.deep_layers)
        input_size = self.field_size *  self.embedding_size
        glorot = np.sqrt(2.0 / (input_size + self.deep_layers[0]))
        weights['layer_0'] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(input_size, self.deep_layers[0])), dtype=np.float32
        )
        weights['bias_0'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[0])), 
                                                            dtype=np.float32)
        
        for i in range(1, num_layer):
            glorot = np.sqrt(2.0 / (self.deep_layers[i - 1] + self.deep_layers[i]))
            weights['layer_%d'%i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[i - 1], self.deep_layers[i])),
                dtype=np.float32
            )
            weights['bias_%d'%i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[i])),
                dtype=np.float32
            )

        # final concat projection layer
        if self.use_fm and self.use_deep:
            input_size = self.field_size + self.embedding_size + self.deep_layers[-1]
        elif self.use_fm:
            input_size = self.field_size + self.embedding_size
        elif self.use_deep:
            input_size = self.deep_layers[-1]
        glorot = np.sqrt(2.0 / (input_size + 1))
        weights['concat_projection'] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(input_size, 1)),
            dtype=np.float32
        )
        weights['concat_bias'] = tf.Variable(tf.constant(0.01), dtype=np.float32)

        return weights

    
    def train(self, Xi, Xv, Y):
        for epoch in range(self.epochs):
            permu = np.random.permutation(len(Xi))
            Xi = Xi[permu]
            Xv = Xv[permu]
            Y = Y[permu]

            for i in range(0, len(Xi), self.batch_size):
                feed_dict = {
                    self.feat_index: Xi, 
                    self.feat_value: Xv, 
                    self.label: np.reshape(Y, [-1, 1]),
                    self.dropout_keep_fm: self.dropout_fm,
                    self.dropout_keep_deep: self.dropout_deep}
                _, error = self.sess.run([self.train_op, self.rmse], feed_dict)
            print('episode %d, rmse: %.3f'%(epoch, error))
            

    def eval(self, Xi, Xv, Y):
        assert len(Xi) == len(Xv) == len(Y)
        feed_dict = {
            self.feat_index: Xi,
            self.feat_value: Xv,
            self.label: np.reshape(Y, [-1, 1]),
            self.dropout_keep_fm: np.ones([len(self.dropout_fm)]),
            self.dropout_keep_deep: np.ones([len(self.dropout_deep)])
        }
        error = self.sess.run(self.rmse, feed_dict)
        return error

    
    def get_score(self, test_data_list):
        nrmse = []
        for Xi, Xv, Y in test_data_list:
            if len(Xi) != 0:
                error = self.eval(Xi, Xv, Y)
                nrmse.append(error/np.mean(Y))
        print("Your score is %.3f!"%(1 - np.mean(nrmse)))



if __name__ == '__main__':
    dp = DataParser('Train/train_sales_data.csv', 
        ignore_cols=['province'], label_name='salesVolume', dense=False)
    dp.gen_feat_dict()
    dp.gen_vectors()
    Xi, Xv, Y, _, _, _ = dp.gen_train_test()
    sep_test_data = dp.gen_fine_grained_test(partial_cols=['adcode', 'model'])

    fm = DFM(dp.feat_dim, dp.field_dim)
    fm.train(Xi, Xv, Y)
    fm.get_score(sep_test_data)