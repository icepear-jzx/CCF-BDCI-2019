from dataparser import *
import tensorflow as tf
import numpy as np


class MLP:
    def __init__(self, input_dim, output_dim=1, layers=[128, 256, 64, 32], 
                 act=tf.nn.relu, drop=True, drop_keep=[0.5]*4, learning_rate=0.01, l2_factor=0.01, 
                 batch_size=16, epochs=100, eval_interval=50):
        if drop:
            assert len(layers) == len(drop_keep), 'Drop keeping probilities don\'t match laysers number.'
 
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = layers
        self.act = act
        self.drop = drop
        self.drop_keep = drop_keep
        self.learning_rate = learning_rate
        self.l2_factor = 0.01
        self.batch_size = batch_size
        self.epochs = epochs
        self.eval_interval = eval_interval

        self.weights = self._init_weights()
        self._build_graph()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())


    def _build_graph(self):
        self.x = tf.placeholder(tf.float32, [None, self.input_dim])
        self.drop_keep_in = [tf.placeholder(tf.float32, []) for i in range(len(self.drop_keep))]
        
        h = self.x
        for i in range(len(self.layers) + 1):
            h = self.act(tf.add(tf.matmul(h, self.weights['weight_%d'%(i)]), self.weights['bias_%d'%i]))
            if self.drop and i >= 1:
                h = tf.nn.dropout(h, keep_prob=self.drop_keep_in[i - 1])
        self.y_pred = h

        self.y_real = tf.placeholder(tf.float32, [None])
        self.l2_loss = sum([tf.nn.l2_loss(self.weights['weight_%d'%(i)]) for i in range(len(self.layers) + 1)])
        self.target_loss = tf.reduce_mean((self.y_pred - self.y_real)**2)
        self.loss = self.l2_factor*self.l2_loss + self.target_loss
        self.rmse = self.target_loss**0.5
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)    


    def _init_weights(self):
        weights = {}
        layers = [self.input_dim] + self.layers + [1]
        for i in range(0, len(layers) - 1):
            glorot = np.sqrt(2.0 / (layers[i] + layers[i + 1]))
            weights['weight_%d'%(i)] = \
                tf.Variable(tf.truncated_normal([layers[i], layers[i + 1]], stddev=glorot))
            weights['bias_%d'%(i)] = \
                tf.Variable(tf.zeros([layers[i + 1]]))
        
        return weights






if __name__ == '__main__':
    dp = DataParser('Train/train_sales_data.csv', 
        ignore_cols=['province'], label_name='salesVolume', dense=True)
    dp.gen_feat_dict()
    dp.gen_vectors()
    data_train, y_train, data_test, y_test = dp.gen_train_test()
    sep_test_data = dp.gen_fine_grained_test(partial_cols=['adcode', 'model'])

    mlp = MLP(dp.feat_dim)
    mlp.train(data_train, y_train)
    mlp.get_score(sep_test_data)