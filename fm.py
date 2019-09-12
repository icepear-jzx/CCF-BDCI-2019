import numpy as np
import tensorflow as tf
from dataparser import *
import os

# Comment this line to use gpu.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class FM:
    def __init__(self, input_dim, latent_dim=20, l2_factor=1, learning_rate=0.01,
                 epochs=1000, batch_size=1000):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.l2_factor = l2_factor
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

        self._build_graph()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _build_graph(self):
        self.w0 = tf.Variable(tf.truncated_normal([]))
        self.w = tf.Variable(tf.truncated_normal([1, self.input_dim]))
        self.V = tf.Variable(tf.truncated_normal([self.latent_dim, self.input_dim]))

        self.x = tf.placeholder(tf.float32, [None, self.input_dim])
        self.y = tf.placeholder(tf.float32, [None])
        linear_part = self.w0 + tf.reduce_sum(self.x * self.w, -1)
        interactive_part = 0.5*tf.reduce_sum(tf.matmul(self.x, tf.transpose(self.V))**2 - tf.matmul(self.x**2, tf.transpose(self.V)**2), 1)
        self.y_ = linear_part + interactive_part

        target_loss = tf.reduce_mean((self.y - self.y_)**2)
        l2_norm = tf.nn.l2_loss(self.w) + tf.nn.l2_loss(self.V)
        self.loss = target_loss + self.l2_factor*l2_norm
        self.rmse = target_loss**0.5

        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


    def train(self, data, labels):
        assert len(data) == len(labels)
        for epoch in range(self.epochs):
            permu = np.random.permutation(len(data))
            data = data[permu]
            labels = labels[permu]

            errors = []
            for i in range(0, len(data), self.batch_size):
                feed_dict = {
                    self.x: data[i:i + self.batch_size], 
                    self.y: labels[i:i + self.batch_size], 
                }
                _, error = self.sess.run([self.train_op, self.rmse], feed_dict)
            print('In epoch %d, rmse: %.3f'%(epoch + 1, error))


    def eval(self, data, labels):
        assert len(data) == len(labels)
        feed_dict = {
            self.x: data,
            self.y: labels
        }
        error = self.sess.run(self.rmse, feed_dict)
        return error

    
    def get_score(self, test_data_list):
        nrmse = []
        for x, y in test_data_list:
            if len(x) != 0:
                error = self.eval(x, y)
                nrmse.append(error/np.mean(y))
        print("Your score is %.3f!"%(1 - np.mean(nrmse)))


if __name__ == '__main__':
    dp = DataParser('Train/train_sales_data.csv', 
        ignore_cols=['province'], label_name='salesVolume', dense=True)
    dp.gen_feat_dict()
    dp.gen_vectors()
    data_train, y_train, data_test, y_test = dp.gen_train_test()
    sep_test_data = dp.gen_fine_grained_test(partial_cols=['adcode', 'model'])

    fm = FM(dp.feat_dim)
    fm.train(data_train, y_train)
    fm.get_score(sep_test_data)