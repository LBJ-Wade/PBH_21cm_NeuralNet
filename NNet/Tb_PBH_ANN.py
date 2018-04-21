import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import itertools
import copy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)

class Tb_PBH_Nnet(object):
    def __init__(self, mPBH, globalTb=True, HiddenNodes=25, epochs=10000, zfix=17.57):
        self.mPBH = mPBH
        self.globalTb=globalTb
        self.N_EPOCHS = epochs
        self.zfix = zfix
        self.h_size = HiddenNodes
        if self.globalTb:
            self.grad_stepsize = 1e-5
            self.errThresh = 10
            self.dirName = 'MetaGraphs/Tb_PBH_Mass_{:.0e}_Global'.format(self.mPBH)
            self.fileN = self.dirName + '/PBH21cm_Graph_Global_Mpbh_{:.0e}'.format(self.mPBH)
        else:
            self.grad_stepsize = 1e-7
            self.errThresh = 1.
            self.dirName = 'MetaGraphs/Tb_PBH_Mass_{:.0e}_Power_Zval_{:.2f}'.format(self.mPBH, self.zfix)
            self.fileN = self.dirName + '/PBH21cm_Graph_Power_Mpbh_{:.0e}_Zval_{:.2f}'.format(self.mPBH, self.zfix)

        if not os.path.exists(self.dirName):
            os.mkdir(self.dirName)
    
    def init_weights(self, shape):
        """ Weight initialization """
        weights = tf.random_normal(shape, stddev=0.5)
        return tf.Variable(weights)

    def forwardprop(self, X, w_1, w_2, w_3):
        """
        Forward-propagation.
        """
        hid1 = tf.nn.sigmoid(tf.matmul(X, w_1))
        hid2 = tf.nn.sigmoid(tf.matmul(hid1, w_2))
        yhat = tf.matmul(hid2, w_3)
        return yhat

    def get_data(self, frac_test=0.25):
        self.scalar = StandardScaler()
        if self.globalTb:
            fileNd = '../TbFiles/TbFull_Mpbh_{:.0e}.dat'.format(self.mPBH)
            inputN = 6
        else:
            fileNd = '../TbFiles/TbFull_Power_Mpbh_{:.0e}_Zval_{:.2f}.dat'.format(self.mPBH, self.zfix)
            inputN = 6

        tbVals = np.loadtxt(fileNd)
        np.random.shuffle(tbVals)
        data = tbVals[:, :inputN]
        target = tbVals[:, inputN:]
        dataSTD = self.scalar.fit_transform(data)
    
        self.train_size = (1.-frac_test)*len(tbVals[:,0])
        self.test_size = frac_test*len(tbVals[:,0])
        # Prepend the column of 1s for bias
        N, M  = data.shape
        all_X = np.ones((N, M + 1))
        all_X[:, 1:] = dataSTD
        
        return train_test_split(all_X, target, test_size=frac_test, random_state=RANDOM_SEED)

    def main_nnet(self):#, train_nnet=True, eval_nnet=False, evalVec=[], keep_training=False):
        
        self.train_X, self.test_X, self.train_y, self.test_y = self.get_data()
        self.err_train = np.zeros_like(self.train_y)
        self.err_test = np.zeros_like(self.test_y)
        for i in range(len(self.err_train)):
            if self.train_y[i] < self.errThresh:
                self.err_train[i] = self.errThresh
            else:
                self.err_train[i] = self.train_y[i]
        for i in range(len(self.err_test)):
            if self.test_y[i] < self.errThresh:
                self.err_test[i] = self.errThresh
            else:
                self.err_test[i] = self.test_y[i]

        # Layer's sizes
        self.x_size = self.train_X.shape[1]   # Number of input nodes: [z, k?, fpbh, zeta_UV, zetaX, tmin, nalpha]
        self.y_size = self.train_y.shape[1]   # Value of Tb

        # Symbols
        self.X = tf.placeholder("float", shape=[None, self.x_size])
        self.y = tf.placeholder("float", shape=[None, self.y_size])

        # Weight initializations
        w_1 = self.init_weights((self.x_size, self.h_size))
        w_2 = self.init_weights((self.h_size, self.h_size))
        w_3 = self.init_weights((self.h_size, self.y_size))

        # Forward propagation
        self.yhat = self.forwardprop(self.X, w_1, w_2, w_3)
        

        # Backward propagation
        self.cost = tf.reduce_sum(tf.square((self.y - self.yhat), name="cost"))
        self.updates = tf.train.GradientDescentOptimizer(self.grad_stepsize).minimize(self.cost)
        
        # Error Check
        self.perr_train = tf.reduce_sum(tf.abs((self.y - self.yhat)/self.err_train))
        self.perr_test = tf.reduce_sum(tf.abs((self.y - self.yhat)/self.err_test))

        self.saveNN = tf.train.Saver()

        return

    def train_NN(self, evalVec, keep_training=False):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            if keep_training:
                self.saveNN.restore(sess, self.fileN)
                print 'Model Restored.'
            BATCH_SIZE = 20
            train_count = len(self.train_X)
            for i in range(1, self.N_EPOCHS + 1):
                for start, end in zip(range(0, train_count, BATCH_SIZE),
                                      range(BATCH_SIZE, train_count + 1,BATCH_SIZE)):
                    sess.run(self.updates, feed_dict={self.X: self.train_X[start:end],
                                                      self.y: self.train_y[start:end]})

                if i % 100 == 0:
                    train_accuracy = sess.run(self.perr_train, feed_dict={self.X: self.train_X, self.y: self.train_y})
                    test_accuracy = sess.run(self.perr_test, feed_dict={self.X: self.test_X, self.y: self.test_y})
                    print("Epoch = %d, train accuracy = %.7e, test accuracy = %.7e"
                          % (i + 1, train_accuracy/len(self.train_X), test_accuracy/len(self.test_X)))
                    
                    predictions = sess.run(self.yhat, feed_dict={self.X: np.insert(self.scalar.transform(evalVec), 0, 1., axis=1)})
                    print 'Current Predictions: ', predictions
            self.saveNN.save(sess, self.fileN)
        return

    def eval_NN(self, evalVec):
        with tf.Session() as sess:
            #sess.run(tf.global_variables_initializer())
            saverMeta = tf.train.import_meta_graph(self.fileN + '.meta')
            self.saveNN.restore(sess, self.fileN)
            predictions = sess.run(self.yhat, feed_dict={self.X: np.insert(self.scalar.transform(evalVec), 0, 1., axis=1)})
        return predictions




