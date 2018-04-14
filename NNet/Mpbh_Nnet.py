import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import itertools
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)

class PBH_Nnet(object):
    def __init__(self, mPBH, globalTb=True):
        self.mPBH = mPBH
        self.h_size = 30               # Number of hidden nodes
        self.grad_stepsize = 1e-6
        self.errThresh = 10
        self.N_EPOCHS = 15000
        self.globalTb=globalTb
        
        self.saver_loaded = False
        
        if self.globalTb:
            self.dirName = 'MetaGraphs/PBH_Mass_{:.0e}_Global'.format(self.mPBH)
            self.fileN = self.dirName + '/PBH21cm_Graph_Global_Mpbh_{:.0e}'.format(self.mPBH)
        else:
            self.dirName = 'MetaGraphs/PBH_Mass_{:.0e}_Power'.format(self.mPBH)
            self.fileN = self.dirName + '/PBH21cm_Graph_Power_Mpbh_{:.0e}'.format(self.mPBH)
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
            fileNd = '../TbFiles/TbFull_Power_Mpbh_{:.0e}.dat'.format(self.mPBH)
            inputN = 7
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
        # Use reduced set for error calculation
        self.train_reduced_x = self.train_X[list(itertools.chain(*np.abs(self.train_y) > self.errThresh))]
        self.train_reduced_y = self.train_y[list(itertools.chain(*np.abs(self.train_y) > self.errThresh))]
        self.test_reduced_x = self.test_X[list(itertools.chain(*np.abs(self.test_y) > self.errThresh))]
        self.test_reduced_y = self.test_y[list(itertools.chain(*np.abs(self.test_y) > self.errThresh))]

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
        self.perr = tf.reduce_sum(tf.abs((self.y - self.yhat)/self.y))
        
        if not self.saver_loaded:
            self.saveNN = tf.train.Saver()
            self.saver_loaded = True
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
                    train_accuracy = sess.run(self.perr, feed_dict={self.X: self.train_reduced_x, self.y: self.train_reduced_y})
                    test_accuracy = sess.run(self.perr, feed_dict={self.X: self.test_reduced_x, self.y: self.test_reduced_y})
                    print("Epoch = %d, train accuracy = %.7e, test accuracy = %.7e"
                          % (i + 1, train_accuracy/len(self.train_reduced_x), test_accuracy/len(self.test_reduced_x)))
                    
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




