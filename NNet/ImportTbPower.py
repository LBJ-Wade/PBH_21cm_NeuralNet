import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler

class ImportGraph():
    def __init__(self, loc, mpbh, zfix):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            saver = tf.train.import_meta_graph(loc + '.meta')
            saver.restore(self.sess, loc)
            self.activation = tf.get_collection('activation')[0]
        self.mpbh = mpbh
        self.zfix = zfix

        self.scalar = StandardScaler()
        fileNd = '../TbFiles/TbFull_Power_Mpbh_{:.0e}_Zval_{:.2f}.dat'.format(self.mpbh, self.zfix)
        tbVals = np.loadtxt(fileNd)
        data = tbVals[:, :6]
        dataSTD = self.scalar.fit_transform(data)
        return
    
    def run_yhat(self, data):
        inputV = np.insert(self.scalar.transform(data), 0, 1., axis=1)
        return np.power(10., self.sess.run(self.activation, feed_dict={"X:0": inputV}))
