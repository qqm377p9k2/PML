from numpy import *

class Data(object):
    def __init__(self,data, labels=None, batch_size=None):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
    def batches(self):
        datasz = self.data.shape[0]
        assert(mod(datasz ,self.batch_size)==0)
        n_batches = datasz/self.batch_size
        idcs = random.permutation(datasz).reshape((n_batches, self.batch_size))
        for bid in xrange(n_batches):
            yield self.data[idcs[bid]]

class DataSet(object):
    def __init__(self,training, test=None, valid=None):
        self.training = training
        self.test = test
        self.valid = valid

