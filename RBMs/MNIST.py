import cPickle, gzip, numpy
from Data import *

def data(batch_size=None):
    f = gzip.open('../data/mnist.pkl.gz', 'rb')
    training, valid, test = cPickle.load(f)
    f.close()
    training = {'data': training[0], 'labels': training[1]}
    valid =    {'data': valid[0],    'labels': valid[1]}
    test  =    {'data': test[0],     'labels': test[1]}
    data = DataSet(training = Data(data  = training['data'], 
                                   labels= training['labels'], 
                                   batch_size = batch_size),
                   test     = Data(data  = test['data'], 
                                   labels= test['labels'],
                                   batch_size = batch_size),
                   valid    = Data(data  = valid['data'], 
                                   labels= valid['labels'],
                                   batch_size = batch_size))

    return data

def main():
    dat = data()
    print [d[0].shape for d in dat]

if __name__ == "__main__":
    main()
