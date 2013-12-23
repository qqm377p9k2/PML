import cPickle, gzip, numpy
from Data import *

def data():
    f = gzip.open('mnist.pkl.gz', 'rb')
    training, valid, test = cPickle.load(f)
    f.close()
    training = {'data': training[0], 'labels': training[1]}
    valid =    {'data': valid[0],    'labels': valid[1]}
    test  =    {'data': test[0],     'labels': test[1]}
    data = DataSet(training = Data(data  = training['data'], 
                                   labels= training['labels']),
                   test     = Data(data  = test['data'], 
                                   labels= test['labels']),
                   valid    = Data(data  = valid['data'], 
                                   labels= valid['labels']))

    return data

def main():
    dat = data()
    print [d[0].shape for d in dat]

if __name__ == "__main__":
    main()
