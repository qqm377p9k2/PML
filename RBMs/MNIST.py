import cPickle, gzip, numpy
from Data import *

def data(batch_size=None):
    f = gzip.open('../data/mnist.pkl.gz', 'rb')
    training, valid, test = cPickle.load(f)
    f.close()
    return training, valid, test

def main():
    dat = data()
    print [d[0].shape for d in dat]

if __name__ == "__main__":
    main()
