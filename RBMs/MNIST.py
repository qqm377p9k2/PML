import cPickle, gzip, numpy

def data():
    f = gzip.open('mnist.pkl.gz', 'rb')
    training, valid, test = cPickle.load(f)
    f.close()
    training = {'data': training[0], 'labels': training[1]}
    valid =    {'data': valid[0],    'labels': valid[1]}
    test  =    {'data': test[0],     'labels': test[1]}
    return {'training': training, 'valid': valid, 'test': test}

def main():
    dat = data()
    print [d[0].shape for d in dat]

if __name__ == "__main__":
    main()
