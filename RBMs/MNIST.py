import cPickle, gzip, numpy

def data():
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_set, valid_set, test_set = cPickle.load(f)
    f.close()
    return (training_set, valid_set, test_set)

def main():
    dat = data()
    print [d[0].shape for d in dat]

if __name__ == "__main__":
    main()
