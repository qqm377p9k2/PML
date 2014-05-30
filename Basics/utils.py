import gzip, cPickle

def pickle(filename, object):
    with gzip.open(filename, 'wb') as f:
        cPickle.dump(object, f)

def unpickle(filename):
    with gzip.open(filename, 'rb') as f:
        return (cPickle.load(f))

