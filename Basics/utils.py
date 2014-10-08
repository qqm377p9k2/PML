import gzip, cPickle
import time

def pickle(filename, object):
    with gzip.open(filename, 'wb') as f:
        cPickle.dump(object, f)

def unpickle(filename):
    with gzip.open(filename, 'rb') as f:
        return (cPickle.load(f))

class measuring_speed(object):
    def __init__(self):
        pass
    def __enter__(self):
        self.timer = time.time()
        return self
    def __exit__(self, type, value, traceback):
        self.timer = time.time() - self.timer
    def __repr__(self):
        return 'Duration :%g [sec]'%self.timer
    def float(self):
        return self.timer
