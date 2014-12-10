import gzip, cPickle
import time
import os

def pickle(filename, object):
    with gzip.open(filename, 'wb') as f:
        cPickle.dump(object, f)

def unpickle(filename):
    if filename[-3:] == '.gz':
        with gzip.open(filename, 'rb') as f:
            return (cPickle.load(f))
    else:
        with open(filename, 'rb') as f:        
            return cPickle.load(f)

def static_var(varname, value):
    def decorate(func):
        setattr(func, varname, value)
        return func
    return decorate

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

class inside():
    def __init__(self, location):
        self.original_dir = os.path.realpath(os.curdir)
        self.location = location
    def __enter__(self):
        if not os.path.isdir(self.location):
            os.makedirs(self.location)
        os.chdir(self.location)        
    def __exit__(self, type, value, traceback):    
        os.chdir(self.original_dir) 

