import gzip, cPickle
import time
import os
from numpy_ import *

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

def mesh(ranges, N=100.):
    return asarray([arange(min, max, (max-min)/N) for min, max in ranges])

class measuring_speed(object):
    def __init__(self, message=None, mode='verbose'):
        self.verbose = (mode == 'verbose')
        if message is None:
            message = 'Starting computation...'
        self.message = message
    def __enter__(self):
        if self.verbose:
            print self.message
        self.timer = time.time()
        return self
    def __exit__(self, type, value, traceback):
        self.timer = time.time() - self.timer
        if self.verbose:
            print 'Done in %g secs'%self.timer
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

def load_if_exists_otherwise(location, generator):
    if os.path.exists(location):
        obj = unpickle(location)
    else:
        obj = generator()
        print 'saving...'
        pickle(location, obj)
        print 'Done.'

