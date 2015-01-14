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

class messages(object):
    def __init__(self, message='Starting...', verbose=True):
        self.verbose = verbose
        self.message = message
    def __enter__(self):
        if self.verbose:
            print self.message
    def __exit__(self, type, value, traceback):    
        if self.verbose:
            print 'Done'

class measuring_speed(messages):
    def __init__(self, starting_message='Starting computation...', ending_message='Done in',
                 unit='sec', verbose=True):
        super(measuring_speed, self).__init__(starting_message, verbose)
        self.ending_message = ending_message
        self.obuffer = zeros(1)
        self.norm = 1.0
        self.unit = unit
        if unit == 'min':
            self.norm = 60.0
        elif unit == 'hour':
            self.norm = 60.0**2
    def __enter__(self):
        super(measuring_speed, self).__enter__()
        self.timer = time.time()
        return self.obuffer
    def __exit__(self, type, value, traceback):
        self.timer = time.time() - self.timer
        self.obuffer[0] = self.timer
        if self.verbose:
            print self.ending_message +  ('%g %ss'(self.timer/self.norm, self.unit))
    def __repr__(self):
        return 'Duration :%g [%ss]'%(self.timer/self.norm, self.unit)
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

