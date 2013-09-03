import math
import numpy as np

ispv = lambda x: math.fabs(np.sum(x)-1)<1e-10

class baseDist(object):
    """An abstract class for base distribution"""
    def __init__(self):
        raise NotImplementedError( "Should have implemented this" )

    def samplePost(self,obs):
        raise NotImplementedError( "Should have implemented this" )

    def Zpost(self, obs):
        raise NotImplementedError( "Should have implemented this" )

    class lFunSet(object):
        def __init__(self, dist, size):
            raise NotImplementedError( "Should have implemented this" )

        def countUp(self, tableIdx):
            assert(tableIdx in range(self.__pointer))
            self.__counter[tableIdx] += 1

        def counter(self):
            return self.__counter[:self.__pointer]
            
        def length(self):
            return self.__pointer

        def compute(self, obs):
            raise NotImplementedError( "Should have implemented this" )

        def theta(self, table):
            raise NotImplementedError( "Should have implemented this" )
