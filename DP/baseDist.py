import math
import numpy as np

ispv = lambda x: math.fabs(np.sum(x)-1)<1e-10

class baseDist:
    """An abstract class for base distribution"""
    def __init__(self):
        assert(False)   #instance of baseDist can not be made

    def sample(self):
        """returns a instance of sampledDist"""
        pass

    def Zpost(self, observation):
        pass

    class lFunSet:
        def countUp(self, tableIdx):
            assert(tableIdx in range(self.__pointer))
            self.__counter[tableIdx] += 1

        def counter(self):
            return self.__counter[:self.__pointer]
            
        def length(self):
            return self.__pointer

