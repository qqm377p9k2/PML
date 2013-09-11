import matplotlib.pyplot as plt
import __builtin__ as base
from numpy import *
from numpy.random import rand, randn, permutation, randint

class agent(object):
    def __init__(self,N):
        self.environment = None
        self.__init_Qmap()

    def __init_Qmap(self):
        self.__Qmap = None
        
    def takeAction(self):
        pass
        

class gridworld(object):
    def __init__(self, N = 10):
        assert(N==10)
        self.__init_field(N)
        self.actionSet = {'up':    lambda x: (x[0]-1, x[1]), 
                          'down':  lambda x: (x[0]+1, x[1]),
                          'left':  lambda x: (x[0],   x[1]-1), 
                          'right': lambda x: (x[0],   x[1]+1)}
        self.state = None

    def field_(self):
        return self.__field

    def field(self):
        foo = self.__field.copy()
        if self.state != None:
            foo[self.state] = 7
        return foo

        
    def N(self):
        return shape(self.__field)[0]
        
    def __init_field(self, N):
        self.__field = zeros((N,N), dtype=int)
        self.__field[0,:] = 1
        self.__field[-1,:] = 1
        self.__field[:,0] = 1
        self.__field[:,-1] = 1
        self.__field[1:5][:,3] = 1
        self.__field[3,6] = 2


    def isAvailable(self, place):
        return (self.__field[place] != 1)

    def availablePlaces(self):
        return [(r,c)
                for r in range(self.N())
                for c in range(self.N())
                if self.__field[r,c] == 0]

    def state(self):
       return self.state

    def initEpoch(self):
        candidate = self.availablePlaces()
        self.state = candidate[randint(len(candidate))]
        return self
        
    def transit(self, action):
        tmp = self.actionSet.get(action)(self.state)
        if self.isAvailable(tmp):
            self.state = tmp
            if self.__field[tmp] == 7:
                return 1
            else:
                return -1
        else:
            return -10


def main():
    gw = gridworld()
    gw.initEpoch()
    print(gw.field())
    print(gw.state)
    print(gw.transit('up'))
    print(gw.state)
    print(gw.field())
    print(gw.availablePlaces())
        
if __name__=="__main__":
    main()
