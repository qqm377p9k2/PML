import matplotlib.pyplot as plt
import __builtin__ as base
from numpy import *
from numpy.random import rand, randn, permutation, randint

class agent(object):
    def __init__(self, environment):
        assert(isinstance(environment, gridworld))
        self.environment = environment
        self.__init_Qmap()
        self.epsilon = 0.1
        self.alpha = 0.1
        self.counter = 0

    def __init_Qmap(self):
        env = self.environment
        self.__Qmap = 0.01*zeros(env.dim() + (len(env.actionSet),))

    def initEpoch(self):
        self.environment.initEpoch()
        self.counter = 0

    def Q(self):
        return self.__Qmap.copy()

    def actions(self):
        return self.environment.actionSet.keys()

    def takeAction(self):
        self.counter += 1
        env = self.environment
        state = env.state()
        actions = self.actions()
        if rand() < self.epsilon:
            action = actions[randint(len(actions))]
        else:
            action =  actions[argmax(self.__Qmap[state])]
        return (env.transit(action),state,action)

    def updateQ(self, rwd, state, action):
        index = state + (self.actions().index(action),)
        q = self.__Qmap
        q[index] += self.alpha * (rwd - q[index])

    def train(self):
        pass

class gridworld(object):
    def __init__(self, N = 10):
        assert(N==10)
        self.__init_field(N)
        self.actionSet = {'up':    lambda x: (x[0]-1, x[1]), 
                          'down':  lambda x: (x[0]+1, x[1]),
                          'left':  lambda x: (x[0],   x[1]-1), 
                          'right': lambda x: (x[0],   x[1]+1)}
        self.__state = None

    def field_(self):
        return self.__field

    def state(self):
        return self.__state

    def field(self):
        tmp = self.__field.copy()
        if self.__state != None:
            tmp[self.__state] = 7
        return tmp

    def dim(self):
        return shape(self.__field)

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

    def status(self):
        return {0:'Floor', 1:'Wall', 2:'Goal'}.get(self.__field[self.__state])

    def availablePlaces(self):
        dim = self.dim()
        return [(r,c)
                for r in range(dim[0])
                for c in range(dim[1])
                if self.__field[r,c] == 0]

    def state(self):
       return self.__state

    def initEpoch(self):
        candidate = self.availablePlaces()
        self.__state = candidate[randint(len(candidate))]
        return self
        
    def transit(self, action):
        tmp = self.actionSet.get(action)(self.__state)
        if self.isAvailable(tmp):
            self.__state = tmp
            if self.__field[tmp] == 7:
                return 100
            else:
                return -1
        else:
            return -10


def test():
    gw = gridworld()
    a = agent(gw)
    gw.initEpoch()
    print(gw.field())
    print(gw.state())
    #print(gw.transit('up'))
    print(a.takeAction())
    print(gw.state())
    print(gw.field())
    print(gw.availablePlaces())
        
def main():
    gw = gridworld()
    a = agent(gw)

    for epoch in range(20):
        a.initEpoch()
        while True:
            rwd, stat, act = a.takeAction()
            a.updateQ(rwd, stat, act)
            if gw.status() == 'Goal':
                break
            if mod(a.counter, 10)==0:
                print(gw.state())
                print(gw.field())
        print('Finished')
        print(a.counter)
        print(gw.state())
        print(gw.field())
        Q = transpose(a.Q(), (2,0,1))
        for i in range(4):
            plt.subplot(2,2,i)
            plt.imshow(Q[i], interpolation='nearest')
            plt.title(a.actions()[i])
            plt.colorbar()
        plt.show()

if __name__=="__main__":
    main()
