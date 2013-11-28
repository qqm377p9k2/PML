import profile
from numpy import *
from numpy.random import randn, random

M = 1000000
dim = 10

def replacingTest():
    A = randn(dim, dim)  #init A
    B = A               #B references A
    C = zeros((dim,dim))#new data
    for i in xrange(M):
        B = C           #replaces B s.t. this points to the new data
    print(A[0:2, 0:2])
    print(B[0:2, 0:2])
        

def overwritingTest():
    A = randn(dim, dim)  #init A
    B = A               #B references A
    C = zeros((dim,dim))#new data
    for i in xrange(M):
        B[:] = C        #overwites the memory space referenced by A and B
    print(A[0:2, 0:2])
    print(B[0:2, 0:2])

def main():
    profile.run('overwritingTest()')
    profile.run('replacingTest()')

if __name__ == "__main__":
    main()
