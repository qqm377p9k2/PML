import scipy as sp
import scipy.sparse as spsp
from Basics.numpy_ import*


nwords = 61188

def readData(filename, shape):
    ndocs, nwords = shape
    i = []
    j = []
    cnts=[]
    with open(filename) as f:
        for l in f:
            ii,jj,cnt = [int(str) for str in l.strip().split(' ')]
            i.append(ii-1)
            j.append(jj-1)
            cnts.append(cnt)
    matrix = spsp.coo_matrix((cnts, (i, j)), shape=(ndocs, nwords))
    return asarray(matrix.todense())

def readLabels(filename, ndocs):
    labels=[]
    with open(filename) as f:
        for l in f:
            labels.append(int(l.strip()))
    return asarray(labels)


def readMaps(filename, ncategories=20):
    categories = ['']
    revcategories = {}
    
    with open(filename) as f:
        for i, l in enumerate(f):
            label, idx = l.strip().split(' ')
            idx = int(idx)
            assert(idx==(i+1))
            categories.append(label)
            revcategories[label] = idx
    return categories, revcategories

for name, ndocs in ('train', 11269), ('test', 7505):
    data = readData('matlab/'+name+'.data', (ndocs, nwords))
    label= readLabels('matlab/'+name+'.label', ndocs)
    maps = readMaps('matlab/'+name+'.map')
    print data
    print data.shape
    print label
    print label.shape
    print maps[0]
    print maps[1]
        


