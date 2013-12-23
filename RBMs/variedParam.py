from numpy import *

class variedParam(object):
    def __init__(self, initialValue, schedule=[]):
        self.initialValue = initialValue
        self.schedule = schedule

    def value(self, epoch):
        value = self.initialValue
        for command in self.schedule:
            if command[0] == 'switchToAValueAt':
                if command[1] < epoch:
                    value = command[2]
        return value
        
def test():
    ls = variedParam(0.1, schedule=[['switchToAValueAt', 5, 0.25]])
    for epc in xrange(30):
        print(ls.value(epc))
    ls = variedParam(0.1)
    for epc in xrange(30):
        print(ls.value(epc))
        
if __name__ == "__main__":
    test()
