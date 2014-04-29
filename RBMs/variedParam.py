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
            if command[0] == 'linearlyDecayFor':
                value *= (1-epoch/float(command[1]))
                value = 0 if value<0 else value
            if command[0] == 'exponentiallyDecayToAValueFor':
                rate = (float(command[1])/value)**(epoch/float(command[2]))
                value *= rate
        return value
        
def test():
    for ls in [variedParam(0.1),
               variedParam(0.1, schedule=[['switchToAValueAt', 5, 0.25]]),
               variedParam(0.1, schedule=[['linearlyDecayFor', 20]]),
               variedParam(0.1, schedule=[['exponentiallyDecayToAValueFor', 0.01, 20]])]:
        for epc in xrange(30):
            print(ls.value(epc))
        
if __name__ == "__main__":
    test()
