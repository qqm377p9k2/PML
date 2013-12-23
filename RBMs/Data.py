class Data(object):
    def __init__(self,data, labels=None):
        self.data = data
        self.labels = labels

class DataSet(object):
    def __init__(self,training, test=None, valid=None):
        self.training = training
        self.test = test
        self.valid = valid

