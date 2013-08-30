class baseDist:
    """An abstract class for base distribution"""
    def __init__(self):
        assert(False)   #instance of baseDist can not be made

    def sample(self):
        """returns a instance of sampledDist"""
        pass

    def Zpost(self, observation):
        pass

class sampledDist:
    pass
