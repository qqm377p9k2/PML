class baseDist:
    """An abstract class for base distribution"""
    def __init__(self):
        assert(False)   #instance of baseDist can not be made

    def sample(self):
        pass

    def Zpost(self, observation):
        pass
