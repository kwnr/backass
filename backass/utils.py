import numpy as np

class ExponentialMovingAverage():
    def __init__(self, w) -> None:
        """
        s_0 = x_0 \n
        s_t = w*x_{t-1} + (1-w)*s_{t-1} \n
        """
        self.prev = None
        self.weight = w
    
    def update(self, val):
        if self.prev == None:
            self.prev = val
        ret = self.weight * val + (1 - self.weight) * self.prev
        self.prev = ret
        return ret
