import numpy as np


class ReLU:
    """
    ReLU Activation Function.
    """
    
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        # return x if x > 0 else 0
        return np.maximum(0, x)

class BatchNormalize:
    """
    ReLU Activation Function.
    """
    
    def __init__(self):
        self.x = None

    def forward(self, x):
        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0)
        x = (x - mean) / (std + 1e-8)
        return x
