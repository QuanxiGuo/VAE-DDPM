# utils.py

import numpy as np
import torch

class Scaler:
    def __init__(self):
        self.min = None
        self.max = None

    def fit(self, data):
        self.min = np.min(data, axis=0)
        self.max = np.max(data, axis=0)

    def transform(self, data):
        if self.min is None or self.max is None:
            raise ValueError("Scaler has not been fitted. Call .fit() first.")
        range_val = self.max - self.min
        range_val[range_val == 0] = 1e-8
        return (data - self.min) / range_val

    def inverse_transform(self, data_scaled):

        if self.min is None or self.max is None:
            raise ValueError("Scaler has not been fitted. Call .fit() first.")
        return data_scaled * (self.max - self.min) + self.min

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False