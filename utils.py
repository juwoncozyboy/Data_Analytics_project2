# utils.py

import os
import random
import numpy as np
import torch
import pandas as pd

    
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


class MetricTracker:
    def __init__(self, *keys):
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        self._data.total[key] += value  # `value * n` 에서 `value`로 수정
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / (self._data.counts[key] + 1e-10)

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
    