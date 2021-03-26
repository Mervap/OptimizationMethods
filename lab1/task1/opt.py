from abc import abstractmethod
import pandas as pd


class Opt:
    def __init__(self, f, eps, bounds):
        self.eps = eps
        self.f = f
        self.left = bounds[0]
        self.right = bounds[1]
        self._log = []
        self._log_headers = ['Iter', 'l', 'r']

    @abstractmethod
    def _step(self, x1, x2):
        pass

    @abstractmethod
    def opt(self):
        pass

    def log_frame(self):
        return pd.DataFrame([[i] + self._log[i] for i in range(len(self._log))],
                            columns=self._log_headers)
