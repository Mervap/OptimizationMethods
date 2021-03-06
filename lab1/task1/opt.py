from abc import abstractmethod
import pandas as pd

from lab1.watcher import Watcher


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

    def _opt_inner(self):
        left = self.left
        right = self.right
        self._log.append([left, right])
        while right - left > self.eps:
            x1, x2 = self._step(left, right)
            if self.f(x1) < self.f(x2):
                right = x2
            else:
                left = x1
            self._log.append([left, right])

        return (left + right) / 2

    def opt(self):
        is_watcher = isinstance(self.f, Watcher)
        if is_watcher:
            self.f.reset()
            self.f.start_count()
        res = self._opt_inner()
        if is_watcher:
            self.f.stop_count()
        return res

    def log_frame(self):
        return pd.DataFrame([[i] + self._log[i] for i in range(len(self._log))],
                            columns=self._log_headers)
