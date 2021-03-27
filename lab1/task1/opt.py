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

    def opt(self):
        left = self.left
        right = self.right

        self._log.append([left, right])
        self.f.reset()
        self.f.start_count()
        while right - left > self.eps:
            x1, x2 = self._step(left, right)
            if self.f(x1) < self.f(x2):
                right = x2
            else:
                left = x1
            self._log.append([left, right])

        self.f.stop_count()
        return (left + right) / 2

    def log_frame(self):
        return pd.DataFrame([[i] + self._log[i] for i in range(len(self._log))],
                            columns=self._log_headers)
