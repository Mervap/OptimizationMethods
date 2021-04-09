from abc import abstractmethod
import numpy as np
import pandas as pd


class UndimOpt:
    def __init__(self, f, eps, unopt):
        self.eps = eps
        self.f = f
        self._log = []
        self.unopt = unopt

    def get_iterations_cnt(self):
        return len(self._log) - 1

    @abstractmethod
    def _step(self, x):
        pass

    def opt(self, st):
        cur = np.array(st)

        self.f.reset()
        self._log.append(st + [self.f(*st)])
        self.f.start_count()
        iter = 0
        while True:
            new = self._step(cur)
            diff = new - cur
            if np.linalg.norm(diff) > 1e9:
                raise RuntimeError("Diverges")
            if np.linalg.norm(diff) < self.eps:
                break
            cur = new
            self.f.stop_count()
            self._log.append(list(cur) + [self.f(*cur)])
            self.f.start_count()
            iter = iter + 1
            if iter > 1e5:
                raise RuntimeError("Too many ops")

        self.f.stop_count()
        return cur

    def log_frame(self):
        return pd.DataFrame([[i] + self._log[i] for i in range(len(self._log))],
                            columns=['ind', 'x', 'y', 'f'])