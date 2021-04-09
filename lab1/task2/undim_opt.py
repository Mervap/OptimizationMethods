from abc import abstractmethod
import numpy as np
import pandas as pd


class UndimOpt:
    def __init__(self, f, eps, unopt, preserve_logs=True):
        self.eps = eps
        self.f = f
        self._log = []
        self.unopt = unopt
        self.preserve_logs = preserve_logs

    def get_iterations_cnt(self):
        return len(self._log) - 1

    @abstractmethod
    def _step(self, x):
        pass

    def opt(self, st):
        cur = np.array(st)

        self.f.reset()
        if self.preserve_logs:
            self._log = [list(st) + [self.f(*st)]]
        else:
            self._log = []
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
            if self.preserve_logs:
                self.f.stop_count()
                self._log.append(list(cur) + [self.f(*cur)])
                self.f.start_count()
            iter = iter + 1
            if iter > 3 * 1e4:
                raise RuntimeError("Too many ops")

        self.f.stop_count()
        return cur

    def log_frame(self):
        if not self.preserve_logs:
            raise RuntimeError("Log not preserved")
        return pd.DataFrame([[i] + self._log[i] for i in range(len(self._log))],
                            columns=['ind', 'x', 'y', 'f'])
