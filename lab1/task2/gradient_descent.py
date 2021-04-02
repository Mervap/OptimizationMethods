import numpy as np
import pandas as pd


class GradientDescent:

    def __init__(self, f, eps, unopt):
        self.eps = eps
        self.f = f
        self.unopt = unopt
        self._log = []

    def opt(self, st):
        cur = np.array(st)

        self.f.reset()
        self.f.start_count()
        self._log.append(st)
        while True:
            grad = self.f.grad(cur)
            l = self.unopt(lambda x: self.f(*(cur - x * grad)), 1e-5, [0, 10]).opt()
            new = cur - l * grad
            diff = new - cur
            if np.linalg.norm(diff) > 1e9:
                raise RuntimeError("Diverges")
            if np.linalg.norm(diff) < self.eps:
                break
            cur = new
            self._log.append(list(cur))

        self.f.stop_count()
        return cur

    def log_frame(self):
        return pd.DataFrame([[i] + self._log[i] for i in range(len(self._log))],
                            columns=['ind', 'x', 'y'])
