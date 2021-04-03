import numpy as np
import pandas as pd

from lab1.watcher import Watcher


class FastestGradientDescent:

    def __init__(self, f, eps, unopt):
        self.eps = eps
        self.f = f
        self._log = []
        self.unopt = unopt

    def get_iterations_cnt(self):
        return len(self._log) - 1

    def _desire_step(self, cur, grad):
        def gr(x):
            dx, dy = self.f.grad(cur - x * grad)
            return grad[0] * dx + grad[1] * dy
        f = Watcher(lambda x: self.f(*(cur - x * grad)), gr)
        return self.unopt(f, 1e-7, [0, 1]).opt()

    def opt(self, st):
        cur = np.array(st)

        self.f.reset()
        self._log.append(st + [self.f(*st)])
        self.f.start_count()
        iter = 0
        while True:
            grad = self.f.grad(cur)
            l = self._desire_step(cur, grad)
            new = cur - l * grad
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
