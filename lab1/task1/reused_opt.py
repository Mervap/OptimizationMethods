from abc import ABC

from lab1.task1.opt import Opt


class ReusedOpt(Opt, ABC):
    def __init__(self, f, eps, bounds):
        super().__init__(f, eps, bounds)
        self._log_headers = self._log_headers + ['x1', 'x2']

    def _opt_inner(self):
        left = self.left
        right = self.right

        x1, x2 = self._step(left, right)
        self._log.append([left, right, x1, x2])
        f1 = self.f(x1)
        f2 = self.f(x2)
        while True:
            if f1 < f2:
                right = x2
                if right - left < self.eps:
                    break
                x1, x2 = self._step(left, right)
                f2 = f1
                f1 = self.f(x1)
            else:
                left = x1
                if right - left < self.eps:
                    break
                x1, x2 = self._step(left, right)
                f1 = f2
                f2 = self.f(x2)

            self._log.append([left, right, x1, x2])

        return (left + right) / 2
