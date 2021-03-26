import math

from lab1.task1.opt import Opt


class GoldenRatio(Opt):
    golder_c = (math.sqrt(5) - 1) / 2

    def __init__(self, f, eps, bounds):
        super().__init__(f, eps, bounds)
        self._log_headers = self._log_headers + ['x1', 'x2']

    def _step(self, x1, x2):
        diff = x2 - x1
        return x2 - GoldenRatio.golder_c * diff, x1 + GoldenRatio.golder_c * diff

    def opt(self):
        left = self.left
        right = self.right

        self.f.reset()
        self.f.start_count()

        x1, x2 = self._step(left, right)
        self._log.append([left, right, x1, x2])
        f1 = self.f(x1)
        f2 = self.f(x2)
        while right - left > self.eps:
            if f1 < f2:
                right = x2
                x1, x2 = self._step(left, right)
                f2 = f1
                f1 = self.f(x1)
            else:
                left = x1
                x1, x2 = self._step(left, right)
                f1 = f2
                f2 = self.f(x2)

            self._log.append([left, right, x1, x2])

        self.f.stop_count()
        return (left + right) / 2

