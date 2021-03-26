from lab1.task1.opt import Opt


class Dichotomy(Opt):

    def _step(self, x1, x2):
        m = (x1 + x2) / 2
        return m - self.eps / 3, m + self.eps / 3

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
