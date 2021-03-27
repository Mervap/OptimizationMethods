from lab1.task1.opt import Opt


class Dichotomy(Opt):

    def _step(self, x1, x2):
        m = (x1 + x2) / 2
        return m - self.eps / 3, m + self.eps / 3
