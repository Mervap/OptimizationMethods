import numpy as np

from lab1.task2.undim_opt import UndimOpt

class ConjGrad(UndimOpt):
    name = "ConjGrad"
    w = None
    p = None
    x_prev = None

    def norm(self, x):
        return np.dot(np.transpose(x), x)
    
    def _step(self, x):
        if self.w is None:
            self.w = -self.f.grad(x)
            self.p = self.w
            self.x_prev = x
            xi = self.unopt(lambda xi: self.f(*(x + xi * self.p)), 1e-7, [-1, 1]).opt()
        else:
            w = -self.f.grad(x)
            gamma = max(0, self.norm(w) / self.norm(self.w))
            self.p = w + gamma * self.p
            xi = self.unopt(lambda xi: self.f(*(x + xi * self.p)), 1e-7, [-1, 1]).opt()
            self.w = w
        self.x_prev = x
        return x + xi * self.p
