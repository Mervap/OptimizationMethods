import numpy as np

from lab1.task2.undim_opt import UndimOpt


def to_pos_def(x):
    eigenvalues = np.linalg.eigvals(x)
    mn = min(eigenvalues)
    if mn > 0:
        return x
    else:
        return x + (np.eye(len(x)) * (-mn) * (1 + 1e-2))


class Newton(UndimOpt):
    name = "Newton"

    def _step(self, x):
        grad = self.f.grad(x)
        hessian = to_pos_def(self.f.hessian(x))
        inv = np.linalg.inv(hessian)
        direction = inv.dot(grad)
        mul = self.unopt(lambda l: self.f(*(x - l * direction)), 1e-7, [0, 1]).opt()
        return x - mul * direction
