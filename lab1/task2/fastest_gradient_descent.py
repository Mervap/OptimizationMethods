from lab1.task2.undim_opt import UndimOpt
from lab1.watcher import Watcher


class FastestGradientDescent(UndimOpt):

    def _desire_step(self, cur, grad):
        def gr(x):
            dx, dy = self.f.grad(cur - x * grad)
            return grad[0] * dx + grad[1] * dy
        f = Watcher(lambda x: self.f(*(cur - x * grad)), gr)
        return self.unopt(f, 1e-7, [0, 1]).opt()

    def _step(self, x):
        grad = self.f.grad(x)
        l = self._desire_step(x, grad)
        return x - l * grad
