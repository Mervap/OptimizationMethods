from lab1.task1.opt import Opt


class BinarySearch(Opt):

    name = "Binary search"

    def _step(self, x1, x2):
        return (x1 + x2) / 2

    def _opt_inner(self):
        left = self.left
        right = self.right
        self._log.append([left, right])
        l_grad = self.f.grad(left)
        while right - left > self.eps:
            m = self._step(left, right)
            if (self.f.grad(m) < 0) == (l_grad < 0):
                left = m
            else:
                right = m
            self._log.append([left, right])

        return (left + right) / 2
