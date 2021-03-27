from lab1.task1.reused_opt import ReusedOpt


class Fibonacci(ReusedOpt):
    name = "Fibonacci"

    def __init__(self, f, eps, bounds):
        super().__init__(f, eps, bounds)
        self.n = -1
        self.fib = []
        self.fib_n = -1

    def __next_x(self, x, pop=True):
        if pop:
            last = self.fib.pop(-1)
        else:
            last = self.fib[-1]
        return x + last / self.fib_n

    def _step(self, x1, x2):
        nx2 = self.__next_x(x1)
        nx1 = self.__next_x(x1, pop=False)
        return nx1, nx2

    def opt(self):
        self.__fib_precalc()
        return super().opt()

    def __fib_precalc(self):
        self.fib = [1, 1, 2]
        bound = (self.right - self.left) / self.eps
        while bound >= self.fib[-1]:
            self.fib.append(self.fib[-1] + self.fib[-2])
        self.n = len(self.fib) - 2
        self.fib_n = self.fib.pop(-1) / (self.right - self.left)
