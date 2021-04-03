from collections import Iterable


class Watcher:
    def __init__(self, f, grad=None):
        self.__f = f
        self.invocations = 0
        self.grad_invocations = 0
        self._grad = grad
        self.__is_count = False

    def __call__(self, *args, **kwargs):
        if self.__is_count:
            self.invocations = self.invocations + 1
        return self.__f(*args, **kwargs)

    def grad(self, xs):
        if self._grad is None:
            raise RuntimeError("No info about gradient")
        if self.__is_count:
            self.grad_invocations = self.grad_invocations + 1
        if isinstance(xs, Iterable):
            return self._grad(*xs)
        else:
            return self._grad(xs)

    def stop_count(self):
        self.__is_count = False

    def start_count(self):
        self.__is_count = True

    def reset(self):
        self.__is_count = False
        self.invocations = 0
        self.grad_invocations = 0
