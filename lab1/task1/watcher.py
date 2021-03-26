class Watcher:
    def __init__(self, f):
        self.__f = f
        self.invocations = 0
        self.__is_count = True

    def __call__(self, *args, **kwargs):
        if self.__is_count:
            self.invocations = self.invocations + 1
        return self.__f(*args, **kwargs)

    def stop_count(self):
        self.__is_count = False

    def start_count(self):
        self.__is_count = True

    def reset(self):
        self.invocations = 0
