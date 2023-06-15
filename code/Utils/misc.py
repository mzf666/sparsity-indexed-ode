import sys


class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def func_timer(function):
    from functools import wraps

    @wraps(function)
    def function_timer(*args, **kwargs):
        import time
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print('[{name}() finished, time elapsed: {time:.5f}s]'.format(name=function.__name__, time=t1 - t0))
        return result

    return function_timer
