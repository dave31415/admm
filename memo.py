import pickle as cPickel
# import cPickle


def memo(f):
    """ Memoization decorator for a function taking a single argument """
    class memodict(dict):
        def __missing__(self, key):
            ret = self[key] = f(key)
            return ret
    return memodict().__getitem__


class memo_mutable:
    """ Memoization decorator for a function with args and keywords
        and works with arguments that are mutable (e.g. numpy arrays)
        Adapted from Python cookbook.
    """
    def __init__(self, fn):
        self.fn = fn
        self.memo_dict = {}

    def __call__(self, *args, **kwds):
        string = cPickle.dumps(args, 1) + cPickle.dumps(kwds, 1)
        if not string in self.memo_dict:
            self.memo_dict[string] = self.fn(*args, **kwds)

        return self.memo_dict[string]


class memo_array:
    """ Memoization decorator for a function with single array argument.
        About twice as fast as memo_mutable but less general.
    """
    def __init__(self, fn):
        self.fn = fn
        self.memo_dict = {}

    def __call__(self, arg):
        bytes = arg.tobytes()
        if not bytes in self.memo_dict:
            self.memo_dict[bytes] = self.fn(arg)

        return self.memo_dict[bytes]