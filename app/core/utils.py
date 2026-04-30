import threading


class LazyProxy:
    def __init__(self, cls):
        object.__setattr__(self, '_cls', cls)
        object.__setattr__(self, '_obj', None)
        object.__setattr__(self, '_lock', threading.Lock())

    def _load(self):
        obj = object.__getattribute__(self, '_obj')
        if obj is not None:
            return obj
        lock = object.__getattribute__(self, '_lock')
        with lock:
            obj = object.__getattribute__(self, '_obj')
            if obj is None:
                cls = object.__getattribute__(self, '_cls')
                obj = cls()
                object.__setattr__(self, '_obj', obj)
            return obj

    def __getattr__(self, name):
        return getattr(self._load(), name)
