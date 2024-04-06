import enum


class DefUse:
    def __init__(self, function):
        self.function = function
        self.def_use = {}
        raise NotImplementedError


class DefUseEdge(enum.IntEnum):
    def_ = 1
    use = 2
