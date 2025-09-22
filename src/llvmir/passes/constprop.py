from ..function import Function

class ConstPropagation:
    def __init__(self, function: Function):
        self.function = function