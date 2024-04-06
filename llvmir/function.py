"""
Implementation of LLVM IR function
"""

from .value import Value


class Function(Value):

    def __init__(self, name, parent):
        self.name = name
        self.blocks = []
        self.arguments = []
        self.return_type = None
        self.parent = parent


class Argument(Value):
    def __init__(self, name, type):
        super().__init__(type)
        self.name = name

    def has_name(self):
        return True

    def get_name(self):
        return self.name

    def __str__(self):
        return f"{self.type} {self.name}"
