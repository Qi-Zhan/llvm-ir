"""
Implementation of LLVM IR function
"""

from .value import Value
from .utils import indent, get_name_str
from .typed import FunctionType


class Function(Value):

    def __init__(self, name, parent):
        self.name = name
        self.blocks = []
        self.arguments = []
        self.return_type = None
        self.parent = parent
        super().__init__(None)

    @property
    def type(self):
        return FunctionType(self.return_type, [arg.type for arg in self.arguments])

    def has_name(self):
        return True

    def get_name(self):
        return self.name

    def __str__(self):
        args_str = ", ".join([str(arg) for arg in self.arguments])
        blocks_str = "\n".join([indent(str(block)) for block in self.blocks])
        return (
            f"define {self.return_type} @{self.name}({args_str}) {{\n{blocks_str}\n}}"
        )


class Argument(Value):
    def __init__(self, name, type):
        super().__init__(type)
        self.name = name

    def has_name(self):
        return True

    def get_name(self):
        return self.name

    def __str__(self):
        return f"{self.type} {get_name_str(self.name)}"
