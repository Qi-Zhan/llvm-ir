"""
Implementation of LLVM IR basic block
"""


from .type import LabelType
from .value import Value
from .utils import get_name_str


class BasicBlock(Value):
    def __init__(self, name, parent):
        super().__init__(LabelType())
        self.name = name
        self.instructions = []
        self.parent = parent

    def has_name(self):
        return True

    def get_name(self):
        return self.name

    def sname(self):
        return get_name_str(self.name)

    def all_but_last(self):
        return self.instructions[:-1]

    def terminator(self):
        return self.instructions[-1]

    def dot_str(self):
        instructions = "\n".join([f"  {instr}" for instr in self.instructions])
        return f"%{self.name}\n{instructions}"

    def __str__(self):
        instructions = "\n".join([f"  {instr}" for instr in self.instructions])
        return f"%{self.name}\n{instructions}"
