import collections

from .function import Function
from typing import Optional


class Module:
    def __init__(self, source_file, name, triple, data_layout, struct_types):
        self.name = name
        self.source_file = source_file
        self.triple = triple
        self.data_layout = data_layout
        self.struct_types = struct_types
        self.globals = collections.OrderedDict()
        self.functions = {}
        self.declarations = {}
        self.global_vars = []

    def save(self, path):
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def get_function(self, name) -> Optional[Function]:
        """
        Get a function by name.
        """
        if name not in self.functions:
            return None
        return self.functions[name]

    @property
    def global_values(self):
        """
        An iterable of global values in this module.
        """
        return self.globals.values()

    def get_global(self, name):
        """
        Get a global value by name.
        """
        return self.globals[name]

    def __repr__(self):
        lines = []
        # Header
        lines += [
            '; ModuleID = "%s"' % (self.name,),
            'target triple = "%s"' % (self.triple,),
            'target datalayout = "%s"' % (self.data_layout,),
            '']

        return "\n".join(lines)
