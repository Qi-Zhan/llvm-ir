from .cfg import CFG
from .def_use import DefUse


class ModuleAnalysisManager:
    pass


class FunctionAnalysisManager:
    def __init__(self, function):
        self.function = function
        self.cfg = None
        self.def_use = None

    def get_cfg(self):
        if self.cfg is None:
            self.cfg = CFG(self.function)
        return self.cfg

    def get_def_use(self):
        if self.def_use is None:
            self.def_use = DefUse(self.function)
        return self.def_use
