from .cfg import CFG, CFGEdge, EntryNode, ExitNode


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
