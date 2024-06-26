class Value:
    """ Value class is the base class of all values computed by a program that may be used as operands to other values. 
    Value is the super class of other important classes such as Instruction and Function. 
    All Values have a Type. Type is not a subclass of Value. 

    See <https://llvm.org/doxygen/classllvm_1_1Value.html>
    """

    def __init__(self, type):
        self.type = type
        self.uses = []

    def get_use_list(self):
        return self.uses

    def get_use(self, idx):
        return self.uses[idx]

    def add_use(self, use):
        self.uses.append(use)

    def get_type(self):
        return self.type

    def has_name(self):
        raise NotImplementedError(f"has_name is not implemented in {
                                  self.__class__.__name__}")

    def get_name(self):
        raise NotImplementedError(f"get_name is not implemented in {
                                  self.__class__.__name__}")

    def sname(self):
        """ Get the simple name of the value, specifically when other value wants to print"""
        return self.get_name()

    def __str__(self):
        assert False, f"__str__ is not implemented in {
            self.__class__.__name__}"

    def __repr__(self):
        if self.has_name():
            return f"<{self.__class__.__name__} name='{self.get_name()}' type='{self.type}'>"
        return f"<{self.__class__.__name__} type='{self.type}' ...>"
