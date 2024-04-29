from .value import Value


class Constant(Value):
    __match_args__ = ('value',)

    def __init__(self, type, value):
        super().__init__(type)
        self.value = value

    def __eq__(self, other) -> bool:
        return self.type == other.type and self.value == other.value

    def has_name(self):
        return False

    def sname(self):
        return str(self)

    def __str__(self):
        return f"{self.type} {self.value}"


class ConstantInt(Constant):
    __match_args__ = ('value',)

    def __init__(self, type, value):
        super().__init__(type, value)
        self.value = value

    def __eq__(self, value) -> bool:
        if not isinstance(value, ConstantInt):
            return False
        return self.value == value.value


class Null(Constant):
    def __init__(self):
        super().__init__(0, 0)

    def __str__(self):
        return "null"

    def __eq__(self, value) -> bool:
        return id(self) == id(value)


Null = Null()  # Singleton


class InlineAsm(Constant):
    # __match_args__ = ('asm_str', 'constraints', 'side_effects',
    #                   'type', 'has_side_effects', 'is_align_stack')

    def __init__(self, type, asm_str):
        super().__init__(type, asm_str)

    def __str__(self):
        return f"asm {self.asm_str}, {self.constraints}, {self.side_effects}, {self.has_side_effects}, {self.is_align_stack}"
