class Type:
    """https://llvm.org/doxygen/classllvm_1_1Type.html"""

    def size(self) -> int:
        assert False, f"You should implement in subclass {
            self.__class__.__name__}"

    def __eq__(self, _other):
        assert False, f"You should implement in subclass {
            self.__class__.__name__}"

    def __str__(self):
        assert False, f"You should implement in subclass {
            self.__class__.__name__}"


class LabelType(Type):

    def size(self):
        return 0

    def __str__(self):
        return "label"

    def __call__(self):
        return self

    def __eq__(self, other):
        return self is other


LabelType = LabelType()  # Singleton


class VoidType(Type):
    def __init__(self):
        self.width = 0

    def size(self):
        return 0

    def __str__(self):
        return "void"

    def __call__(self):
        return self

    def __eq__(self, other):
        return self is other


VoidType = VoidType()  # Singleton


class PointerType(Type):
    __match_args__ = ('element_type',)

    def __init__(self, width, element_type: Type):
        self.width = width
        self.element_type = element_type

    def size(self):
        # TODO: assume 64-bit pointer
        return 8

    def __str__(self):
        return f"{self.element_type}*"

    def __eq__(self, other):
        return isinstance(other, PointerType) and self.element_type == other.element_type


class IntegerType(Type):
    __match_args__ = ('width')

    def __init__(self, width: int):
        self.width = width

    def size(self):
        return self.width // 8

    def __str__(self):
        return f"i{self.width}"

    def __eq__(self, other):
        return isinstance(other, IntegerType) and self.width == other.width


class DoubleType(Type):
    __match_args__ = ('width')

    def __init__(self, width):
        self.width = width

    def size(self):
        return self.width // 8

    def __str__(self):
        return "double"

    def __eq__(self, other):
        return isinstance(other, DoubleType) and self.width == other.width


class FloatType(Type):
    __match_args__ = ('width')

    def __init__(self, width):
        self.width = width

    def size(self):
        return self.width // 8

    def __str__(self):
        return "float"

    def __eq__(self, other):
        return isinstance(other, FloatType) and self.width == other.width


class ArrayType(Type):
    __match_args__ = ('element_count', 'element_type')

    def __init__(self, element_count: int, element_type: Type):
        self.element_type = element_type
        self.element_count = element_count

    def size(self):
        return self.element_count * self.element_type.size()

    def __str__(self):
        return f"[{self.element_count} x {self.element_type}]"

    def __eq__(self, other):
        return isinstance(other, ArrayType) and self.element_type == other.element_type and self.element_count == other.element_count


class StructType(Type):
    __match_args__ = ('name', 'elements')
    named_structs = {}

    @staticmethod
    def exists(name):
        return name in StructType.named_structs

    @staticmethod
    def get(name):
        return StructType.named_structs[name]

    def __init__(self, name):
        self.name = name
        self.elements = []  # set elements later
        StructType.named_structs[name] = self

    def size(self):
        return sum(e.size() for e in self.elements)

    def size_up_to(self, index):
        return sum(e.size() for e in self.elements[:index])

    def __str__(self):
        return f"{self.name}"

    def __eq__(self, other):
        return isinstance(other, StructType) and self.name == other.name


class FunctionType(Type):
    __match_args__ = ('return_type', 'args')

    def __init__(self, return_type: Type, args: list[Type]):
        self.return_type = return_type
        self.args = args

    def __str__(self):
        return f"{self.return_type} ({', '.join(map(str, self.args))})"

    def __eq__(self, other):
        return isinstance(other, FunctionType) and self.return_type == other.return_type and self.args == other.args


class MetadataType(Type):
    def __str__(self):
        return "metadata"

    def __eq__(self, other):
        return self is other
