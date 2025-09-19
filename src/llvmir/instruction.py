"""
Implementation of LLVM IR instructions.
"""

import enum
from typing import Optional

from .typed import VoidType
from .value import Value
from .constant import Constant
from .utils import get_name_str

import llvmir.binding as llvm


class Instruction(Value):

    def __init__(self, type, parent, debugloc):
        super().__init__(type)
        self.parent = parent
        self.debugloc = debugloc

    def has_name(self):
        """default implementation"""
        return True

    def get_name(self):
        return self.name

    def sname(self):
        return get_name_str(self.name)


class AllocaInst(Instruction):
    __match_args__ = ("alloca_type",)

    def __init__(self, type, alloca_type, name, parent, debugloc=None):
        super().__init__(type, parent, debugloc)
        self.name = name
        self.alloca_type = alloca_type

    def __str__(self):
        return f"{get_name_str(self.name)} = alloca {self.alloca_type}"


class StoreInst(Instruction):
    __match_args__ = ("value", "address")

    def __init__(self, value: Value, address: Value, parent, debugloc=None):
        super().__init__(VoidType(), parent, debugloc)
        self.value = value
        self.address = address
        self.value.add_use(self)
        self.address.add_use(self)

    def has_name(self):
        return False

    def __str__(self):
        match self.value:
            case Constant(c):
                value = c
            case _:
                value = f"%{self.value.get_name()}"
        return f"store {self.value.type} {value}, {get_name_str(self.address.sname())}"


class LoadInst(Instruction):
    __match_args__ = ("ptr",)

    def __init__(self, type, name, ptr: Value, parent, debugloc=None):
        super().__init__(type, parent, debugloc)
        self.name = name
        self.ptr = ptr
        self.ptr.add_use(self)

    def __str__(self):
        return f"{get_name_str(self.name)} = load {self.type} {get_name_str(self.ptr.sname())}"


class GetElementPtrInst(Instruction):
    __match_args__ = ("ptr", "indices")

    def __init__(
        self, type, name, ptr: Value, indices: list[Value], parent, debugloc=None
    ):
        super().__init__(type, parent, debugloc)
        self.name = name
        self.ptr = ptr
        self.indices = indices
        self.ptr.add_use(self)
        for index in self.indices:
            index.add_use(self)

    def __str__(self):
        indices = ", ".join([index.sname() for index in self.indices])
        return f"{get_name_str(self.name)} = getelementptr {self.ptr.type}, {self.ptr.sname()}, {indices}"


class IntPredicate(enum.Enum):
    EQ = "eq"
    NE = "ne"
    UGT = "ugt"
    UGE = "uge"
    ULT = "ult"
    ULE = "ule"
    SGT = "sgt"
    SGE = "sge"
    SLT = "slt"
    SLE = "sle"

    def __str__(self):
        return self.value


class ICmpInst(Instruction):
    __match_args__ = ("predicate", "lhs", "rhs")

    def __init__(
        self, type, name, predicate, lhs: Value, rhs: Value, parent, debugloc=None
    ):
        super().__init__(type, parent, debugloc)
        self.name = name
        match predicate:
            case llvm.value.IntPredicate.eq:
                self.predicate = IntPredicate.EQ
            case llvm.value.IntPredicate.ne:
                self.predicate = IntPredicate.NE
            case llvm.value.IntPredicate.ugt:
                self.predicate = IntPredicate.UGT
            case llvm.value.IntPredicate.uge:
                self.predicate = IntPredicate.UGE
            case llvm.value.IntPredicate.ult:
                self.predicate = IntPredicate.ULT
            case llvm.value.IntPredicate.ule:
                self.predicate = IntPredicate.ULE
            case llvm.value.IntPredicate.sgt:
                self.predicate = IntPredicate.SGT
            case llvm.value.IntPredicate.sge:
                self.predicate = IntPredicate.SGE
            case llvm.value.IntPredicate.slt:
                self.predicate = IntPredicate.SLT
            case llvm.value.IntPredicate.sle:
                self.predicate = IntPredicate.SLE
        self.lhs = lhs
        self.rhs = rhs
        self.lhs.add_use(self)
        self.rhs.add_use(self)

    def __str__(self):
        return f"{get_name_str(self.name)} = icmp {self.predicate} {self.lhs.sname()}, {self.rhs.sname()}"


class CallInst(Instruction):
    __match_args__ = ("callee", "args")

    def __init__(
        self, type, name: str, callee: Value, args: list[Value], parent, debugloc=None
    ):
        super().__init__(type, parent, debugloc)
        self.name = name
        self.callee = callee
        self.args = args
        self.callee.add_use(self)
        for arg in self.args:
            arg.add_use(self)

    def __str__(self):
        args = ", ".join([arg.sname() for arg in self.args])
        if self.type == VoidType():
            return f"call {self.callee}({args})"
        return f"{get_name_str(self.name)} = call {self.type} {self.callee.get_name()}({args})"


class BinOp(enum.Enum):
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    UDIV = "udiv"
    SDIV = "sdiv"
    UREM = "urem"
    SREM = "srem"
    SHL = "shl"
    LSHR = "lshr"
    ASHR = "ashr"
    AND = "and"
    OR = "or"
    XOR = "xor"
    FADD = "fadd"
    FSUB = "fsub"
    FMUL = "fmul"
    FDIV = "fdiv"

    def __str__(self):
        return self.value


class BinaryOperator(Instruction):
    __match_args__ = ("op", "lhs", "rhs")

    def __init__(
        self, type, name, op: str, lhs: Value, rhs: Value, parent, debugloc=None
    ):
        super().__init__(type, parent, debugloc)
        self.name = name
        match op:
            case "add":
                self.op = BinOp.ADD
            case "sub":
                self.op = BinOp.SUB
            case "mul":
                self.op = BinOp.MUL
            case "udiv":
                self.op = BinOp.UDIV
            case "sdiv":
                self.op = BinOp.SDIV
            case "urem":
                self.op = BinOp.UREM
            case "srem":
                self.op = BinOp.SREM
            case "shl":
                self.op = BinOp.SHL
            case "lshr":
                self.op = BinOp.LSHR
            case "ashr":
                self.op = BinOp.ASHR
            case "and":
                self.op = BinOp.AND
            case "or":
                self.op = BinOp.OR
            case "xor":
                self.op = BinOp.XOR
            case "fadd":
                self.op = BinOp.FADD
            case "fsub":
                self.op = BinOp.FSUB
            case "fmul":
                self.op = BinOp.FMUL
            case "fdiv":
                self.op = BinOp.FDIV
            case _:
                raise ValueError(f"Unknown binary operator {op}")
        self.lhs = lhs
        self.rhs = rhs
        self.lhs.add_use(self)
        self.rhs.add_use(self)

    def __str__(self):
        return f"{get_name_str(self.name)} = {self.op} {self.lhs.sname()}, {self.rhs.sname()}"


class UnOp(enum.Enum):
    NEG = "neg"
    NOT = "not"


class UnaryOperator(Instruction):
    __match_args__ = (
        "op",
        "value",
    )

    def __init__(self, type, name, op: str, value: Value, parent, debugloc=None):
        super().__init__(type, parent, debugloc)
        self.name = name
        match op:
            case "neg":
                self.op = UnOp.NEG
            case "not":
                self.op = UnOp.NOT
            case _:
                raise ValueError(f"Unknown unary operator {op}")
        self.value = value
        self.value.add_use(self)


class PhiNode(Instruction):
    __match_args__ = "incoming_values"

    def __init__(
        self, type, name, incoming_values: list[Value, Value], parent, debugloc=None
    ):
        super().__init__(type, parent, debugloc)
        self.name = name
        self.incoming_values = incoming_values
        for value, block in self.incoming_values:
            value.add_use(self)
            block.add_use(self)

    def __str__(self):
        incoming_values = ", ".join(
            [
                f"[{value.sname()}, {block.sname()}]"
                for value, block in self.incoming_values
            ]
        )
        return f"{get_name_str(self.name)} = phi {self.type} {incoming_values}"


class SelectInst(Instruction):
    __match_args__ = ("condition", "true", "false")

    def __init__(
        self,
        type,
        name,
        condition: Value,
        true: Value,
        false: Value,
        parent,
        debugloc=None,
    ):
        super().__init__(type, parent, debugloc)
        self.name = name
        self.condition = condition
        self.true = true
        self.false = false
        self.condition.add_use(self)
        self.true.add_use(self)
        self.false.add_use(self)

    def __str__(self):
        return f"{get_name_str(self.name)} = select {self.condition.type} {self.condition.sname()}, {self.true.type} {self.true.sname()}, {self.false.type} {self.false.sname()}"


class BitCastInst(Instruction):
    __match_args__ = ("value",)

    def __init__(self, type, name, value: Value, parent, debugloc=None):
        super().__init__(type, parent, debugloc)
        self.name = name
        self.value = value
        self.value.add_use(self)

    def __str__(self):
        return f"{get_name_str(self.name)} = bitcast {self.value.type} {self.value.sname()} to {self.type}"


class SExtInst(Instruction):
    __match_args__ = ("value",)

    def __init__(self, type, name, value: Value, parent, debugloc=None):
        super().__init__(type, parent, debugloc)
        self.name = name
        self.value = value
        self.value.add_use(self)

    def __str__(self):
        return f"{get_name_str(self.name)} = sext {self.value.type} {self.value.sname()} to {self.type}"


class ZExtInst(Instruction):
    __match_args__ = ("value",)

    def __init__(self, type, name, value: Value, parent, debugloc=None):
        super().__init__(type, parent, debugloc)
        self.name = name
        self.value = value
        self.value.add_use(self)

    def __str__(self):
        return f"{get_name_str(self.name)} = zext {self.value.type} {self.value.sname()} to {self.type}"


class FPExtInst(Instruction):
    __match_args__ = ("value",)

    def __init__(self, type, name, value: Value, parent, debugloc=None):
        super().__init__(type, parent, debugloc)
        self.name = name
        self.value = value
        self.value.add_use(self)

    def __str__(self):
        return f"{get_name_str(self.name)} = fpext {self.value.type} {self.value.sname()} to {self.type}"


class PtrToIntInst(Instruction):
    __match_args__ = ("value",)

    def __init__(self, type, name, value: Value, parent, debugloc=None):
        super().__init__(type, parent, debugloc)
        self.name = name
        self.value = value
        self.value.add_use(self)

    def __str__(self):
        return f"{get_name_str(self.name)} = ptrtoint {self.value.type} {self.value.sname()} to {self.type}"


class IntToPtrInst(Instruction):
    __match_args__ = ("value",)

    def __init__(self, type, name, value: Value, parent, debugloc=None):
        super().__init__(type, parent, debugloc)
        self.name = name
        self.value = value
        self.value.add_use(self)

    def __str__(self):
        return f"{get_name_str(self.name)} = inttoptr {self.value.type} {self.value.sname()} to {self.type}"


class UIToFPInst(Instruction):
    __match_args__ = ("value",)

    def __init__(self, type, name, value: Value, parent, debugloc=None):
        super().__init__(type, parent, debugloc)
        self.name = name
        self.value = value
        self.value.add_use(self)

    def __str__(self):
        return f"{get_name_str(self.name)} = uitofp {self.value.type} {self.value.sname()} to {self.type}"


class SIToFPInst(Instruction):
    __match_args__ = ("value",)

    def __init__(self, type, name, value: Value, parent, debugloc=None):
        super().__init__(type, parent, debugloc)
        self.name = name
        self.value = value
        self.value.add_use(self)

    def __str__(self):
        return f"{get_name_str(self.name)} = sitofp {self.value.type} {self.value.sname()} to {self.type}"


class TruncInst(Instruction):
    __match_args__ = ("value",)

    def __init__(self, type, name, value: Value, parent, debugloc=None):
        super().__init__(type, parent, debugloc)
        self.name = name
        self.value = value
        self.value.add_use(self)

    def __str__(self):
        return f"{get_name_str(self.name)} = trunc {self.value.type} {self.value.sname()} to {self.type}"


class FPTruncInst(Instruction):
    __match_args__ = ("value",)

    def __init__(self, type, name, value: Value, parent, debugloc=None):
        super().__init__(type, parent, debugloc)
        self.name = name
        self.value = value
        self.value.add_use(self)

    def sname(self):
        return get_name_str(self.name)

    def __str__(self):
        return f"{get_name_str(self.name)} = fptrunc {self.value.type} {self.value.sname()} to {self.type}"


class InsertValueInst(Instruction):
    __match_args__ = ("value", "index")

    def __init__(
        self, type, name, value: Value, index: list[int], parent, debugloc=None
    ):
        super().__init__(type, parent, debugloc)
        self.name = name
        self.value = value
        self.index = index
        self.value.add_use(self)

    def __str__(self):
        return f"{get_name_str(self.name)} = insertvalue {self.value.type} {self.value.sname()}, {self.type} undef, {self.index}"


class Terminator(Instruction):
    def __init__(self, type, parent, debugloc):
        super().__init__(type, parent, debugloc)

    # Terminator defalut no name
    def has_name(self):
        return False


class ReturnInst(Terminator):
    __match_args__ = ("value",)

    def __init__(self, value: Optional[Value], parent, debugloc=None):
        super().__init__(VoidType(), parent, debugloc)
        self.value = value
        if self.value:
            self.value.add_use(self)

    def __str__(self):
        if self.value is None:
            return "ret void"
        return f"ret {self.value.type} {self.value.sname()}"


class BranchInst(Terminator):
    __match_args__ = ("dest",)

    def __init__(self, dest, parent, debugloc=None):
        super().__init__(VoidType(), parent, debugloc)
        self.dest = dest
        self.dest.add_use(self)

    def __str__(self):
        return f"br label {self.dest.sname()}"


class CondBrInst(Terminator):
    __match_args__ = ("condition", "true", "false")

    def __init__(
        self, condition: Value, false: Value, true: Value, parent, debugloc=None
    ):
        super().__init__(VoidType(), parent, debugloc)
        self.condition = condition
        self.true = true
        self.false = false
        self.condition.add_use(self)
        self.true.add_use(self)
        self.false.add_use(self)

    def __str__(self):
        return f"br i1 {self.condition.sname()}, label {self.true.sname()}, label {self.false.sname()}"


class SwitchInst(Terminator):
    __match_args__ = ("condition", "default", "cases")

    def __init__(
        self,
        condition: Value,
        default: Value,
        cases: list[tuple[Value, Value]],
        parent,
        debugloc=None,
    ):
        super().__init__(VoidType(), parent, debugloc)
        self.condition = condition
        self.default = default
        self.cases = cases
        self.condition.add_use(self)
        self.default.add_use(self)
        for value, block in self.cases:
            value.add_use(self)
            block.add_use(self)

    def __str__(self):
        cases = ", ".join(
            [f"[{value.sname()}, {block.sname()}]" for value, block in self.cases]
        )
        return (
            f"switch {self.condition.sname()}, label {self.default.sname()} [{cases}]"
        )


class UnreachableInst(Terminator):
    def __init__(self, parent, debugloc=None):
        super().__init__(VoidType(), parent, debugloc)

    def __str__(self):
        return "unreachable"
