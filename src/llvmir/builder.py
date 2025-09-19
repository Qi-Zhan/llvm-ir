from .module import Module
from .function import Function, Argument
from .block import BasicBlock
from .instruction import *
from .typed import *
from .value import Value
from .constant import *

from .binding.module import ModuleRef, parse_bitcode, parse_assembly
from .binding.typeref import TypeRef, TypeKind
from .binding.value import ValueKind


class Context:
    def __init__(self):
        self._id2value = {}

    def save_value(self, ffi_value, value: Value):
        id_ = ffi_value.address()
        self._id2value[id_] = value

    def get_value(self, ffi_value) -> Value:
        id_ = ffi_value.address()
        return self._id2value[id_]


class ModuleBuilder:
    def __init__(self):
        self.module = None
        self.function = None
        self.basic_block = None

    @staticmethod
    def from_path(path: str) -> Module:
        if path.endswith(".bc"):
            return ModuleBuilder.from_bc_path(path)
        elif path.endswith(".ll"):
            return ModuleBuilder.from_ir_path(path)
        else:
            raise ValueError("Invalid file type, expected .bc or .ll")

    @staticmethod
    def from_bc_path(path: str) -> Module:
        with open(path, "rb") as f:
            bitcode = f.read()
        ffi_module = parse_bitcode(bitcode)
        return ModuleBuilder._from_ffi_module(ffi_module)

    @staticmethod
    def from_ir_path(path: str) -> Module:
        with open(path, "r") as f:
            ir = f.read()
        ffi_module = parse_assembly(ir)
        return ModuleBuilder._from_ffi_module(ffi_module)

    @staticmethod
    def _from_ffi_module(ffi_module: ModuleRef) -> Module:
        transformer = ModuleBuilder()
        return transformer.build_module(ffi_module)

    @staticmethod
    def from_pickle(path: str) -> Module:
        import pickle

        with open(path, "rb") as f:
            mod = pickle.load(f)
        return mod

    def build_module(self, ffi_mod) -> Module:
        self.context = Context()
        source_file = ffi_mod.source_file
        name = ffi_mod.name
        triple = ffi_mod.triple
        data_layout = ffi_mod.data_layout
        struct_types = ffi_mod.struct_types
        module = Module(source_file, name, triple, data_layout, struct_types)
        self.module = module
        # scan all functions first
        for ffi_func in ffi_mod.functions:
            assert ffi_func.is_function
            function = Function(ffi_func.name, self.module)
            self.context.save_value(ffi_func, function)
            if ffi_func.is_declaration:
                module.declarations[function.name] = function
            else:
                module.functions[function.name] = function
        # build function body
        for ffi_func in ffi_mod.functions:
            function = module.get_function(ffi_func.name)
            self.build_function(function, ffi_func)
        return module

    def build_function(self, function: Function, ffi_func) -> Function:
        arguments = [self.build_argument(arg) for arg in ffi_func.arguments]
        function.arguments = arguments
        function.return_type = self.build_type(ffi_func.return_type())
        self.function = function
        # scan all basic blocks first
        basic_blocks = []
        for ffi_bb in ffi_func.blocks:
            assert ffi_bb.is_block
            name = ffi_bb.name
            basic_block = BasicBlock(name, self.function)
            self.context.save_value(ffi_bb, basic_block)
            basic_blocks.append(basic_block)
        # build instructions
        for i, ffi_bb in enumerate(ffi_func.blocks):
            basic_block = self.build_basicblock(ffi_bb, basic_blocks[i])
        function.blocks = basic_blocks
        # assign names
        self.build_function_name(function)
        return function

    def build_basicblock(self, ffi_bb, bb) -> BasicBlock:
        assert ffi_bb.is_block
        self.basic_block = bb
        for ffi_instr in ffi_bb.instructions:
            instruction = self.build_instruction(ffi_instr)
            self.context.save_value(ffi_instr, instruction)
            bb.instructions.append(instruction)
        return bb

    def build_instruction(self, instr) -> Instruction:
        assert instr.is_instruction
        name = instr.name
        match instr.opcode:
            case "alloca":
                type = self.build_type(instr.type)
                allocate_type = self.build_type(instr.allocated_type())
                return AllocaInst(type, allocate_type, name, self.basic_block)
            case "store":
                operands = [self.build_operand(operand) for operand in instr.operands]
                assert len(operands) == 2
                return StoreInst(operands[0], operands[1], self.basic_block)
            case "load":
                operand = self.build_operand(next(instr.operands))
                type = self.build_type(instr.type)
                return LoadInst(type, name, operand, self.basic_block)
            case "icmp":
                operands = [self.build_operand(operand) for operand in instr.operands]
                type = self.build_type(instr.type)
                predicate = instr.predicate
                return ICmpInst(
                    type, name, predicate, operands[0], operands[1], self.basic_block
                )
            case "ret":
                operands = [self.build_operand(operand) for operand in instr.operands]
                if len(operands) == 0:
                    return ReturnInst(None, self.basic_block)
                else:
                    return ReturnInst(operands[0], self.basic_block)
            case "call":
                type = self.build_type(instr.type)
                called_value = self.context.get_value(instr.called_value)
                if type != VoidType():
                    operands = [
                        self.build_operand(operand) for operand in instr.operands
                    ]
                    return CallInst(
                        type,
                        name,
                        called_value,
                        operands[:-1],
                        self.basic_block,
                    )
                else:
                    operands = [
                        self.build_operand(operand) for operand in instr.operands
                    ]
                    return CallInst(
                        type,
                        name,
                        called_value,
                        operands[:-1],
                        self.basic_block,
                    )
            case "br":
                operands = [self.build_operand(operand) for operand in instr.operands]
                length = len(operands)
                if length == 1:
                    return BranchInst(operands[0], self.basic_block)
                elif length == 3:
                    return CondBrInst(
                        operands[0], operands[1], operands[2], self.basic_block
                    )
                else:
                    assert False, f"Unsupported branch instruction {instr}"
            case "getelementptr":
                operands = [self.build_operand(operand) for operand in instr.operands]
                type = self.build_type(instr.type)
                return GetElementPtrInst(
                    type, name, operands[0], operands[1:], self.basic_block
                )
            case (
                "add"
                | "sub"
                | "mul"
                | "sdiv"
                | "udiv"
                | "shl"
                | "lshr"
                | "ashr"
                | "and"
                | "or"
                | "xor"
                | "srem"
                | "urem"
                | "fadd"
                | "fsub"
                | "fmul"
                | "fdiv"
            ):
                operands = [self.build_operand(operand) for operand in instr.operands]
                assert len(operands) == 2
                type = self.build_type(instr.type)
                return BinaryOperator(
                    type,
                    name,
                    instr.opcode,
                    operands[0],
                    operands[1],
                    self.basic_block,
                )
            case "bitcast":
                operands = [self.build_operand(operand) for operand in instr.operands]
                type = self.build_type(instr.type)
                return BitCastInst(type, name, operands[0], self.basic_block)
            case "sext":
                type = self.build_type(instr.type)
                operand = self.build_operand(next(instr.operands))
                return SExtInst(type, name, operand, self.basic_block)
            case "zext":
                type = self.build_type(instr.type)
                operand = self.build_operand(next(instr.operands))
                return ZExtInst(type, name, operand, self.basic_block)
            case "fpext":
                type = self.build_type(instr.type)
                operand = self.build_operand(next(instr.operands))
                return FPExtInst(type, name, operand, self.basic_block)
            case "ptrtoint":
                type = self.build_type(instr.type)
                operand = self.build_operand(next(instr.operands))
                return PtrToIntInst(type, name, operand, self.basic_block)
            case "inttoptr":
                type = self.build_type(instr.type)
                operand = self.build_operand(next(instr.operands))
                return IntToPtrInst(type, name, operand, self.basic_block)
            case "uitofp":
                type = self.build_type(instr.type)
                operand = self.build_operand(next(instr.operands))
                return UIToFPInst(type, name, operand, self.basic_block)
            case "sitofp":
                type = self.build_type(instr.type)
                operand = self.build_operand(next(instr.operands))
                return SIToFPInst(type, name, operand, self.basic_block)
            case "trunc":
                type = self.build_type(instr.type)
                operand = self.build_operand(next(instr.operands))
                return TruncInst(type, name, operand, self.basic_block)
            case "fptrunc":
                type = self.build_type(instr.type)
                operand = self.build_operand(next(instr.operands))
                return FPTruncInst(type, name, operand, self.basic_block)
            case "phi":
                type = self.build_type(instr.type)
                blocks = [
                    self._context.get_obj_by_id(self._context.get_id(incoming))
                    for incoming in instr.incoming_blocks
                ]
                operands = [self.build_operand(operand) for operand in instr.operands]
                incoming_values = list(zip(operands, blocks))
                return PhiNode(type, name, incoming_values, self.basic_block)
            case "switch":
                operands = [self.build_operand(operand) for operand in instr.operands]
                condition = operands[0]
                default = operands[1]
                cases = []
                for i in range(2, len(operands), 2):
                    value = operands[i]
                    block = operands[i + 1]
                    cases.append((value, block))
                return SwitchInst(condition, default, cases, self.basic_block)
            case "insertvalue":  # FIXME: not implemented
                operands = [self.build_operand(operand) for operand in instr.operands]
                type = self.build_type(instr.type)
                return InsertValueInst(
                    type, operands[0], operands[1], operands[1:], self.basic_block
                )
            case "select":
                operands = [self.build_operand(operand) for operand in instr.operands]
                type = self.build_type(instr.type)
                return SelectInst(
                    type, name, operands[0], operands[1], operands[2], self.basic_block
                )
            case "unreachable":
                return UnreachableInst(self.basic_block)
            case _:
                assert (
                    False
                ), f"Unsupported instruction {
                    instr} opcode {instr.opcode}"

    def build_type(self, type: TypeRef) -> Type:
        assert isinstance(type, TypeRef)
        match type.type_kind:
            case TypeKind.void:
                return VoidType()
            case TypeKind.pointer:
                return PointerType(type.type_width)
            case TypeKind.integer:
                return IntegerType(type.type_width)
            case TypeKind.array:
                element_count = type.element_count
                types = [self.build_type(element) for element in type.elements]
                assert len(types) == 1
                return ArrayType(element_count, types[0])
            case TypeKind.struct:
                # build struct type is a little bit tricky
                # we need to check if the struct type is already built to avoid infinite recursion
                if not StructType.exists(type.name):
                    struct_type = StructType(type.name)
                    types = [self.build_type(element) for element in type.elements]
                    struct_type.elements = types
                    return struct_type
                else:
                    return StructType.get(type.name)
            case TypeKind.function:
                types = [self.build_type(element) for element in type.elements]
                return_type = types[0]
                arguments = types[1:]
                return FunctionType(return_type, arguments)
            case TypeKind.vector:
                assert False, "Vector type is not supported"
            case TypeKind.metadata:
                return MetadataType()
            case TypeKind.double:
                return DoubleType(type.type_width)
            case TypeKind.float:
                return FloatType(type.type_width)
            case _:
                assert False, f"Unsupported type {type} {type.type_kind}"

    def build_operand(self, ffi_operand) -> Value:
        assert ffi_operand.is_operand
        if ffi_operand.is_constant:
            return self.build_constant(ffi_operand)
        match ffi_operand.value_kind:
            case ValueKind.inline_asm:
                type = self.build_type(ffi_operand.type)
                return InlineAsm(type, str(ffi_operand))
            case ValueKind.argument:
                defined_operand = self.context.get_value(ffi_operand)
                return defined_operand
            case ValueKind.instruction:
                defined_operand = self.context.get_value(ffi_operand)
                return defined_operand
            case ValueKind.basic_block:
                defined_operand = self.context.get_value(ffi_operand)
                return defined_operand
            case _:
                raise ValueError(f"Unsupported operand {ffi_operand}")

    def build_argument(self, ffi_arg) -> Argument:
        assert ffi_arg.is_argument
        name = ffi_arg.name
        type = self.build_type(ffi_arg.type)
        argument = Argument(name, type)
        self.context.save_value(ffi_arg, argument)
        return argument

    def build_constant(self, constant) -> Constant:
        assert constant.is_constant
        type_ = self.build_type(constant.type)
        value = constant.get_constant_value()
        match constant.value_kind:
            case ValueKind.constant_int:
                return ConstantInt(type_, value)
            case ValueKind.constant_expr:
                return Constant(type_, value)
            case ValueKind.function:
                return Constant(type_, value)
            case ValueKind.constant_pointer_null:
                return Null
            case ValueKind.global_variable:
                return GlobalValue(type_, value)
            case _:
                print("build constant", constant)
                breakpoint()

    def build_function_name(self, function: Function):
        """Assign names to unnamed instructions in the function"""
        name_id = 0

        def next_name_id():
            nonlocal name_id
            name_id += 1
            return name_id - 1

        for arg in function.arguments:
            if arg.name == "":
                arg.name = next_name_id()
        for block in function.blocks:
            if block.name == "":
                block.name = next_name_id()
            for instr in block.instructions:
                if instr.has_name() and instr.name == "":
                    instr.name = next_name_id()
