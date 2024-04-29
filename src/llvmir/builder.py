import logging

from .module import Module
from .function import Function, Argument
from .block import BasicBlock
from .instruction import *
from .typed import *
from .value import Value
from .constant import *

from .binding.module import ModuleRef, parse_bitcode, parse_assembly
from .binding.typeref import TypeRef, TypeKind
from .binding.passmanagers import create_module_pass_manager
from .binding.value import ValueKind

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def needs_name(ffi_obj):
    if ffi_obj.name != '':
        return False
    opcodes = ['store', 'fence', 'br', 'ret', 'switch',
               'unreachable', 'indirectbr', 'resume', 'cleanupret', 'catchret']

    if ffi_obj.is_instruction:
        opcode = ffi_obj.opcode
        if opcode == 'call':
            return str(ffi_obj.type) != 'void'
        if opcode in opcodes:
            return False
    return True


class FunctionContext:
    def __init__(self):
        self._counter = 0
        self._str2id = {}
        self._id2obj = {}

    def set_id(self, ffi_object):
        """ Save the ffi object/str and return an unique id as its name """
        self._str2id[str(ffi_object)] = self._counter
        self._counter += 1

    def get_id(self, ffi_object):
        return self._str2id[str(ffi_object)]

    def save_object(self, object: Value):
        if object.has_name():
            self._id2obj[object.get_name()] = object

    def get_obj_by_id(self, name) -> Value:
        return self._id2obj[name]


class ModuleBuilder:
    def __init__(self):
        self._module = None
        self._function = None
        self._basic_block = None
        self._counter = 0
        # matain a set of already built objects
        self._already_built = {}

    @staticmethod
    def from_path(path: str, run_mem2reg=False, run_instnamer=False) -> Module:
        if path.endswith(".bc"):
            return ModuleBuilder.from_bc_path(path, run_mem2reg, run_instnamer)
        elif path.endswith(".ll"):
            return ModuleBuilder.from_ir_path(path, run_mem2reg, run_instnamer)
        else:
            raise ValueError("Invalid file type")

    @staticmethod
    def from_bc_path(path: str, run_mem2reg=False, run_instnamer=False) -> Module:
        with open(path, "rb") as f:
            bitcode = f.read()
        ffi_module = parse_bitcode(bitcode)
        return ModuleBuilder._from_ffi_module(ffi_module, run_mem2reg, run_instnamer)

    @staticmethod
    def from_ir_path(path: str, run_mem2reg=False, run_instnamer=False) -> Module:
        with open(path, "r") as f:
            ir = f.read()
        ffi_module = parse_assembly(ir)
        return ModuleBuilder._from_ffi_module(ffi_module, run_mem2reg, run_instnamer)

    @staticmethod
    def _from_ffi_module(ffi_module: ModuleRef, run_mem2reg, run_instnamer) -> Module:
        """ Translate ModuleRef to Module

        Parameters:
        mod: ModuleRef, ffi object

        Returns:
        Module, python data structure
        """
        transformer = ModuleBuilder()
        pm = create_module_pass_manager()
        if run_mem2reg:
            pm.add_memcpy_optimization_pass()
        if run_instnamer:
            pm.add_instruction_namer_pass()
        pm.run(ffi_module)
        return transformer.build_mod(ffi_module)

    @staticmethod
    def from_pickle(path: str) -> Module:
        import pickle
        with open(path, "rb") as f:
            mod = pickle.load(f)
        return mod

    def build_mod(self, ffi_mod) -> Module:
        source_file = ffi_mod.source_file
        name = ffi_mod.name
        triple = ffi_mod.triple
        data_layout = ffi_mod.data_layout
        struct_types = ffi_mod.struct_types
        modules = Module(source_file, name, triple, data_layout, struct_types)
        self._module = modules
        for ffi_func in ffi_mod.functions:
            assert (ffi_func.is_function)
            function = self.build_function(ffi_func)
            if ffi_func.is_declaration:
                modules.declarations[function.name] = function
            else:
                modules.functions[function.name] = function
        return modules

    def build_function(self, func) -> Function:
        function = Function(func.name, self._module)
        self._context = self.collect_names(func)
        arguments = [self.build_argument(arg) for arg in func.arguments]
        for arg in arguments:
            self._context.save_object(arg)
        function.arguments = arguments
        logger.debug(f'function {func.name}')
        logger.debug(f'arguments {arguments}')
        self._function = function
        # build basic blocks first
        basic_blocks = []
        for bb in func.blocks:
            assert (bb.is_block)
            name = self._context.get_id(bb) if needs_name(bb) else bb.name
            logger.debug(f'basic block {name}')
            basic_block = BasicBlock(name, self._function)
            self._context.save_object(basic_block)
            basic_blocks.append(basic_block)
        # build instructions
        for (i, ffi_bb) in enumerate(func.blocks):
            # give ffi_bb and our basic_block
            basic_block = self.build_basicblock(ffi_bb, basic_blocks[i])
        function.blocks = basic_blocks
        return function

    def collect_names(self, func) -> FunctionContext:
        # each function has its own context
        context = FunctionContext()
        for arg in func.arguments:
            if needs_name(arg):
                context.set_id(arg)
        for bb in func.blocks:
            if needs_name(bb):
                context.set_id(bb)
            for instr in bb.instructions:
                if needs_name(instr):
                    context.set_id(instr)
        return context

    def build_basicblock(self, bb, basic_block) -> BasicBlock:
        assert (bb.is_block)
        self._basic_block = basic_block
        for ffi_instr in bb.instructions:
            instruction = self.build_instruction(ffi_instr)
            if instruction is None:  # skip debug info
                continue
            logger.debug(f'get instruction {instruction}')
            self._context.save_object(instruction)
            basic_block.instructions.append(instruction)
        return basic_block

    def build_instruction(self, instr) -> Instruction:
        assert (instr.is_instruction)
        name = self._context.get_id(instr) if needs_name(instr) else instr.name
        match instr.opcode:
            case "alloca":
                type = self.build_type(instr.type)
                return AllocaInst(type, name, self._basic_block)
            case "store":
                operands = [self.build_operand(operand)
                            for operand in instr.operands]
                assert len(operands) == 2
                return StoreInst(operands[0], operands[1], self._basic_block)
            case "load":
                operand = self.build_operand(next(instr.operands))
                type = self.build_type(instr.type)
                return LoadInst(type, name, operand, self._basic_block)
            case "icmp":
                operands = [self.build_operand(operand)
                            for operand in instr.operands]
                type = self.build_type(instr.type)
                predicate = instr.predicate
                return ICmpInst(type, name, predicate, operands[0], operands[1], self._basic_block)
            case "ret":
                operands = [self.build_operand(operand)
                            for operand in instr.operands]
                if len(operands) == 0:
                    return ReturnInst(None, self._basic_block)
                else:
                    return ReturnInst(operands[0], self._basic_block)
            case "call":
                if "@llvm.dbg" in str(instr):
                    return
                type = self.build_type(instr.type)
                if type != VoidType():
                    operands = [self.build_operand(operand)
                                for operand in instr.operands]
                    return CallInst(type, name, instr.called_value.name, operands[:-1], self._basic_block)
                else:
                    operands = [self.build_operand(operand)
                                for operand in instr.operands]
                    return CallInst(type, name, instr.called_value.name, operands[:-1], self._basic_block)
            case "br":
                operands = [self.build_operand(operand)
                            for operand in instr.operands]
                if len(operands) == 1:
                    return BranchInst(operands[0], self._basic_block)
                elif len(operands) == 3:
                    return CondBrInst(operands[0], operands[1], operands[2], self._basic_block)
                else:
                    assert False, f"Unsupported branch instruction {instr}"
            case "getelementptr":
                operands = [self.build_operand(operand)
                            for operand in instr.operands]
                type = self.build_type(instr.type)
                return GetElementPtrInst(type, name, operands[0], operands[1:], self._basic_block)
            case "add" | "sub" | "mul" | "sdiv" | "udiv" | "shl" | "lshr" | "ashr" | "and" | "or" | "xor" | "srem" | "urem" | "fadd" | "fsub" | "fmul" | "fdiv":
                operands = [self.build_operand(operand)
                            for operand in instr.operands]
                assert len(operands) == 2
                type = self.build_type(instr.type)
                return BinaryOperator(type, name, instr.opcode, operands[0], operands[1], self._basic_block)
            case "bitcast":
                operands = [self.build_operand(operand)
                            for operand in instr.operands]
                type = self.build_type(instr.type)
                return BitCastInst(type, name, operands[0], self._basic_block)
            case "sext":
                type = self.build_type(instr.type)
                operand = self.build_operand(next(instr.operands))
                return SExtInst(type, name, operand, self._basic_block)
            case "zext":
                type = self.build_type(instr.type)
                operand = self.build_operand(next(instr.operands))
                return ZExtInst(type, name, operand, self._basic_block)
            case "fpext":
                type = self.build_type(instr.type)
                operand = self.build_operand(next(instr.operands))
                return FPExtInst(type, name, operand, self._basic_block)
            case "ptrtoint":
                type = self.build_type(instr.type)
                operand = self.build_operand(next(instr.operands))
                return PtrToIntInst(type, name, operand, self._basic_block)
            case "inttoptr":
                type = self.build_type(instr.type)
                operand = self.build_operand(next(instr.operands))
                return IntToPtrInst(type, name, operand, self._basic_block)
            case "uitofp":
                type = self.build_type(instr.type)
                operand = self.build_operand(next(instr.operands))
                return UIToFPInst(type, name, operand, self._basic_block)
            case "sitofp":
                type = self.build_type(instr.type)
                operand = self.build_operand(next(instr.operands))
                return SIToFPInst(type, name, operand, self._basic_block)
            case "trunc":
                type = self.build_type(instr.type)
                operand = self.build_operand(next(instr.operands))
                return TruncInst(type, name, operand, self._basic_block)
            case "fptrunc":
                type = self.build_type(instr.type)
                operand = self.build_operand(next(instr.operands))
                return FPTruncInst(type, name, operand, self._basic_block)
            case "phi":
                type = self.build_type(instr.type)
                blocks = [self._context.get_obj_by_id(self._context.get_id(
                    incoming)) for incoming in instr.incoming_blocks]
                operands = [self.build_operand(operand)
                            for operand in instr.operands]
                incoming_values = list(zip(operands, blocks))
                return PhiNode(type, name, incoming_values, self._basic_block)
            case "switch":
                operands = [self.build_operand(operand)
                            for operand in instr.operands]
                condition = operands[0]
                default = operands[1]
                cases = []
                for i in range(2, len(operands), 2):
                    value = operands[i]
                    block = operands[i + 1]
                    cases.append((value, block))
                return SwitchInst(condition, default, cases, self._basic_block)
            case "insertvalue":  # FIXME: not implemented
                operands = [self.build_operand(operand)
                            for operand in instr.operands]
                type = self.build_type(instr.type)
                return InsertValueInst(type, operands[0], operands[1], operands[1:], self._basic_block)
            case "select":
                operands = [self.build_operand(operand)
                            for operand in instr.operands]
                type = self.build_type(instr.type)
                return SelectInst(type, name, operands[0], operands[1], operands[2], self._basic_block)
            case "unreachable":
                return UnreachableInst(self._basic_block)
            case _:
                assert False, f"Unsupported instruction {
                    instr} opcode {instr.opcode}"

    def build_type(self, type: TypeRef) -> Type:
        assert isinstance(type, TypeRef)
        match type.type_kind:
            case TypeKind.void:
                return VoidType()
            case TypeKind.pointer:
                return PointerType(type.type_width, self.build_type(type.element_type))
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
                    types = [self.build_type(element)
                             for element in type.elements]
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

    def build_operand(self, operand) -> Value:
        assert operand.is_operand
        if operand.is_constant:
            return self.build_constant(operand)
        match operand.value_kind:
            case ValueKind.inline_asm:
                type = self.build_type(operand.type)
                return InlineAsm(type, str(operand))
            case ValueKind.instruction | ValueKind.argument | ValueKind.basic_block:
                name = self._context.get_id(str(operand)) if needs_name(
                    operand) else operand.name
                define_obj = self._context.get_obj_by_id(name)
                return define_obj
            case _:
                raise ValueError(f"Unsupported operand {operand}")

    def build_argument(self, arg) -> Argument:
        assert arg.is_argument
        name = self._context.get_id(arg) if needs_name(arg) else arg.name
        type = self.build_type(arg.type)
        return Argument(name, type)

    def build_constant(self, constant) -> Constant:
        assert constant.is_constant
        type = self.build_type(constant.type)
        value = constant.get_constant_value()
        match constant.value_kind:
            case ValueKind.constant_int:
                return ConstantInt(type, value)
            case ValueKind.constant_expr:
                return Constant(type, value)
            case ValueKind.function:
                # TODO: it is strange that function is a constant
                return Constant(type, value)
            case ValueKind.constant_pointer_null:
                return Null
            case _:
                print('build constant', constant)
                breakpoint()

    @property
    def module(self):
        return self._module

    @property
    def function(self):
        return self._function

    @property
    def basic_block(self):
        return self._basic_block
