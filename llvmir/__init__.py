from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

from .module import Module
from .function import Function
from .block import BasicBlock
from .instruction import *
from .builder import ModuleBuilder
from .type import *
