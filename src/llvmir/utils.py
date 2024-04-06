import os
import sys


def get_name_str(x):
    if isinstance(x, int):
        return f"%{x}"
    elif isinstance(x, str):
        return x
    else:
        assert False, f"Unknown type {type(x)}"


def get_library_name():
    """
    Return the name of the llvmlite shared library file.
    """
    if os.name == 'posix':
        if sys.platform == 'darwin':
            return 'libllvmlite.dylib'
        else:
            return 'libllvmlite.so'
    else:
        assert os.name == 'nt'
        return 'llvmlite.dll'


def get_library_files():
    """
    Return the names of shared library files needed for this platform.
    """
    files = [get_library_name()]
    if os.name == 'nt':
        files.extend(['msvcr120.dll', 'msvcp120.dll'])
    return files
