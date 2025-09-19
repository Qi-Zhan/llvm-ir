import os
import sys


def get_name_str(x):
    if isinstance(x, int):
        return f"%{x}"
    elif isinstance(x, str):
        return x
    assert False, f"Unknown type {type(x)}"


def indent(text: str, n: int = 2) -> str:
    """Indent instructions inside blocks by n spaces, but not the block label."""
    lines = text.splitlines()
    result = []
    for line in lines:
        if line.endswith(":"):  # label 行，不缩进
            result.append(line)
        else:
            result.append(" " * n + line)
    return "\n".join(result)


def get_library_name():
    """
    Return the name of the llvmlite shared library file.
    """
    if os.name == "posix":
        if sys.platform == "darwin":
            return "libllvmlite.dylib"
        else:
            return "libllvmlite.so"
    else:
        assert os.name == "nt"
        return "llvmlite.dll"


def get_library_files():
    """
    Return the names of shared library files needed for this platform.
    """
    files = [get_library_name()]
    if os.name == "nt":
        files.extend(["msvcr120.dll", "msvcp120.dll"])
    return files
