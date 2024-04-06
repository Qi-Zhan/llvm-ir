# llvm-ir

A Python library for parsing and analyzing existing LLVM IR.

-----------------------------------------------------------

The [llvmlite](https://github.com/numba/llvmlite) project provides a lightweight LLVM API for Python, which includes a simple Python binding for writing the LLVM IR builder and JIT compiler.
However, they do not provide a way to transform existing LLVM IR code to Python data structures.

Our project does not aim to provide a way to generate LLVM IR code, but to analyze existing LLVM IR code.
Our project bases on the `llvmlite` project, mainly its `binding` module and ffi.

## Key Benefits

* The instruction supports `match` method to match the instruction with a pattern.
* Some built-in analysis.
* Same to llvmlite, Most of llvmlite uses the LLVM C API which is small but very stable
  (low maintenance when changing LLVM version).

### Compatibility

llvmlite has been tested with Python 3.12 and is likely to work with greater versions.

| llvm-ir versions | compatible LLVM versions |
|-------------------|--------------------------|
| 0.0.1            | 14.x.x                   |

## Example

## Development

Make sure you have the LLVM development libraries installed.
On MacOS, you can install them with:

    $ brew install llvm
  
$ export LLVM_CONFIG=/usr/local/opt/llvm/bin/llvm-config

Then install the package in development mode:

    $ pip install -e .
