.PHONY: install test clean build

install: 
	pip install -e .

build:
	python -m build

test:
	python -m unittest

clean:
	rm -rf __pycache__
	rm -rf tests/__pycache__
	rm -rf src/llvmir/__pycache__
	rm -rf build
	rm -rf dist
	rm -rf src/llvmir.egg-info
	rm -rf src/llvmir/llvmir.egg-info
	pip uninstall -y llvmir