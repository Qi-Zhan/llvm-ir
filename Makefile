.PHONY: install clean

install: 
	pip install -e .

clean:
	rm -rf __pycache__
	rm -rf tests/__pycache__
	rm -rf src/llvmir/__pycache__
	rm -rf src/llvmir/analysis/__pycache__
	rm -rf src/llvmir/binding/__pycache__
	rm -rf build
	rm -rf dist
	rm -rf src/llvmir.egg-info
	rm -rf src/llvmir/llvmir.egg-info
	pip uninstall -y llvmir