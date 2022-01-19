\PYTHON = python3
.PHONY = all help clean build run
.DEFAULT_GOAL = help

help:
	@echo "----------HELP----------"
	@echo "HPC Lab1 Makefile."
	@echo "------------------------"

clean:
	@echo "----------CLEANING----------"
	rm CythonWrapper.cpp CythonWrapper.cpython-38-x86_64-linux-gnu.so && rm -rf build
	@echo "---------------------------"

build:
	@echo "----------BUILD CUDA FILES----------"
	nvcc -Xcompiler /MD -lib -odir lib -o lib/libmatmul.lib cuda/matmul.cu
	@echo "---------------------------"
	@echo "----------BUILD CYTHON FILES----------"
	python setup.py build_ext --inplace
	@echo "---------------------------"

run:
	@echo "----------RUN----------"
	python Main.py
	@echo "-------------------------"


all: build run

