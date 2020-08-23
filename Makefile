.PHONEY=build run clean

SOURCE=$(wildcard *.f90)
FLAGS=-Wall -Wextra -Wconversion -pedantic

SOURCE=rollout.f90
FLAGS=-Wall -Wextra -Wconversion -pedantic
EXEC=python
NP_MODULE=numpy
MODULE_NAME=fast_rollout
F2PY=f2py

build:
	$(EXEC) -m $(NP_MODULE).$(F2PY) -c $(SOURCE) -m $(MODULE_NAME)

run: build
	@./execute.py

clean:
	rm -rf *.so
