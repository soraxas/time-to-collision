# SOURCE=$(wildcard *.f*)
# FLAGS=-Wall -Wextra -Wconversion -pedantic

# a.out: $(SOURCE)
# 	gfortran $(FLAGS) -g $< -o a.out

# clean:
# 	rm -rf a.out

# SOURCE=$(wildcard *.f*)

.PHONEY=run clean

SOURCE=rollout.f
FLAGS=-Wall -Wextra -Wconversion -pedantic
MODULE_NAME=fast_rollout

FORTRAN_MODULE=fast_rollout.cpython-37m-x86_64-linux-gnu.so

run: $(FORTRAN_MODULE)
	@./execute.py

$(FORTRAN_MODULE): rollout.f
	python -m numpy.f2py -c rollout.f -m $(MODULE_NAME)

clean:
	rm -rf a.out
