.PHONY: all build run clean

current_dir := $(shell dirname $(abspath $(lastword $(MAKEFILE_LIST))))
PY_EXEC=python
FLAGS=-Wall -Wextra -Wconversion -pedantic
EXT_SUFFIX=$(shell python-config --extension-suffix)

FLAGS=-Wall -Wextra -Wconversion -pedantic
NP_MODULE=numpy
ROLLOUT_MODULE_NAME=fast_rollout
F_PROP_TRAJ_MODULE_NAME=forward_prop_traj
F2PY=f2py
SRC_ROLLOUT=rollout.f90
SRC_F_PROP_J=$(F_PROP_TRAJ_MODULE_NAME).pyx

TARGET_ROLLOUT=$(ROLLOUT_MODULE_NAME)$(EXT_SUFFIX)
TARGET_FORWARD_PROP_TRAJ=$(F_PROP_TRAJ_MODULE_NAME)$(EXT_SUFFIX)

all: $(TARGET_ROLLOUT) $(TARGET_FORWARD_PROP_TRAJ)

$(TARGET_ROLLOUT):
	$(PY_EXEC) -m $(NP_MODULE).$(F2PY) -c $(SRC_ROLLOUT) -m $(ROLLOUT_MODULE_NAME)

$(TARGET_FORWARD_PROP_TRAJ): $(SRC_F_PROP_J)
	$(PY_EXEC) setup.py build_ext --inplace

run: all
	$(PY_EXEC) $(current_dir)/execute.py

clean:
	rm -f *.so
