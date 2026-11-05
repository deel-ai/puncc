.PHONY: help prepare-dev install-user test test-all \
        test-numpy test-tensorflow test-torch test-jax \
        doc check_all updatetools

.DEFAULT_GOAL := help

PYTHON ?= python
BACKEND ?= numpy

help:
	@echo "make prepare-dev"
	@echo "       create and prepare development environment"
	@echo "make install-user"
	@echo "       create a user environment for notebooks"
	@echo "make test"
	@echo "       run tests with BACKEND (default: numpy)"
	@echo "make test-all"
	@echo "       run tests on all four backends using tox"
	@echo "make test-numpy"
	@echo "       run tests on NumPy using tox"
	@echo "make test-tensorflow"
	@echo "       run tests on TensorFlow using tox"
	@echo "make test-torch"
	@echo "       run tests on PyTorch using tox"
	@echo "make test-jax"
	@echo "       run tests on JAX using tox"
	@echo "make doc"
	@echo "       build Sphinx documentation"
	@echo "make check_all"
	@echo "       check all files using pre-commit"
	@echo "make updatetools"
	@echo "       update pre-commit hooks"

prepare-dev:
	$(PYTHON) -m venv puncc-dev-env
	. puncc-dev-env/bin/activate && python -m pip install --upgrade pip
	. puncc-dev-env/bin/activate && python -m pip install -e ".[dev,numpy,sklearn]"
	. puncc-dev-env/bin/activate && pre-commit install
	. puncc-dev-env/bin/activate && pre-commit install-hooks

install-user:
	$(PYTHON) -m venv puncc-user-env
	. puncc-user-env/bin/activate && python -m pip install --upgrade pip
	. puncc-user-env/bin/activate && python -m pip install -e ".[interactive,numpy]"
	. puncc-user-env/bin/activate && python -m ipykernel install --user --name=puncc-user-env

check_all:
	. puncc-dev-env/bin/activate && pre-commit run --all-files

test:
	KERAS_BACKEND=$(BACKEND) $(PYTHON) -m pytest -q tests/ --tb=short -ra

test-all:
	$(PYTHON) -m tox

test-numpy:
	$(PYTHON) -m tox -e backend-numpy

test-tensorflow:
	$(PYTHON) -m tox -e backend-tensorflow

test-torch:
	$(PYTHON) -m tox -e backend-torch

test-jax:
	$(PYTHON) -m tox -e backend-jax

doc:
	. puncc-dev-env/bin/activate && $(MAKE) -C docs clean html

updatetools:
	. puncc-dev-env/bin/activate && pre-commit autoupdate
