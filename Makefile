.PHONY: help prepare-dev test doc
.DEFAULT: help

help:
	@echo "make prepare-dev"
	@echo "       create and prepare development environment, use only once"
	@echo "make test"
	@echo "       run tests and linting on py38"
	@echo "make doc"
	@echo "       build Sphinx docs documentation"
	@echo "make check_all"
	@echo "       check all files using pre-commit tool"
	@echo "make updatetools"
	@echo "       updatetools pre-commit tool"

prepare-dev:
	python3 -m venv puncc-dev-env
	. puncc-dev-env/bin/activate && python -m pip install --upgrade pip
	. puncc-dev-env/bin/activate && python -m pip install -e .[dev]
	. puncc-dev-env/bin/activate && pre-commit install
	. puncc-dev-env/bin/activate && pre-commit install-hooks

install-user:
	python3 -m venv puncc-user-env
	. puncc-user-env/bin/activate && python -m pip install --upgrade pip
	. puncc-user-env/bin/activate && python -m pip install -e .[interactive]
	. puncc-user-env/bin/activate && python -m ipykernel install --user --name=puncc-user-env

check_all:
	. puncc-env/bin/activate && pre-commit run --all-files

test:
	. puncc-dev-env/bin/activate && tox

doc:
	. puncc-dev-env/bin/activate && cd docs && make clean html && cd -

updatetools:
	. puncc-dev-env/bin/activate && pre-commit autoupdate
