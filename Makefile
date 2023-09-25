
PYTHON := $(shell which python)

# Check types and cleanup codebase
check-types:
	@echo "----------- Type Checking -----------"
	mypy --config-file ./mypy.ini famews/ tests/

format:
	@echo "----------- Code Formatting (black) -----------"
	$(PYTHON) -m black --config famews/pyproject.toml famews tests
	@echo "----------- Import Formatting (isort) -----------"
	$(PYTHON) -m isort --settings-path famews/pyproject.toml famews tests

check-format:
	@echo "----------- Check Code Formatting (black) -----------"
	$(PYTHON) -m black --config famews/pyproject.toml --check famews tests
	@echo "----------- Import Formatting (isort) -----------"
	$(PYTHON) -m isort --settings-path famews/pyproject.toml -c famews tests

test:
	@echo "----------- Run Tests (pytest) -----------"
	$(PYTHON) -m pytest --cov-report term --cov=famews/famews tests/
