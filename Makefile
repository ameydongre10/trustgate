.PHONY: install test lint type format clean evaluate demo api ui

PYTHON ?= .venv/bin/python
ifeq ($(OS),Windows_NT)
    PYTHON := .venv/Scripts/python.exe
endif

install:
	uv pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest tests/

lint:
	$(PYTHON) -m ruff check src/ tests/

type:
	$(PYTHON) -m mypy src/

format:
	$(PYTHON) -m ruff format src/ tests/
	$(PYTHON) -m ruff check --fix src/ tests/

clean:
	rm -rf .pytest_cache .coverage htmlcov .mypy_cache dist build *.egg-info

evaluate:
	$(PYTHON) -m trustgate.cli evaluate

demo:
	$(PYTHON) -m trustgate.cli demo

api:
	$(PYTHON) -m uvicorn trustgate.api.app:app --host 0.0.0.0 --port 8000 --reload

ui:
	$(PYTHON) -m streamlit run src/trustgate/ui/dashboard.py
