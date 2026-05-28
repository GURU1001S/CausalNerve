.PHONY: install test smoke

install:
	pip install -e .[all]

test:
	pytest tests/ -v

smoke:
	python tests/test_library_smoke.py
