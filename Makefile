.PHONY: setup test lint verify-data

setup:
	uv sync

test:
	uv run pytest

lint:
	uv run ruff check .

verify-data:
	uv run python scripts/verify_data.py
