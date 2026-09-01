.PHONY: test lint format docker-build docker-up docker-down clean help

help:
	@echo "Available targets:"
	@echo "  test         - Run tests with pytest"
	@echo "  lint         - Run ruff linter"
	@echo "  format       - Auto-fix linting issues"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-up    - Start services with docker compose"
	@echo "  docker-down  - Stop services"
	@echo "  clean        - Remove cache files"
	@echo "  ci           - Run full CI pipeline (lint + test)"

test:
	uv run pytest tests/ -v --tb=short

lint:
	uv run ruff check .

format:
	uv run ruff check . --fix

docker-build:
	docker build -t rag-project .

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down -v

clean:
	rm -rf .pytest_cache .ruff_cache __pycache__ src/__pycache__ src/services/__pycache__ src/routers/__pycache__ tests/__pycache__ scripts/__pycache__ .venv

ci: lint test