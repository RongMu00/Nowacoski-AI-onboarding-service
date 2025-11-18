.PHONY: help install test lint format clean run build docker-build docker-run k8s-deploy

help: ## Show this help message
	@echo 'Usage: make [target] ...'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install dependencies
	uv venv --python 3.13
	source .venv/bin/activate && uv pip install -e ".[dev]"
	source .venv/bin/activate && pre-commit install

test: ## Run tests
	uv run pytest --cov=src --cov-report=term-missing --cov-report=html

test-fast: ## Run tests without coverage
	uv run pytest -x -v

lint: ## Run linting
	uv run black --check src tests
	uv run isort --check-only src tests
	uv run flake8 src tests
	uv run mypy src

format: ## Format code
	uv run black src tests
	uv run isort src tests

clean: ## Clean up generated files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf htmlcov
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info

run: ## Run the development server
	uv run uvicorn enterprise_ai.main:app --reload --host 0.0.0.0 --port 8000

run-prod: ## Run the production server
	uv run uvicorn enterprise_ai.main:app --host 0.0.0.0 --port 8000

build: ## Build the package
	uv build

docker-build: ## Build Docker image
	docker build -f docker/Dockerfile -t enterprise-ai:latest .

docker-run: ## Run Docker container
	docker run -p 8000:8000 enterprise-ai:latest

docker-compose-up: ## Start services with docker-compose
	docker-compose -f docker/docker-compose.yml up -d

docker-compose-down: ## Stop services with docker-compose
	docker-compose -f docker/docker-compose.yml down

k8s-deploy: ## Deploy to Kubernetes
	kubectl apply -f k8s/

k8s-delete: ## Delete from Kubernetes
	kubectl delete -f k8s/

k8s-status: ## Check Kubernetes deployment status
	kubectl get pods -l app=enterprise-ai
	kubectl get services
	kubectl get ingress

check: lint test ## Run all checks (lint and test)

ci: install check ## Run CI pipeline locally
