# Enterprise AI Challenge

A modern Python 3.13 project template with uv package management and Kubernetes deployment support.

## Features

- **Python 3.13** - Latest Python version
- **uv** - Fast Python package manager
- **FastAPI** - Modern web framework for building APIs
- **Pydantic** - Data validation using Python type hints
- **Kubernetes** - Container orchestration for EC2 deployment
- **Docker** - Containerization
- **Testing** - pytest with coverage
- **Code Quality** - black, isort, flake8, mypy
- **Pre-commit hooks** - Automated code quality checks

## Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- Docker
- kubectl (for Kubernetes deployment)

## Installation

1. Install uv if you haven't already:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Create a virtual environment and install dependencies:
```bash
uv venv --python 3.13
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"
```

3. Install pre-commit hooks:
```bash
pre-commit install
```

## Development

### Running the application
```bash
# Development server
uv run uvicorn enterprise_ai.main:app --reload

# Or using the CLI entry point
uv run enterprise-ai
```

### Code Quality
```bash
# Format code
uv run black src tests
uv run isort src tests

# Lint code
uv run flake8 src tests
uv run mypy src

# Run all quality checks
make lint
```

### Testing
```bash
# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=src --cov-report=html

# Run specific test categories
uv run pytest -m "not slow"  # Skip slow tests
uv run pytest -m integration  # Run only integration tests
```

## Project Structure

```
enterprise-ai-challenge-25/
├── src/
│   └── enterprise_ai/
│       ├── __init__.py
│       ├── main.py
│       ├── api/
│       │   ├── __init__.py
│       │   └── routes.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── config.py
│       │   └── logging.py
│       └── models/
│           ├── __init__.py
│           └── schemas.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_main.py
│   └── api/
│       └── test_routes.py
├── k8s/
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── ingress.yaml
│   └── configmap.yaml
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── .github/
│   └── workflows/
│       └── ci.yml
├── pyproject.toml
├── uv.lock
├── Makefile
├── .gitignore
├── .pre-commit-config.yaml
└── README.md
```

## Deployment

### Docker
```bash
# Build image
docker build -f docker/Dockerfile -t enterprise-ai:latest .

# Run container
docker run -p 8000:8000 enterprise-ai:latest
```

### Kubernetes on EC2
```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -l app=enterprise-ai
kubectl get services
```

### Environment Variables

Set the following environment variables for production:

- `ENV`: Environment (development, staging, production)
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `DATABASE_URL`: Database connection string (if applicable)
- `API_KEY`: API key for external services (if applicable)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and quality checks (`make test lint`)
5. Commit your changes (`git commit -m 'Add some amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
