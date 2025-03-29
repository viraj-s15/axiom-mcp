# Axiom MCP

Model Context Protocol (MCP) implementation for connecting AI systems with external data sources.

## Installation

Using uv (recommended):
```bash
uv pip install axiom-mcp
```

Using pip:
```bash
pip install axiom-mcp
```

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/axiom-mcp.git
   cd axiom-mcp
   ```

2. Install uv if you haven't already:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. Create and activate a virtual environment with uv:
   ```bash
   uv venv
   source .venv/bin/activate  # On Unix/Linux
   # or
   .venv\Scripts\activate  # On Windows
   ```

4. Install development dependencies:
   ```bash
   uv pip install -e ".[dev]"
   ```

5. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Pre-commit Hooks

This project uses pre-commit hooks to ensure code quality. The following checks are run automatically before each commit:

- Code formatting with `black`
- Import sorting with `isort`
- Type checking with `mypy`
- Linting with `ruff`
- Basic syntax checks
- Check for large files
- Check for merge conflicts

To manually run all pre-commit hooks:
```bash
uv run pre-commit run --all-files
```

## Running Tests

```bash
uv run python -m pytest
```

For tests with coverage report:
```bash
uv run python -m pytest --cov=axiom_mcp tests/
```

## Code Quality

This project enforces high code quality standards using:

- `black` for consistent code formatting
- `isort` for import sorting
- `mypy` for static type checking
- `ruff` for fast Python linting
- `pytest` for testing with coverage reporting

To run all quality checks:
```bash
black .
isort .
mypy .
ruff .
pytest
```

### Using uv for Fast Dependencies Management

uv provides faster package installation and dependency resolution. Some useful commands:

```bash
# Update dependencies
uv pip compile pyproject.toml -o requirements.txt

# Sync your environment with requirements
uv pip sync requirements.txt

# Add a new dependency
uv pip install package-name
```

## Contributing

1. Fork the repository
2. Create a new branch for your feature
3. Make your changes
4. Run all quality checks:
   ```bash
   pre-commit run --all-files
   pytest
   ```
5. Submit a pull request

## License
GNU General Public License v3 (GPLv3)
