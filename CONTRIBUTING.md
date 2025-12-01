# Contributing to Aircraft Dataset Generator

Thank you for your interest in contributing to the Aircraft Dataset Generator! This document provides guidelines and instructions for contributing to this project.

## Table of Contents

- [Development Setup](#development-setup)
- [Pre-commit Hooks](#pre-commit-hooks)
- [Code Quality Standards](#code-quality-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Code Style Guide](#code-style-guide)

## Development Setup

### Prerequisites

- Python 3.9 or higher
- Git
- Virtual environment (recommended)

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/aircraft-dataset-generator.git
   cd aircraft-dataset-generator
   ```

2. **Create and activate a virtual environment:**

   ```bash
   # Using venv
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate

   # OR using conda
   conda env create -f environment.yml
   conda activate aircraft-dataset
   ```

3. **Install development dependencies:**

   ```bash
   # Install production and development dependencies
   pip install -r requirements-dev.txt

   # Install the package in editable mode
   pip install -e .
   ```

4. **Install pre-commit hooks:**

   ```bash
   pre-commit install
   ```

   This will automatically run code quality checks before each commit.

## Pre-commit Hooks

This project uses [pre-commit](https://pre-commit.com/) to maintain code quality. The hooks run automatically before each commit to ensure code meets our standards.

### What the Hooks Do

Our pre-commit configuration includes hooks that run **automatically** on commit and hooks that run **manually** on demand.

**Automatic Hooks (run on every commit):**

- **File Formatting:**
  - Remove trailing whitespace
  - Ensure files end with a newline
  - Fix mixed line endings
  - Validate YAML, TOML, JSON, and XML syntax

- **Code Formatting:**
  - **Black**: Auto-format Python code to consistent style
  - **isort**: Sort and organize imports

- **Safety Checks:**
  - Detect private keys and merge conflicts
  - Check Python AST validity
  - Detect debug statements

**Manual Hooks (run with `pre-commit run <hook-name> --all-files`):**

- **Type Checking:**
  - **mypy**: Static type checking for type hints

- **Security:**
  - **Bandit**: Check for common security issues

- **Code Quality:**
  - **Ruff**: Fast Python linter with auto-fixes
  - **Flake8**: Comprehensive code style checking

- **Documentation:**
  - **interrogate**: Check docstring coverage
  - **markdownlint**: Lint Markdown files
  - **yamllint**: YAML file linting

Note: Manual hooks are skipped in CI to allow gradual improvement of code quality
without blocking development. They can be run locally to check code quality.

### Manual Hook Execution

```bash
# Run hooks on all files
pre-commit run --all-files

# Run hooks on staged files only
pre-commit run

# Run a specific hook
pre-commit run black --all-files
pre-commit run flake8 --all-files

# Update hook versions
pre-commit autoupdate

# Bypass hooks (use sparingly!)
git commit --no-verify
```

### Handling Hook Failures

If a hook fails:

1. **Auto-fixable issues**: Some hooks (black, isort, ruff) will automatically fix issues. Simply review the changes and commit again.

2. **Manual fixes required**: Read the error message and fix the issue manually:

   ```bash
   # Example: Flake8 error
   # Fix the code issue, then:
   git add <fixed-files>
   git commit
   ```

3. **Type errors**: If mypy reports type errors, add type hints or use `# type: ignore` with a comment explaining why.

## Code Quality Standards

### Code Formatting

We use **Black** with a line length of 100 characters:

```bash
# Format all Python files
black .

# Check what would be formatted without making changes
black --check .
```

### Import Sorting

We use **isort** with Black-compatible settings:

```bash
# Sort imports in all files
isort .

# Check import sorting
isort --check-only .
```

### Linting

We use both **Flake8** and **Ruff** for comprehensive linting:

```bash
# Run Flake8
flake8 aircraft_toolkit tests

# Run Ruff
ruff check .

# Auto-fix Ruff issues
ruff check --fix .
```

### Type Checking

We encourage type hints and use **mypy** for type checking:

```bash
# Run mypy
mypy aircraft_toolkit

# Run mypy on specific files
mypy aircraft_toolkit/core/dataset_3d.py
```

### Security Scanning

We use **Bandit** to check for security issues:

```bash
# Run security scan
bandit -r aircraft_toolkit

# Generate detailed report
bandit -r aircraft_toolkit -f html -o bandit-report.html
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=aircraft_toolkit --cov-report=html

# Run specific test file
pytest tests/test_dataset_2d.py

# Run specific test
pytest tests/test_dataset_2d.py::TestDataset2D::test_basic_generation

# Run tests in parallel (faster)
pytest -n auto

# Run only fast tests (skip slow tests)
pytest -m "not slow"
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use descriptive test names
- Include docstrings for complex tests
- Mark slow tests with `@pytest.mark.slow`

Example:

```python
import pytest
from aircraft_toolkit.core import Dataset2D

class TestDataset2D:
    """Tests for 2D dataset generation."""

    def test_basic_generation(self):
        """Test basic 2D dataset generation with default parameters."""
        dataset = Dataset2D(num_samples=10)
        images, labels = dataset.generate()
        assert len(images) == 10
        assert len(labels) == 10

    @pytest.mark.slow
    def test_large_dataset(self):
        """Test generation of large dataset (marked as slow)."""
        dataset = Dataset2D(num_samples=1000)
        images, labels = dataset.generate()
        assert len(images) == 1000
```

### Coverage Requirements

- Aim for at least 80% code coverage
- All new features should include tests
- Bug fixes should include regression tests

## Pull Request Process

1. **Create a feature branch:**

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes:**
   - Write code following our style guide
   - Add tests for new functionality
   - Update documentation as needed

3. **Ensure all checks pass:**

   ```bash
   # Run pre-commit hooks
   pre-commit run --all-files

   # Run tests
   pytest

   # Check coverage
   pytest --cov=aircraft_toolkit --cov-report=term-missing
   ```

4. **Commit your changes:**

   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

   Use conventional commit messages:
   - `feat:` New feature
   - `fix:` Bug fix
   - `docs:` Documentation changes
   - `style:` Code style changes (formatting, etc.)
   - `refactor:` Code refactoring
   - `test:` Test changes
   - `chore:` Build process or tooling changes

5. **Push to your fork:**

   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request:**
   - Provide a clear description of changes
   - Reference any related issues
   - Ensure CI checks pass

## Code Style Guide

### Python Style

- Follow [PEP 8](https://pep8.org/) (enforced by Black and Flake8)
- Use type hints for function parameters and return values
- Maximum line length: 100 characters
- Use docstrings for all public modules, functions, classes, and methods

### Docstring Format

We use Google-style docstrings:

```python
def generate_dataset(num_samples: int, output_dir: str) -> tuple[list, list]:
    """Generate synthetic aircraft dataset.

    Args:
        num_samples: Number of images to generate.
        output_dir: Directory to save generated images.

    Returns:
        Tuple containing list of images and list of labels.

    Raises:
        ValueError: If num_samples is negative.
        IOError: If output_dir is not writable.

    Example:
        >>> images, labels = generate_dataset(100, "./output")
        >>> print(f"Generated {len(images)} images")
        Generated 100 images
    """
    if num_samples < 0:
        raise ValueError("num_samples must be non-negative")
    # Implementation...
```

### Import Organization

Imports should be organized in the following order (isort handles this automatically):

1. Standard library imports
2. Third-party imports
3. Local application imports

```python
# Standard library
import os
from pathlib import Path

# Third-party
import numpy as np
import pyvista as pv

# Local
from aircraft_toolkit.core import Dataset2D
from aircraft_toolkit.models import F15Eagle
```

### Naming Conventions

- **Classes**: `PascalCase` (e.g., `Dataset2D`, `F15Eagle`)
- **Functions/Methods**: `snake_case` (e.g., `generate_dataset`, `load_model`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_SAMPLES`, `DEFAULT_PATH`)
- **Private members**: Prefix with underscore (e.g., `_internal_method`)

## Questions?

If you have questions or need help, please:

- Open an issue on GitHub
- Check existing documentation
- Review closed issues and pull requests

Thank you for contributing!
