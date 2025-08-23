# CoreCLI

Command line tools for the RL development team.

## Installation

```bash
# Install in development mode
uv sync --dev
```

## Usage

```bash
# Basic hello command
corecli hello

# Get help
corecli --help
```

## Development

This package follows the monorepo standards and patterns. 

### Running Tests

```bash
uv run pytest
```

### Code Quality

```bash
# Run linting
uv run ruff check
uv run pyright

# Format code
uv run ruff format
```