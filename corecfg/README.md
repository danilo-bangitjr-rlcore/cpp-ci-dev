# CoreConfig Service

Configuration management service for the RLTune platform.

## Overview

CoreConfig is responsible for:
- Configuration validation and management
- Live configuration updating
- Configuration access control  
- Audit logging for configuration changes

## Architecture

CoreConfig is part of the RLTune distributed microservices architecture and integrates with:
- **Coredinator**: Service lifecycle management and internal routing
- **CoreTelemetry**: Metrics reporting and logging
- **ConfigDB**: Configuration storage and persistence

## Documentation

For detailed technical specifications, see:
- [Core Technical Specification](../docs/tech_spec.md)
- [Architecture Overview](../docs/diagrams/mvp.md)

## Development

### Requirements
- Python 3.13+
- uv package manager

### Setup
```bash
cd corecfg
uv sync --dev
```

### Testing
```bash
uv run pytest
```

### Code Quality
```bash
uv run ruff check .
uv run pyright
```