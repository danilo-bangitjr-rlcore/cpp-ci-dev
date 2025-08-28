# Internal Development Practices (Confidential)

## Code Quality Standards

### Formatting and Linting
- **Primary Tool**: Ruff
- **Line Length**: 120 characters
- **Import Organization**: Standard, third-party, local

### Type Checking
- **Tool**: Pyright (Microsoft's static type checker)
- **Configuration**: Strict mode enabled
- **Coverage**: 100% type annotation requirement for new code

### Additional Linting
- **Tool**: Pylint with custom rule set
- **Focus**: Code complexity, potential bugs, maintainability

## Testing Strategy

### Test Categories
1. **Small Tests** (`test/small/`): Unit tests, fast execution
2. **Medium Tests** (`test/medium/`): Integration tests with external dependencies
3. **Large Tests** (`test/large/`): End-to-end system tests
4. **Benchmarks** (`test/benchmarks/`): Performance regression tests

### Test Requirements
- **Coverage Target**: 80% minimum for new code
- **Execution Time**: Small tests < 1s, Medium tests < 10s
- **Parallel Execution**: pytest-xdist for faster CI/CD

## Configuration Management

### Validation
- **Schema Validation**: Pydantic models for type safety
- **Runtime Validation**: Configuration health checks
- **Version Control**: Git-tracked configuration files

## Development Workflow

### Feature Development
1. **Branch Creation**: Feature branches from `master`
2. **Development**: High test coverage and fast feedback loops
3. **Code Review**: PR-based review process
4. **Quality Gates**: Automated CI/CD checks
5. **Staging**: Automated deployment to staging

### Commit Standards
- **Format**: Conventional commits format
- **Scope**: Clear scope indicators (feat, fix, docs, etc.)
  - JIRA issue tags or service scopes e.g. `feat(corerl): msg` or `feat(JIRA-123): msg`.
- **Description**: Imperative mood, clear action

## Documentation Standards

### Code Documentation
- **Docstrings**: Generally only used when naming conventions are insufficient
- **Type Hints**: Comprehensive type annotations on inputs, outputs left inferred
  - Aim for no `Any` or `Unknown` in codebase.
- **Comments**: Explain "why" not "what"

### Technical Documentation
- **Architecture**: Mermaid diagrams for system design
- **APIs**: OpenAPI specifications for REST endpoints
- **Deployment**: Deployment guides are part of each service's technical specification, detailing management via `coredinator`.
