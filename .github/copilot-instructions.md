# Core-RL Monorepo Development Instructions

**ALWAYS reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.**

Core-RL is a reinforcement learning platform with multiple microservices including the agent (corerl), I/O service (coreio), coordination service (coredinator), and web UI (coreui). The system uses Docker for deployment and PostgreSQL/TimescaleDB for data storage.

## Working Effectively

### Initial Setup - VALIDATED COMMANDS

1. **Install uv Python package manager:**
   ```bash
   pip install uv
   export PATH="$HOME/.local/bin:$PATH"
   ```

2. **Setup any project component (replace PROJECT with corerl/coreio/coredinator/coreui/server):**
   ```bash
   cd PROJECT/
   uv sync --all-extras --dev --frozen
   ```
   - **NEVER CANCEL: Setup takes 30 seconds to 5 minutes.** Set timeout to 10+ minutes.
   - First time setup downloads Python 3.13.0 and all dependencies.
   - Subsequent setups are faster (~30 seconds) due to package reuse.

3. **Install Node.js dependencies for coreui frontend:**
   ```bash
   cd coreui/client/
   npm install
   ```
   - Takes about 20 seconds. Set timeout to 2+ minutes.

### Linting and Type Checking - VALIDATED COMMANDS

**ALWAYS run these before committing or CI will fail:**

1. **For Python projects (corerl, coreio, coredinator, coreui/server):**
   ```bash
   cd PROJECT/
   uv run ruff check .           # Takes 5-10 seconds
   uv run pyright               # Takes 5-10 seconds  
   ```

2. **For coreui frontend:**
   ```bash
   cd coreui/client/
   npm run lint                 # Takes 5-10 seconds
   npm run format:check         # Takes 5-10 seconds
   ```

### Testing - VALIDATED TIMINGS

**NEVER CANCEL these commands. Wait for completion:**

1. **CoreRL tests (from corerl/ directory):**
   ```bash
   uv run pytest tests/small                                    # ~50 seconds
   uv run pytest tests/medium                                   # ~3 minutes  
   uv run pytest -n auto --dist loadscope tests/large          # ~7 minutes (parallel)
   uv run pytest ../test/test/test_validate_configs.py         # ~6 seconds
   uv run pytest --benchmark-only tests/benchmarks/ --benchmark-json output.json  # ~2 minutes
   ```
   - **NEVER CANCEL: Set timeout to 15+ minutes for large tests, 10+ minutes for medium tests.**
   - Large tests run in parallel but still take 7+ minutes.
   - Medium tests take the longest per test due to ML model training.

2. **All tests together (as CI does):**
   ```bash
   pytest --cov=. --cov-append tests/small
   pytest --cov=. --cov-append tests/medium  
   pytest -n auto --dist loadscope --cov=. --cov-append tests/large
   pytest ../test/test/test_validate_configs.py
   ```
   - **NEVER CANCEL: Total runtime ~11 minutes. Set timeout to 20+ minutes.**

### Building - VALIDATED COMMANDS

1. **Build coreui frontend:**
   ```bash
   cd coreui/client/
   npm run build                # ~7 seconds
   ```

2. **Build coreui for development (starts both frontend and backend):**
   ```bash
   cd coreui/
   python build.py dev          # Starts Vite + FastAPI servers
   ```

### Running Applications - VALIDATED SCENARIOS

1. **Start database for testing:**
   ```bash
   docker compose up timescale-db -d        # ~15 seconds first time (pulls image)
   docker compose ps                        # Check if healthy
   ```

2. **Run CoreRL main application:**
   ```bash
   cd corerl/
   uv run python -m corerl.main --config-name CONFIG_NAME
   ```
   - Available configs in `config/` directory: `mountain_car_continuous`, `dep_mountain_car_continuous`, etc.
   - Application validates config on startup.

3. **Start full system:**
   ```bash
   docker compose up                        # Starts all services
   docker compose up --profile test         # Includes test services
   ```

## Validation Scenarios

**ALWAYS test these scenarios after making changes:**

1. **Basic functionality test:**
   ```bash
   cd corerl/
   uv sync --dev --frozen
   uv run ruff check .
   uv run pyright  
   uv run pytest tests/small
   ```

2. **Application startup test:**
   ```bash
   docker compose up timescale-db -d
   cd corerl/
   uv run python -m corerl.main --config-name mountain_car_continuous --help
   docker compose down
   ```

3. **UI build test:**
   ```bash
   cd coreui/client/
   npm install
   npm run build
   npm run lint
   ```

## Repository Structure

### Key Projects

- **corerl/**: Main RL agent and algorithms
- **coreio/**: I/O service for external system communication  
- **coredinator/**: Coordination and orchestration service
- **coreui/**: Web interface (React frontend + FastAPI backend)
- **libs/**: Shared libraries (lib_agent, lib_config, lib_sql, lib_utils, rl_env)
- **config/**: Configuration files for different environments
- **test/**: Integration and system tests

### Important Files

- `compose.yaml`: Docker services configuration
- `config/*.yaml`: Environment-specific configurations
- `.github/workflows/`: CI/CD pipelines for each component
- Each project has its own `pyproject.toml` with dependencies

## Timing Expectations - CRITICAL

**NEVER CANCEL any of these operations. They may appear to hang but are working:**

- **Initial uv sync**: 4-5 minutes (downloads Python + dependencies)
- **Subsequent uv sync**: 30 seconds (reuses cached packages)  
- **Small tests**: 50 seconds
- **Medium tests**: 3 minutes (ML training involved)
- **Large tests**: 7 minutes (even with parallelization)
- **Benchmarks**: 2 minutes
- **Full test suite**: 11+ minutes
- **Docker image pulls**: 1-2 minutes per service
- **Frontend build**: 7 seconds
- **Frontend deps install**: 20 seconds

## Development Guidelines

1. **Simplicity**: Keep the code simple. Every line should do one thing. Functions have a single responsibility and are concise.
2. **Readability**: Write self-explanatory code. Avoid unnecessary comments. If comments are needed, rewrite the code to make it clearer.
3. **Abstraction**: Build reusable utilities and separate business from data logic.
4. **Types**: Use type hints for inputs, prefer generic/wide types, avoid `Any`, and rely on inference where possible. Assume modern python versions (3.13+).
5. **Minimal changes**: Prefer making minimal changes to existing code. If larger changes are needed, consider staging them in smaller, incremental commits.
6. **Testability**: Write code that is easy to test. Avoid complex dependencies and side effects. Use dependency injection where appropriate.
7. **Documentation**: Never suggest docstrings except for tests. Always require docstrings for tests.

## Testing Guidelines

1. **Integration**: Prefer larger integration tests over unit tests.
2. **Mocks**: Avoid using mocks unless absolutely necessary. Prefer using real objects and data.
3. **Less is more**: Write fewer tests that cover more functionality. Avoid writing tests for trivial code.
4. **Invariants**: Focus on testing invariants rather than specific implementations.
5. **Types**: Avoid testing types. Assume a type-checker is used.

## Common Command Outputs - Quick Reference

### Repository Root Structure
```
ls -la
.git/
.github/
.gitignore
.release-please-manifest.json
.vscode/
Dockerfile
Dockerfile.windows
README.md
compose.yaml
config/
coredinator/
coreio/
corerl/
coreui/
docs/
libs/
projects/
release-please-config.json
research/
standards/
test/
```

### Available Configs
```
ls config/
bsm1.yaml
coreio_test_config.yaml
dep_mountain_car_continuous.yaml
dep_mountain_car_continuous_wide.yaml
dep_tennessee_eastman_process.yaml
ensemble_mountain_car.yaml
env/
mountain_car_continuous.yaml
opc_sim/
saturation.yaml
```

### Python Version Check
```
python3 --version
Python 3.12.3 (system)

uv python list
cpython-3.13.0-linux-x86_64-gnu (installed, active)
```

### Docker Services Status
```
docker compose ps
NAME                     IMAGE                               COMMAND                  SERVICE        CREATED          STATUS                    PORTS
core-rl-timescale-db-1   timescale/timescaledb:2.19.1-pg17   "docker-entrypoint.s…"   timescale-db   1 minute ago     Up 1 minute (healthy)     0.0.0.0:5432->5432/tcp
```

## Miscellaneous

1. We have junior developers on the team. Don't always copy surrounding style. If a better way exists, use it.
2. Our team uses Jira for task management.
