# CoreCLI

Command line tools for the reinforcement learning development team.

## Installation

```bash
# Install in development mode
uv sync --dev
```

## Quick Start

```bash
# Check if coredinator is running
corecli coredinator status

# Start an agent with a config file
corecli agent start config/mountain_car_continuous.yaml

# Check agent status
corecli agent status config/mountain_car_continuous.yaml

# List all running agents
corecli agent list

# Stop the agent
corecli agent stop config/mountain_car_continuous.yaml
```

## Commands

### Development Workflow

#### `corecli dev start-sim <config-path>`
Starts a development simulation environment with all required services.

**What it does:**
- Validates the configuration file
- Ensures coredinator is running (starts if needed)
- Launches OPC server, Grafana, and TimescaleDB via `docker compose up`
- Sets up the complete simulation stack for testing

```bash
# Start simulation for mountain car environment
corecli dev start-sim config/mountain_car_continuous.yaml

# Start with verbose logging
corecli --verbose dev start-sim config/bsm1.yaml
```

#### `corecli dev stop-sim`
Cleanly shuts down the simulation environment.

```bash
# Stop all simulation services
corecli dev stop-sim

# Stop and remove volumes
corecli dev stop-sim --clean
```

#### `corecli dev logs [service]`
View logs from simulation services.

```bash
# View all logs
corecli dev logs

# View specific service logs
corecli dev logs opc-server
corecli dev logs grafana
```

#### `corecli dev monitor-events`
Monitor event bus traffic in real-time. This developer tool connects to the
coredinator XPUB socket (default port 5571) and prints every published
message to the console. It's useful for debugging pub/sub interactions and
verifying that events are flowing between services.

**Notes:**
- The command connects as a subscriber to the event bus; ensure the
	`coredinator` service is running and the event bus is available.
- By default the command subscribes to all topics; you can filter to a
	specific topic if needed.

```bash
# Monitor all event bus traffic (default host localhost, port 5571)
corecli dev monitor-events

# Connect to a specific host/port
corecli dev monitor-events --host 127.0.0.1 --port 5571

# Filter by a single topic
corecli dev monitor-events --topic agent_events
```

### Coredinator Management

#### `corecli coredinator status`
Checks the health and status of the coredinator service.

**Returns:**
- Service health (healthy/unhealthy/unreachable)
- Version information
- Active agent count

```bash
# Check coredinator health
corecli coredinator status

# Output example:
# ✅ Coredinator: HEALTHY (v0.144.0)
# 📊 Active agents: 2
```

#### `corecli coredinator start`
Starts the coredinator service if not running.

```bash
# Start coredinator with default settings
corecli coredinator start

# Start with custom port
corecli coredinator start --port 8081
```

#### `corecli coredinator stop`
Gracefully stops the coredinator service.

```bash
corecli coredinator stop
```

### Agent Management

#### `corecli agent start <config-path>`
Starts a new RL agent using the specified configuration.

**Prerequisites:**
- Valid configuration file
- Coredinator service running
- Required simulation services (if using sim environments)

```bash
# Start agent with basic config
corecli agent start config/mountain_car_continuous.yaml

# Start with custom name
corecli agent start config/bsm1.yaml --name "bsm1-experiment-1"

# Start and watch logs
corecli agent start config/ensemble_mountain_car.yaml --follow-logs
```

#### `corecli agent stop <config-path-or-agent-id>`
Stops a running agent.

```bash
# Stop by config path
corecli agent stop config/mountain_car_continuous.yaml

# Stop by agent ID
corecli agent stop agent-abc123

# Force stop (if graceful shutdown fails)
corecli agent stop config/bsm1.yaml --force
```

#### `corecli agent status <config-path-or-agent-id>`
Shows detailed status information for an agent.

```bash
# Check status by config
corecli agent status config/mountain_car_continuous.yaml

# Check by agent ID
corecli agent status agent-abc123

# Output example:
# 🤖 Agent: agent-abc123
# 📁 Config: config/mountain_car_continuous.yaml
# 🔄 Status: RUNNING
# ⏱️  Uptime: 2h 34m
# 🎯 Latest reward: 89.3
```

#### `corecli agent list`
Lists all agents known to coredinator.

```bash
# List all agents
corecli agent list

# Show only running agents
corecli agent list --running

# Show detailed information
corecli agent list --verbose

# Output example:
# 🤖 Running Agents (2):
# ├── agent-abc123 → config/mountain_car_continuous.yaml (2h 34m)
# └── agent-def456 → config/bsm1.yaml (45m)
```

#### `corecli agent logs <config-path-or-agent-id>`
View logs from a specific agent.

```bash
# Follow logs in real-time
corecli agent logs config/mountain_car_continuous.yaml --follow

# Show last 100 lines
corecli agent logs agent-abc123 --tail 100

# Filter by log level
corecli agent logs agent-abc123 --level ERROR
```

### Configuration Management

#### `corecli config validate <config-path>`
Validates a configuration file against the schema.

```bash
# Validate single config
corecli config validate config/mountain_car_continuous.yaml

# Validate all configs in directory
corecli config validate config/ --recursive

# Show validation details
corecli config validate config/bsm1.yaml --verbose
```

## Development

This package follows the monorepo standards and patterns.

### Running Tests

```bash
uv run pytest

# Run with coverage
uv run pytest --cov=corecli

# Run specific test category
uv run pytest tests/small/
```

### Code Quality

```bash
# Run linting
uv run ruff check
uv run pyright

# Format code
uv run ruff format

# Run all checks
uv run ruff check && uv run pyright && uv run pytest
```

### Adding New Commands

1. Add command function to appropriate module in `corecli/commands/`
2. Register command in `corecli/main.py`
3. Add tests in `tests/commands/`
4. Update this README with usage examples

For detailed development guidelines, see the [monorepo documentation](../README.md).
