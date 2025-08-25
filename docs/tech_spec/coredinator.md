# Coredinator Service Technical Specification

## Overview

Coredinator is the orchestration service responsible for managing the lifecycle of all CoreRL services, coordinating multi-agent interactions, and providing centralized service management capabilities.

## Architecture

### Core Responsibilities
- **Service Lifecycle Management**: Starting, stopping, and monitoring services
- **Multi-Agent Coordination**: Orchestrating interactions between multiple RL agents
- **Health Monitoring**: Continuous health checks and service discovery
- **Configuration Distribution**: Centralized configuration management

### FastAPI Web Service
Coredinator is built as a FastAPI application providing REST APIs for service management:

```python
from fastapi import FastAPI
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize service registry
    await initialize_service_registry()
    yield
    # Shutdown: Cleanup resources
    await cleanup_services()

app = FastAPI(lifespan=lifespan)
```

## Key Components

### Service Registry
```python
@dataclass
class ServiceInfo:
    name: str
    host: str
    port: int
    health_endpoint: str
    status: ServiceStatus
    last_health_check: datetime

class ServiceRegistry:
    def __init__(self):
        self.services: Dict[str, ServiceInfo] = {}

    async def register_service(self, service: ServiceInfo):
        """Register a new service with the orchestrator."""
        self.services[service.name] = service

    async def health_check_all(self):
        """Perform health checks on all registered services."""
        for service in self.services.values():
            await self.check_service_health(service)
```

### Multi-Agent Coordination
```python
class AgentCoordinator:
    def __init__(self):
        self.agents: Dict[str, AgentProxy] = {}

    async def coordinate_agents(self, coordination_request):
        """Coordinate actions between multiple agents."""
        # Collect agent states
        states = await self.gather_agent_states()

        # Compute coordination plan
        plan = await self.compute_coordination_plan(states)

        # Distribute actions to agents
        await self.distribute_actions(plan)
```

## API Endpoints

### Service Management
```http
GET /api/coredinator/services
POST /api/coredinator/services/register
DELETE /api/coredinator/services/{service_id}
GET /api/coredinator/services/{service_id}/health
```

### Health Monitoring
```http
GET /api/coredinator/health
```
Returns comprehensive system health status:

```json
{
  "status": "healthy",
  "services": {
    "corerl": {
      "status": "healthy",
      "last_check": "2025-08-22T10:30:00Z",
      "response_time": "45ms"
    },
    "coreio": {
      "status": "healthy",
      "last_check": "2025-08-22T10:30:00Z",
      "response_time": "23ms"
    }
  },
  "system_metrics": {
    "cpu_usage": "25%",
    "memory_usage": "45%",
    "active_agents": 3
  }
}
```

### Configuration Management
```http
GET /api/coredinator/config/generate
POST /api/coredinator/config/validate
PUT /api/coredinator/config/update
```

### Database Operations
```http
POST /api/coredinator/database/verify
GET /api/coredinator/database/status
```

## Configuration

### Service Configuration
```yaml
coredinator:
  host: "0.0.0.0"
  port: 8002

  services:
    corerl:
      host: "localhost"
      port: 8000
      health_endpoint: "/api/corerl/health"
      restart_policy: "always"

    coreio:
      host: "localhost"
      port: 8001
      health_endpoint: "/api/coreio/health"
      restart_policy: "on-failure"

  health_check:
    interval: 30  # seconds
    timeout: 10   # seconds
    retries: 3

  coordination:
    enabled: true
    sync_interval: 1.0  # seconds
    conflict_resolution: "priority_based"
```

## Multi-Agent Coordination

### Coordination Strategies
- **Centralized Planning**: Single coordinator makes decisions for all agents
- **Decentralized Consensus**: Agents negotiate through coordinator
- **Priority-Based**: High-priority agents get preference in conflicts
- **Resource-Based**: Coordination based on available system resources

### Implementation Example
```python
class MultiAgentCoordinator:
    async def coordinate_step(self):
        """Perform one coordination cycle."""
        # 1. Collect agent intentions
        intentions = await self.collect_agent_intentions()

        # 2. Detect conflicts
        conflicts = self.detect_conflicts(intentions)

        # 3. Resolve conflicts
        resolved_actions = await self.resolve_conflicts(conflicts)

        # 4. Distribute final actions
        await self.distribute_actions(resolved_actions)

    async def resolve_conflicts(self, conflicts):
        """Resolve conflicts between agents."""
        for conflict in conflicts:
            if conflict.type == "resource_contention":
                await self.resolve_resource_conflict(conflict)
            elif conflict.type == "goal_interference":
                await self.resolve_goal_conflict(conflict)
```

## Integration Points

### Service Dependencies
- **CoreConfig**: Configuration validation and management
- **CoreTelemetry**: System metrics and monitoring
- **Database**: Service state persistence
- **All Services**: Health monitoring and lifecycle management

### External Integrations
- **Windows Services**: Manages services on Windows servers.
- **Linux Systemd**: Manages services on Linux servers.

## Deployment

Coredinator is deployed as a bare-metal executable on both Windows and Linux servers. It is responsible for managing the lifecycle of other CoreRL services on the same machine. Configuration is managed via local YAML files.

## Monitoring and Alerting

### Key Metrics
- **Service Availability**: Uptime percentage for each service
- **Response Times**: Health check and API response latencies
- **Error Rates**: Failed health checks and API errors
- **Coordination Efficiency**: Multi-agent coordination success rates

### Alerting Rules
```yaml
alerts:
  - name: ServiceDown
    condition: service_health == 0
    duration: 60s
    severity: critical

  - name: HighResponseTime
    condition: avg_response_time > 1000ms
    duration: 300s
    severity: warning

  - name: CoordinationFailure
    condition: coordination_success_rate < 0.95
    duration: 120s
    severity: warning
```

## Future Enhancements

### Planned Features
- **Circuit Breakers**: Fail-fast patterns for resilience
- **Blue-Green Deployments**: Zero-downtime service updates, orchestrated by Coredinator.
- **Advanced Coordination**: Game-theoretic multi-agent coordination

### Integration Roadmap
- **Observability**: Deeper integration with CoreTelemetry for distributed tracing.
- **Security**: Enhanced secret management for service configurations.
- **Configuration**: GitOps-based configuration management for version-controlled service definitions.
