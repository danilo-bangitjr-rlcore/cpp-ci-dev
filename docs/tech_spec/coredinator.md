# Coredinator Service Technical Specification

## Overview

Coredinator is the orchestration service for managing the lifecycle of CoreRL services, coordinating multi-agent interactions, and providing centralized service management.

## Architecture

### Core Responsibilities
- **Service Lifecycle Management**: Starting, stopping, and monitoring services
- **Multi-Agent Coordination**: Orchestrating interactions between multiple RL agents
- **Health Monitoring**: Continuous health checks and service discovery
- **Configuration Distribution**: Centralized configuration management

### FastAPI Web Service
Coredinator is a FastAPI application providing REST APIs for service management. It uses a lifespan manager to initialize a service registry on startup and clean up resources on shutdown.

## Key Components

### Service Registry
The service registry maintains a dictionary of `ServiceInfo` objects, storing details for each service (name, location, health status). It provides methods to register services and perform health checks.

### Multi-Agent Coordination
The `AgentCoordinator` class manages multi-agent interactions. It gathers agent states, computes a coordination plan, and distributes actions.

## API Endpoints

### Service Management
The service management API includes endpoints for listing, registering, deleting, and checking the health of services.

### Health Monitoring
The health monitoring API provides system health status, including individual service status and system-level metrics.

### Configuration Management
The configuration management API allows for generating, validating, and updating configurations.

### Database Operations
The database operations API provides endpoints for verifying and checking database status.

## Configuration

### Service Configuration
The service is configured via a YAML file specifying the host and port for coredinator, connection and health check details for other services, and multi-agent coordination settings.

## Multi-Agent Coordination

### Coordination Strategies
- **Centralized Planning**: Single coordinator makes decisions for all agents
- **Decentralized Consensus**: Agents negotiate through coordinator
- **Priority-Based**: High-priority agents get preference in conflicts
- **Resource-Based**: Coordination based on available system resources

### Implementation Example
A `MultiAgentCoordinator` class can be implemented to perform coordination cycles by collecting agent intentions, detecting and resolving conflicts, and distributing final actions. Conflict resolution is based on conflict type, such as resource contention or goal interference.

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

Coredinator is deployed as a bare-metal executable on Windows and Linux and manages the lifecycle of other CoreRL services on the same machine. Configuration is managed via local YAML files.

## Monitoring and Alerting

### Key Metrics
- **Service Availability**: Uptime percentage for each service
- **Response Times**: Health check and API response latencies
- **Error Rates**: Failed health checks and API errors
- **Coordination Efficiency**: Multi-agent coordination success rates

### Alerting Rules
Alerting rules are defined in YAML to trigger alerts based on conditions like service downtime, high response times, or coordination failures, with specifiable severity.

## Future Enhancements

### Planned Features
- **Circuit Breakers**: Fail-fast patterns for resilience
- **Blue-Green Deployments**: Zero-downtime service updates, orchestrated by Coredinator.
- **Advanced Coordination**: Game-theoretic multi-agent coordination

### Integration Roadmap
- **Observability**: Integration with CoreTelemetry for distributed tracing.
- **Security**: Secret management for service configurations.
- **Configuration**: GitOps-based configuration management.
