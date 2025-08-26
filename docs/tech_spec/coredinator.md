# Coredinator Service Technical Specification

## Overview

Coredinator is the orchestration and internal routing service. It manages the lifecycle of CoreRL platform services, coordinates multi-agent interactions, maintains the mapping between `agent_id` and concrete instances of services (e.g., CoreRL, CoreIO), and acts as the single internal dispatch point for all requests forwarded from `CoreGateway`.

## Architecture

### Core Responsibilities
- **Ingress Dispatch**: Receives all authenticated external traffic from `CoreGateway`, unwraps forwarded requests, and routes to target service.
- **Service Lifecycle Management**: Starting, stopping, and monitoring services.
- **Multi-Agent Coordination**: Orchestrating interactions between multiple RL agents.
- **Health Monitoring & Discovery**: Continuous health checks and dynamic registry of service instances.
- **Configuration Distribution**: Centralized configuration management and propagation.
- **Agent Instance Mapping**: Maintains agent_id -> (CoreRL/CoreIO/etc.) instance mapping, enabling correct routing for per-agent operations.
 - **Protocol Translation & Transport Adaptation**: Converts incoming forwarded HTTP requests into the appropriate internal transport (e.g., ZeroMQ request/reply, future streaming or pub/sub) and normalizes responses back to HTTP for the gateway.

### FastAPI Web Service
Coredinator is a FastAPI application providing REST APIs for service management. It uses a lifespan manager to initialize a service registry on startup and clean up resources on shutdown.

## Key Components

### Service & Agent Registry
The registry maintains:
1. Service inventory (name, location, health, capabilities)
2. Agent mappings (agent_id -> CoreRL instance, associated CoreIO instance, ancillary resources)
3. Version/channel metadata for supporting blue/green or canary rollouts

It exposes APIs for registration, lookup, and health evaluation used internally by the dispatcher.

### Multi-Agent Coordination
The `AgentCoordinator` class manages multi-agent interactions. It gathers agent states, computes a coordination plan, and distributes actions. Coordination results can influence routing (e.g., selecting an alternate policy instance during staged deployments).

### Ingress Dispatch Flow
1. CoreGateway forwards a wrapped request (including original path, method, headers, body, correlation id, authenticated principal claims).
2. Coredinator authenticates forwarding signature (if applied) and validates authorization against internal policies.
3. Path & agent context extracted; if path is agent-scoped, registry resolves agent_id to service instance endpoints.
4. Request is proxied to target service with enriched internal headers (correlation id, resolved-agent-instance id, authorization context).
5. Response is returned to CoreGateway for delivery to client.

## Protocol Translation & Transport Adaptation

`Coredinator` centralizes all internal transport adaptation to keep the external ingress (`CoreGateway`) thin and stateless. When a forwarded HTTP request arrives:

1. The dispatcher identifies the target service and required transport (currently HTTP preferred; ZeroMQ legacy paths supported where a service has not yet published an HTTP façade).
2. For non-HTTP targets, the transport adapter layer serializes the request payload + relevant headers (correlation id, auth claims, resolved agent instance) into the internal message envelope.
3. A request/reply interaction is executed with enforced timeouts and cancellation propagation; latency and outcome are recorded for routing heuristics.
4. The adapter maps internal response codes / error categories to standardized HTTP status codes and attaches diagnostic metadata (stripped before returning externally if sensitive).

### Error & Resilience Strategies
- Timeouts: Per-transport configurable; enforced via async cancellation tokens.
- Circuit Breakers: Per-target service + global fallback states; open state triggers fast failures and optional degraded-mode routing.
- Retry Policy: Idempotent operations may be retried with jittered backoff; non-idempotent requests are not retried.
- Serialization Failures: Logged with redaction; surfaced as HTTP 502 to the client.

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
- **CoreConfig**: Configuration validation and management.
- **CoreTelemetry**: System metrics, routing performance instrumentation.
- **Datastores**: Persists registry snapshots and agent mappings (implementation detail not fixed here).
- **All Services**: Health monitoring and lifecycle management.
- **CoreGateway**: Sole ingress source; trusted forwarder.

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
- **Circuit Breakers**: Per-target service and global dispatch circuit breakers.
- **Blue-Green Deployments**: Zero-downtime service updates, orchestrated by Coredinator with agent mapping shifts.
- **Advanced Coordination**: Game-theoretic multi-agent coordination.
- **Adaptive Routing**: Latency- and load-aware selection among multiple CoreRL replicas for the same agent (when redundancy is introduced).

### Integration Roadmap
- **Observability**: Integration with CoreTelemetry for distributed tracing and per-route metrics.
- **Security**: Mutual auth with CoreGateway and signed forward envelopes.
- **Configuration**: GitOps-based configuration management.
