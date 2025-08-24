# CoreRL Technical Specification

**Version**: 1.0
**Date**: August 22, 2025
**Status**: Draft

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Architecture Overview](#system-architecture-overview)
3. [Technology Stack](#technology-stack)
4. [Microservices Documentation](#microservices-documentation)
5. [Development Environment](#development-environment)
6. [Internal Tools and Practices](#internal-tools-and-practices)
7. [Deployment and Operations](#deployment-and-operations)
8. [Security Considerations](#security-considerations)
9. [Performance and Scalability](#performance-and-scalability)
10. [Appendices](#appendices)

---

## Executive Summary

CoreRL is a sophisticated reinforcement learning system designed for industrial control and optimization applications. The system provides AI-powered decision-making capabilities for complex industrial processes, with a focus on energy and utilities management.

### Key Features
- **Real-time Decision Making**: Sub-second response times for industrial control
- **Multi-Agent Coordination**: Orchestrated coordination between multiple RL agents
- **Industrial Integration**: Native OPC UA support for seamless industrial system integration
- **High-Performance Computing**: JAX-based neural networks for optimal computational efficiency
- **Scalable Architecture**: Microservices design supporting horizontal scaling
- **Production-Ready**: Comprehensive monitoring, logging, and deployment tools


---

## System Architecture Overview

The CoreRL system implements a distributed microservices architecture optimized for industrial environments. The architecture prioritizes reliability, real-time performance, and integration with existing industrial control systems.

```mermaid
flowchart TD
    CoreUI[CoreUI]
    CoreAuth["CoreAuth<br/> - user authentication<br/> - session management<br/> - access control<br/> - software license key<br/> - service endpoints"]
    Coredinator["Coredinator<br/> - spin up services*<br/> - multi-agent orchestration*<br/> - installation<br/> - healthcheck*"]
    CoreIO["CoreIO<br/> - data ingress<br/> - write setpoints to opc*<br/> - write alarms to opc<br/> - write heartbeat to opc*<br/> - high-freq events triggering<br/> - authenticated opc handshake*<br/> - encrypted comms*<br/> - outbound opc data validation"]
    CoreRL["CoreRL<br/> - agent updating*<br/> - setpoint recommendations*<br/> - internal health metrics*<br/> - estimated value add*<br/> - predicted next setpoint*<br/> - goal satisfaction metrics*<br/> - constraint satisfaction metrics*<br/> - optimization metrics*<br/> - uncertainty estimation"]
    SensorDB[(SensorDB*)]
    DataPipeline["DataPipeline<br/> - virtual tags*<br/> - anomaly detection<br/> - missing data imputation*<br/> - event triggering*<br/> - data preprocessing*<br/> - constraints encoding*<br/> - goal specification*"]
    Buffer[(Buffer)]
    CoreTelemetry["CoreTelemetry<br/> - cpu / ram / disk monitoring<br/> - local metrics caching*<br/> - cloud if outbound internet<br/> - error logging and observability<br/> - server instrumentation"]
    LocalCache[(LocalCache*)]
    CloudMetrics[(CloudMetrics)]
    CoreConfig["CoreConfig<br/> - validation*<br/> - live updating<br/> - config access control<br/> - audit logging"]
    ConfigDB[(ConfigDB)]

    CoreUI --> CoreAuth
    CoreAuth --> Coredinator

    subgraph AuthenticatedUser
        Coredinator --> CoreTelemetry
        Coredinator --> CoreIO
        Coredinator --> CoreRL
        Coredinator <--> CoreConfig

        subgraph Comms
            CoreTelemetry <--> LocalCache
            CoreTelemetry <--> CloudMetrics
            LocalCache -.-> CloudMetrics
            CoreIO --> SensorDB
        end

        SensorDB --> DataPipeline
        CoreConfig <--> ConfigDB
        CoreConfig --> CoreRL

        subgraph Agent
            DataPipeline --> Buffer
            Buffer --> CoreRL
        end
    end
```

### Architecture Principles
1. **Separation of Concerns**: Each service has a single, well-defined responsibility
2. **Async-First Design**: All services built for non-blocking, high-throughput operations
3. **Fault Tolerance**: Services designed to gracefully handle failures and continue operation
4. **Observability**: Comprehensive logging, metrics, and health monitoring
5. **Security by Design**: Authentication, authorization, and encrypted communications

---

## Technology Stack

### Core Technologies

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Runtime** | Python | Primary programming language |
| **Package Manager** | uv | Fast Python package management |
| **ML Framework** | JAX | High-performance numerical computing |
| **Neural Networks** | Haiku | JAX-based neural network library |
| **Web Framework** | FastAPI | REST API services |
| **Database** | TimescaleDB | Time-series data storage |
| **Message Broker** | ZeroMQ | Inter-service messaging |
| **Industrial Comms** | asyncua | OPC UA communication |

### Supporting Technologies

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Data Processing** | Pandas, NumPy, SciPy | Data manipulation and analysis |
| **Configuration** | Pydantic, PyYAML | Type-safe configuration management |
| **Monitoring** | Grafana, Telegraf | Metrics visualization and collection |
| **Containerization** | Docker, Docker Compose | Service packaging and orchestration |
| **Code Quality** | Ruff, Pyright, Pylint | Linting, formatting, and type checking |
| **Testing** | pytest | Unit and integration testing |


---

## Microservices Documentation

### [CoreRL Service](tech_spec/corerl.md)
The main reinforcement learning engine providing AI-powered decision making capabilities.

### [CoreIO Service](tech_spec/coreio.md)
Industrial I/O service handling OPC UA communication and data exchange with control systems.

### [Coredinator Service](tech_spec/coredinator.md)
Orchestration service managing multi-agent coordination and service lifecycle management.

### [Shared Libraries](tech_spec/libraries.md)
Common libraries providing reusable functionality across all services.

### [Research Platform](tech_spec/research.md)
Experimental environment for algorithm development and performance benchmarking.

---

## Development Environment
$1---

## Deployment and Operations

### Containerization Strategy

#### Docker Images
- **Base Images**: Python 3.13 slim images
- **Multi-stage Builds**: Separate build and runtime stages
- **Layer Optimization**: Minimal layers for faster pulls
- **Security**: Non-root users, minimal attack surface

#### Service Configuration
```yaml
services:
  corerl:
    build:
      context: .
      dockerfile: ./corerl/Dockerfile
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@timescale-db:5432/postgres
    healthcheck:
      test: ["CMD", "curl", "--fail", "http://localhost:8000/api/corerl/health"]
      interval: 10s
      timeout: 10s
      retries: 3
```

### Orchestration

#### Development Environment
- **Tool**: Docker Compose
- **Services**: All services with mock external dependencies
- **Hot Reload**: Development volumes for code changes

#### Production Environment
- **Orchestrator**: Kubernetes (recommended) or Docker Swarm
- **Scaling**: Horizontal pod autoscaling
- **Load Balancing**: Ingress controllers with SSL termination

### Monitoring and Observability

#### Metrics Collection
- **Time-series Metrics**: Prometheus/Grafana stack
- **Application Metrics**: Custom business metrics
- **Infrastructure Metrics**: Node exporters, cAdvisor

#### Logging
- **Structured Logging**: JSON format with correlation IDs
- **Centralized Collection**: ELK stack or similar
- **Log Levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL

#### Health Monitoring
- **Health Endpoints**: `/health` for each service
- **Dependency Checks**: Database, external service connectivity
- **Circuit Breakers**: Fail-fast patterns for resilience

### Backup and Recovery

#### Data Backup
- **Database**: Automated TimescaleDB backups
- **Configuration**: Git-based configuration versioning
- **Model Checkpoints**: Periodic RL model snapshots
- **Offsite Backups**: When internet connectivity is available, historical data is backed up to an offsite cloud service provider (e.g., AWS).

#### Disaster Recovery
- **RTO**: Recovery Time Objective < 4 hours
- **RPO**: Recovery Point Objective < 1 hour
- **Testing**: Quarterly disaster recovery drills
- **Continual Learning Resilience**: Because CoreRL (rltune) is a continual learning system, catastrophic data loss is less of a concern. In the event of complete data loss, the agent can be redeployed tabula rasa and will begin learning from scratch on the system. Guardrails and safety measures can be reapplied for the new agent to ensure safe operation.

## Deployment Strategy

We use a blue/green deployment strategy for agent upgrades and releases. The `coredinator` service acts as the orchestrator for deployments, managing both blue and green environments. During an upgrade, the new agent version is deployed to the green environment while the blue environment continues serving production traffic. Once the green environment passes health checks and validation, traffic is switched over. If any issues are detected, `coredinator` can automatically rollback to the previous agent version.

In the event of data corruption, the system rolls back to the previous agent version and deletes all transient data, such as neural network weights. These weights and other ephemeral state are reconstructed live from the process data historian, ensuring system integrity and rapid recovery.

---

## Security Considerations

### Authentication and Authorization

#### Service-to-Service Authentication
- **Method**: JWT tokens with service identity
- **Rotation**: Automatic key rotation every 30 days
- **Validation**: Signature verification and claims validation

#### User Authentication
- **Method**: OAuth 2.0 / OpenID Connect
- **Session Management**: Secure session cookies with CSRF protection
- **Role-Based Access**: Granular permissions per service endpoint

### Data Security

#### Encryption
- **In Transit**: TLS 1.3 for all external communications
- **At Rest**: Database-level encryption for sensitive data
- **Key Management**: External key management service (KMS)

#### Data Privacy
- **PII Handling**: Minimal collection, secure storage
- **Data Retention**: Automated purging based on retention policies
- **Audit Logging**: All data access and modifications logged

### Industrial Security

#### OPC UA Security
- **Authentication**: X.509 certificates for OPC UA connections
- **Encryption**: OPC UA encryption for industrial communications
- **Network Segmentation**: Isolated networks for industrial traffic

#### Network Security
- **Firewall Rules**: Strict ingress/egress rules
- **VPN Access**: Secure remote access for maintenance
- **Intrusion Detection**: Network monitoring and alerting

---

## Performance and Scalability

### Performance Requirements

#### Response Time Targets
- **Real-time Control**: < 100ms for critical control decisions
- **API Responses**: < 500ms for 95th percentile
- **Data Processing**: < 1s for batch processing jobs

#### Throughput Targets
- **OPC UA Messages**: 1000+ messages/second
- **API Requests**: 100+ requests/second per service
- **ML Inference**: 10+ inferences/second per agent

### Scalability Architecture

#### Horizontal Scaling
- **Stateless Services**: All services designed to be stateless
- **Load Balancing**: Round-robin and weighted load balancing
- **Database Sharding**: TimescaleDB partitioning for large datasets

#### Resource Optimization
- **JAX JIT Compilation**: Just-in-time compilation for ML workloads
- **Memory Management**: Efficient memory pooling and garbage collection
- **CPU Optimization**: Multi-threading for I/O-bound operations

### Performance Monitoring

#### Key Performance Indicators (KPIs)
- **Service Response Time**: P50, P95, P99 latencies
- **Error Rates**: Error rate per service endpoint
- **Resource Utilization**: CPU, memory, disk, network usage
- **Business Metrics**: Control accuracy, optimization efficiency

#### Performance Testing
- **Load Testing**: Automated load tests in CI/CD pipeline
- **Stress Testing**: Peak load capacity testing
- **Endurance Testing**: Long-running stability tests

---

## Internal Development Practices

See [internal_practices.md](tech_spec/internal_practices.md) for internal tools, workflows, and code quality standards. This section is for internal use only and should not be shared externally.

---

## Appendices

### Appendix A: Configuration Schema Reference
[Link to detailed configuration schemas](tech_spec/config_schemas.md)

### Appendix B: API Documentation
[Link to OpenAPI specifications](tech_spec/api_docs.md)

### Appendix C: Troubleshooting Guide
[Link to common issues and solutions](tech_spec/troubleshooting.md)

### Appendix D: Performance Benchmarks
[Link to performance test results](tech_spec/benchmarks.md)

### Appendix E: Security Audit Reports
[Link to security assessment results](tech_spec/security_audit.md)

---

*This document is maintained by the CoreRL engineering team and is updated with each major release.*
