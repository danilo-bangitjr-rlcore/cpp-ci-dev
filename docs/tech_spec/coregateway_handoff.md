# Handoff: CoreGateway Implementation

**To:** Development Team
**From:** Architecture Team
**Date:** August 24, 2025
**Subject:** Handoff for CoreGateway Service Implementation

## 1. Executive Summary

This document outlines the implementation plan for the new **CoreGateway** service. The gateway will serve as the single, secure entry point for all public-facing REST APIs on the CoreRL platform, running as a single instance managed by `coredinator`.

The full technical specification, which is the single source of truth, can be found here: [CoreGateway Service Technical Specification](coregateway.md)

## 2. Architecture Overview

The `CoreGateway` is a lightweight, stateless API gateway built with FastAPI. Its key responsibilities are:
-   **Authentication**: Validating JWTs via local middleware.
-   **Routing**: Forwarding requests to internal services based on a Python dataclass configuration (`routes.py`).
-   **Rate Limiting**: Protecting services from abuse using a global, in-memory rate limiter.
-   **Lifecycle Management**: The service is deployed as a bare-metal executable and its lifecycle (start, stop, restart) is managed by `coredinator`.

## 3. Implementation Tasks

The implementation plan is based on the checklist from the technical specification.

| Task                                      | Priority | Estimated Effort |
| ----------------------------------------- | :------: | :--------------: |
| 1. Scaffold FastAPI Service + Simple Proxy |  **1**   |      1 day       |
| 2. Implement Routing Dataclasses & Engine |  **1**   |      2 days      |
| 3. Implement JWT Auth Middleware (JWK)    |  **1**   |      2 days      |
| 4. Add In-Memory Rate Limiter             |  **2**   |      1 day       |
| 5. Add Metrics Pushing to CoreTelemetry   |  **2**   |      1 day       |
| 6. Write E2E Tests (with mock services)   |  **3**   |      2 days      |

## 4. Key Acceptance Criteria

The implementation will be considered complete when the following criteria are met:

1.  The gateway successfully routes requests to the correct downstream services as defined in the `routes.py` configuration.
2.  Protected endpoints return a `401` or `403` error for invalid, missing, or insufficient-permission JWTs, based on the defined `Role` enum.
3.  Rate limiting returns a `429` error when requests exceed the configured global limits.
4.  The gateway is deployed as a bare-metal executable and can be successfully managed by `coredinator`.
5.  Key operational metrics are pushed directly to the `CoreTelemetry` service on each request/event.

## 5. Risks and Mitigations

-   **Performance Bottleneck**: As a single instance, the gateway's performance is critical.
    -   **Mitigation**: Ensure all processing (auth, routing) is highly efficient. Conduct load testing to validate performance against NFRs. If needed, performance can be improved by vertical scaling (more CPU/RAM).
-   **Single Point of Failure**: If the gateway fails, the entire platform is inaccessible.
    -   **Mitigation**: `coredinator` is responsible for high availability. It will continuously monitor the gateway's `/healthcheck` endpoint and automatically restart the service if it becomes unresponsive.
-   **Complex Routing Logic**: The routing logic could become complex over time.
    -   **Mitigation**: Keep the routing configuration declarative in the Python dataclasses. Avoid embedding business logic in the gateway.

Please review the full technical specification for detailed requirements before beginning implementation.
