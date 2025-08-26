# Shared Libraries Technical Specification

## Overview

The CoreRL system includes shared libraries that provide common functionality across microservices, promoting code reuse and consistency.

## Library Architecture

```
libs/
├── lib_agent/        # RL agent components and algorithms
├── lib_config/       # Configuration management and validation
├── lib_utils/        # General utilities and JAX helpers
├── lib_defs/         # Type definitions and constants
└── rl_env/           # RL environments and simulators
```

## lib_agent: RL Agent Components

### Purpose
Provides RL components like neural networks, buffers, critics, and actors.

### Key Components

#### Neural Networks (`lib_agent/network/`)
This module provides neural networks for RL, with JAX-based implementations, ensemble methods, configurable architectures, and support for dropout and batch normalization.

#### Experience Buffers (`lib_agent/buffer/`)
This module contains experience replay buffers, including a standard FIFO circular buffer and a mixed-history buffer for prioritized sampling. Buffers use JAX array storage with automatic memory management.

#### Actor Networks (`lib_agent/actor/`)
This module provides actor networks for policy learning, featuring a percentile-based policy, support for continuous action distributions (e.g., Gaussian), and JIT compilation with JAX.

#### Critic Networks (`lib_agent/critic/`)
This module contains critic networks for value function estimation, using gradient-based TD learning, ensemble techniques, and a rolling reset mechanism.

### Data Types
The library defines shared data structures, like `JaxTransition`, which encapsulates data for a single RL interaction step in a JAX-native format.

## lib_config: Configuration Management

### Purpose
Provides configuration management with validation, type safety, and dynamic updates.

### Core Features
- **Type-Safe Configuration**: Pydantic for schema validation and type-safe configuration objects.
- **Hierarchical Configuration**: Supports nested configuration structures.
- **Dynamic Configuration Updates**: Watches for configuration changes and updates service configurations at runtime.
- **Validation**: Framework for schema-level and custom business logic validation, with support for environment-specific overrides.

## lib_utils: General Utilities

### Purpose
Provides common utilities and JAX-related helpers.

### JAX Utilities (`lib_utils/jax.py`)
This module provides helpers for JAX, including decorators for JIT compilation, utilities for vectorization (vmap), and helpers for computing gradients with auxiliary data.

### Data Utilities
This module contains helpers for standard Python data structures, including utilities for deep-merging dictionaries, safely accessing nested values, chunking and flattening lists, and composing operations on optional values.

### Error Handling
This module provides custom error types and context managers to standardize error handling.

## lib_defs: Type Definitions

### Purpose
Provides shared type definitions, constants, and interfaces.

### Common Types
This module defines common type aliases for clarity and consistency (e.g., `StateArray`, `ActionArray`). These types are used to ensure data shape and type consistency.

### Protocol Definitions
This module defines `Protocol` classes as interfaces for core components (agents, environments, buffers) to allow for dependency inversion and easier testing.

### Constants and Enums
This module contains shared constants and enumerations (e.g., default learning rates, max episode lengths).

## rl_env: RL Environments

### Purpose
Provides RL environments for testing, simulation, and training.

## Integration and Dependencies

### Dependency Graph
Libraries are organized in a directed acyclic graph to prevent circular dependencies. `lib_defs` is the base library. Main services (`corerl`, `coreio`) depend on these shared libraries.

### Version Management
Each library follows semantic versioning. Dependencies are managed via `pyproject.toml`.

### Local Development
The development environment is managed with `uv`. Libraries can be installed in editable mode. Each library includes its own tests and type-checking configurations.

## Testing Strategy

### Unit Testing
Each library has a suite of unit tests that can be run independently, with coverage reporting.

### Integration Testing
Cross-library integration tests verify that libraries work together correctly.

### Performance Testing
Performance-critical functions are benchmarked to prevent regressions.

## Best Practices

### Library Design Principles
1. **Single Responsibility**: Each library has a focused purpose.
2. **Minimal Dependencies**: Avoid circular dependencies.
3. **Type Safety**: Type hints and validation.
4. **Performance**: JAX-optimized implementations.
5. **Testability**: Test coverage and clear interfaces.
