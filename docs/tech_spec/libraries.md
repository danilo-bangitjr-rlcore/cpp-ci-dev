# Shared Libraries Technical Specification

## Overview

The CoreRL system includes several shared libraries that provide common functionality across all microservices. These libraries promote code reuse, maintain consistency, and provide specialized capabilities for reinforcement learning applications.

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
Provides core reinforcement learning components including neural networks, buffers, critics, and actors that can be shared across different RL implementations.

### Key Components

#### Neural Networks (`lib_agent/network/`)
This module provides high-performance neural networks optimized for reinforcement learning. It includes features like JAX-based implementations for high-performance computing, ensemble methods for uncertainty estimation, configurable architectures, and support for dropout and batch normalization.

#### Experience Buffers (`lib_agent/buffer/`)
This module contains various experience replay buffers. It offers a standard FIFO circular buffer for recent experiences, as well as a mixed-history buffer that allows for prioritized sampling from different time periods. The buffers are implemented using efficient JAX array storage with automatic memory management.

#### Actor Networks (`lib_agent/actor/`)
This module provides actor networks for policy learning. It features a percentile-based policy for robust learning, support for continuous action distributions (e.g., Gaussian), and is JIT-compiled with JAX for high performance.

#### Critic Networks (`lib_agent/critic/`)
This module contains critic networks for value function estimation. It utilizes gradient-based temporal difference (TD) learning methods, ensemble techniques for robust value estimation, and a rolling reset mechanism to prevent overfitting and encourage exploration.

### Data Types
The library defines shared data structures, such as `JaxTransition`, which encapsulates the data for a single step of an RL interaction (state, action, reward, etc.) in a JAX-native format.

## lib_config: Configuration Management

### Purpose
Provides robust configuration management with validation, type safety, and dynamic updates for all services.

### Core Features
- **Type-Safe Configuration**: Leverages Pydantic for schema validation and type-safe configuration objects, ensuring that configurations are correct at load time.
- **Hierarchical Configuration**: Supports nested configuration structures, allowing for modular and organized configuration files.
- **Dynamic Configuration Updates**: Includes a mechanism to watch for configuration changes and update service configurations at runtime without requiring a restart.
- **Validation**: Provides a framework for both schema-level validation and custom business logic validation, with support for environment-specific overrides.

## lib_utils: General Utilities

### Purpose
Provides common utilities, especially JAX-related helpers, that are used across the entire system.

### JAX Utilities (`lib_utils/jax.py`)
This module provides a collection of helpers to simplify working with JAX. It includes decorators for JIT compilation of methods and functions, utilities for advanced vectorization (vmap), and helpers for computing gradients with auxiliary data.

### Data Utilities
This module contains helper functions for working with standard Python data structures. It includes utilities for deep-merging dictionaries, safely accessing nested values, chunking and flattening lists, and composing operations on optional values in a functional style.

### Error Handling
This module provides a set of custom error types and context managers to standardize error handling across services. This ensures that errors are handled consistently and provide meaningful context.

## lib_defs: Type Definitions

### Purpose
Provides shared type definitions, constants, and interfaces used across all services.

### Common Types
This module defines a set of common type aliases for clarity and consistency, such as `StateArray`, `ActionArray`, and `BatchedStates`. These types are used throughout the codebase to ensure that data shapes and types are consistent.

### Protocol Definitions
This module defines a set of `Protocol` classes that act as interfaces for core components like agents, environments, and buffers. This allows for dependency inversion and makes it easier to test components in isolation.

### Constants and Enums
This module contains a set of shared constants and enumerations, such as default learning rates, maximum episode lengths, and data modes. This ensures that these values are consistent across all services.

## rl_env: RL Environments

### Purpose
Provides standardized RL environments for testing, simulation, and training.

## Integration and Dependencies

### Dependency Graph
The libraries are organized in a directed acyclic graph to prevent circular dependencies. `lib_defs` is the base library with no dependencies, while other libraries like `lib_agent` and `rl_env` depend on the utility and configuration libraries. The main services, such as `corerl` and `coreio`, then depend on these shared libraries.

### Version Management
Each library follows semantic versioning. Dependencies between libraries are managed through `pyproject.toml` files, ensuring that services use compatible library versions.

### Local Development
The development environment is managed using `uv`. Libraries can be installed in editable mode for local development, and each library includes its own set of tests and type-checking configurations.

## Testing Strategy

### Unit Testing
Each library has a comprehensive suite of unit tests that can be run independently. This includes tests for individual components and utilities, with coverage reporting to ensure test quality.

### Integration Testing
Cross-library integration tests are used to ensure that the libraries work together correctly. These tests cover key interactions, such as an agent using data from a replay buffer to perform an update.

### Performance Testing
Performance-critical functions within the libraries are benchmarked to prevent regressions. This includes benchmarks for buffer operations, network forward passes, and other computationally intensive tasks.

## Best Practices

### Library Design Principles
1. **Single Responsibility**: Each library has a clear, focused purpose
2. **Minimal Dependencies**: Avoid circular dependencies between libraries
3. **Type Safety**: Comprehensive type hints and validation
4. **Performance**: JAX-optimized implementations where appropriate
5. **Testability**: Extensive test coverage and clear interfaces
