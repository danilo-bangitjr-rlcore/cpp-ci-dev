# CoreRL Service Technical Specification

## Overview

CoreRL is the central reinforcement learning engine that provides AI-powered decision-making capabilities for industrial control systems. It implements state-of-the-art RL algorithms optimized for real-time industrial applications.

## Architecture

### Core Components

#### Agent Framework
- **Actor-Critic Architecture**: Implements actor-critic methods for continuous control
- **Ensemble Methods**: Multiple critics for uncertainty estimation and robust decision-making
- **JAX-based Networks**: High-performance neural networks using JAX for GPU acceleration

#### Data Pipeline
- **Real-time Processing**: Sub-second data processing for control decisions
- **Feature Engineering**: Automated feature extraction and normalization
- **Missing Data Handling**: Robust imputation strategies for industrial sensor data

#### Learning System
- **Online Learning**: Continuous learning from real-time data
- **Offline Training**: Batch training on historical data
- **Transfer Learning**: Knowledge transfer between similar industrial processes

### Key Modules

```python
corerl/
├── agent/
│   ├── base.py           # Abstract agent interface
│   └── greedy_ac.py      # Greedy actor-critic implementation
├── data_pipeline/
│   ├── datatypes.py      # Data structures and types
│   ├── pipeline.py       # Main data processing pipeline
│   └── imputers/         # Missing data imputation methods
├── eval/
│   ├── agent.py          # Agent evaluation metrics
│   └── representation.py # Representation learning evaluation
├── models/
│   ├── base.py           # Model interface
│   └── dummy.py          # Testing/baseline models
└── interaction/
    └── deployment_interaction.py # Production deployment interface
```

## Technical Implementation

### JAX Integration

CoreRL leverages JAX for high-performance numerical computing:

```python
import jax
import jax.numpy as jnp
import lib_utils.jax as jax_u

@jax_u.method_jit
def get_action(self, state: jax.Array) -> jax.Array:
    """JIT-compiled action selection for maximum performance."""
    return self._actor.apply(self._params, state)
```

### Actor-Critic Architecture

#### Actor Network
- **Purpose**: Policy network that selects actions
- **Architecture**: Multi-layer perceptron with configurable activation functions
- **Output**: Continuous action distributions (typically Gaussian)

#### Critic Network  
- **Purpose**: Value function estimation
- **Architecture**: Ensemble of Q-networks for robust value estimation
- **Output**: State-action value estimates with uncertainty quantification

### Performance Optimizations

#### JIT Compilation
All critical paths are JIT-compiled for optimal performance:
- Action selection: < 1ms latency
- Value estimation: < 2ms latency  
- Gradient computation: Optimized for batch processing

#### Memory Management
- **Buffer Management**: Circular buffers for experience replay
- **Gradient Accumulation**: Efficient gradient batching
- **Model Checkpointing**: Periodic model state persistence

## Configuration

### Agent Configuration
```yaml
agent:
  type: "greedy_ac"
  state_dim: 10
  action_dim: 4
  learning_rate: 0.0003
  ensemble_size: 5
  
network:
  hidden_layers: [256, 256, 128]
  activation: "relu"
  dropout_rate: 0.1
  
training:
  batch_size: 256
  replay_buffer_size: 100000
  target_update_freq: 1000
```

### Environment Configuration
```yaml
environment:
  observation_space:
    type: "continuous"
    low: [-10.0, -5.0, 0.0]
    high: [10.0, 5.0, 100.0]
  
  action_space:
    type: "continuous" 
    low: [0.0, 0.0]
    high: [1.0, 1.0]
    
  constraints:
    power_limit: 1000.0
    efficiency_min: 0.8
```

## API Endpoints

### Health Check
```http
GET /api/corerl/health
```
Returns service health status and basic metrics.

### Agent Status
```http
GET /api/corerl/agent/status
```
Returns current agent state, learning progress, and performance metrics.

### Action Prediction
```http
POST /api/corerl/agent/predict
Content-Type: application/json

{
  "state": [1.2, 3.4, 5.6, ...],
  "constraints": {
    "power_limit": 950.0
  }
}
```

### Model Management
```http
POST /api/corerl/model/checkpoint
POST /api/corerl/model/load
GET /api/corerl/model/info
```

## Metrics and Monitoring

### Performance Metrics
- **Action Latency**: Time from state input to action output
- **Learning Progress**: Training loss, value function accuracy
- **Control Performance**: Setpoint tracking, constraint satisfaction

### Business Metrics
- **Value Add Estimation**: Estimated improvement over baseline control
- **Efficiency Gains**: Energy efficiency improvements
- **Constraint Violations**: Safety and operational constraint adherence

### Health Metrics
- **Memory Usage**: Neural network memory consumption
- **CPU Utilization**: JAX computation resource usage
- **Model Drift**: Distribution shift detection in state/action spaces

## Integration Points

### Data Sources
- **CoreIO**: Real-time sensor data and control commands
- **DataPipeline**: Processed and engineered features
- **ConfigDB**: Dynamic configuration updates

### External Systems
- **TimescaleDB**: Historical data storage and retrieval
- **CoreTelemetry**: Metrics reporting and alerting
- **Coredinator**: Service lifecycle management

## Deployment Considerations

### Resource Requirements
- **CPU**: 4+ cores for JAX computations
- **Memory**: 8GB+ for model and buffer storage
- **GPU**: Optional, recommended for large networks
- **Storage**: 50GB+ for model checkpoints and logs

### Scaling Strategies
- **Vertical Scaling**: Increase CPU/memory for larger models
- **Model Parallelism**: Distribute ensemble members across instances
- **Data Parallelism**: Batch processing across multiple workers

### Reliability Features
- **Graceful Degradation**: Fallback to simpler control when ML fails
- **Model Rollback**: Automatic rollback to previous stable model
- **Circuit Breakers**: Fail-safe mechanisms for critical failures

## Development and Testing

### Unit Testing
```bash
# Run CoreRL-specific tests
cd corerl/
uv run pytest tests/small/

# Run with coverage
uv run pytest tests/small/ --cov=corerl
```

### Integration Testing
```bash
# Test with mock industrial environment
uv run pytest tests/medium/integration/

# End-to-end testing with full stack
docker-compose -f docker-compose.test.yml up --abort-on-container-exit
```

### Performance Testing
```bash
# Benchmark critical paths
uv run pytest tests/benchmarks/test_agent.py

# Profile memory usage
uv run python -m cProfile -o profile.out corerl/main.py
```

## Troubleshooting

### Common Issues

#### High Action Latency
- **Symptoms**: Action selection > 10ms
- **Causes**: Large network size, inefficient JIT compilation
- **Solutions**: Reduce network size, pre-compile critical paths

#### Model Divergence
- **Symptoms**: Increasing training loss, poor control performance
- **Causes**: Learning rate too high, data distribution shift
- **Solutions**: Reduce learning rate, implement domain adaptation

#### Memory Leaks
- **Symptoms**: Increasing memory usage over time
- **Causes**: JAX memory management, buffer overflow
- **Solutions**: Explicit memory cleanup, buffer size limits

### Debugging Tools
```python
# Enable JAX debugging
import jax
jax.config.update("jax_debug_nans", True)
jax.config.update("jax_debug_infs", True)

# Memory profiling
jax.profiler.start_trace("profile_output")
# ... run code ...
jax.profiler.stop_trace()
```

## Future Enhancements

### Planned Features
- **Multi-Agent Coordination**: Distributed RL for multiple agents
- **Hierarchical RL**: Temporal abstraction for complex tasks
- **Meta-Learning**: Fast adaptation to new industrial processes
- **Explainable AI**: Interpretability tools for industrial operators

### Research Directions
- **Safety-Critical RL**: Formal verification of RL policies
- **Federated Learning**: Privacy-preserving multi-site learning
- **Neuromorphic Computing**: Energy-efficient neural architectures