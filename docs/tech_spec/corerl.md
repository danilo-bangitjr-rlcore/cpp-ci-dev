# CoreRL Service Technical Specification

## Overview

CoreRL is the reinforcement learning engine providing decision-making for industrial control systems. It functions as a supervisory controller, optimizing the setpoints of lower-level control systems..

## Deployment Considerations

Refer to the authoritative [Service Deployment Model](coredinator.md#service-deployment-model-authoritative). CoreRL-specific notes:
- Prefers NUMA locality where available (pin worker threads if host is multi-socket).

### Resource Requirements (See baseline sizing matrix in root tech spec)
- Recommended: 4+ CPU cores for JAX computations (vectorized inference + background updates)
- Recommended: 8GB+ memory per active agent (model + replay + feature cache)

## Architecture

### Core Components

#### Agent Framework
- **Actor-Critic Architecture**: Actor-critic methods for continuous control.
- **Ensemble Methods**: Multiple critics for uncertainty estimation.
- **JAX-based Networks**: Neural networks using JAX.

#### Data Pipeline
- **Real-time Processing**: Sub-second data processing.
- **Feature Engineering**: Automated feature extraction and normalization.
- **Missing Data Handling**: Imputation strategies for sensor data.
- **Drift Detection**: Online detection and adaptation to data distribution shifts.
- **Anomaly Detection**: Real-time identification of anomalous data.

#### Learning System
- **Online Learning**: Continuous learning from real-time data to adapt to changing process dynamics.
- **Offline Training**: Batch training on historical data.
- **Process-Specific Agents**: Agents are tailored to a specific industrial process.


## Technical Implementation

### Data Pipeline Implementation

- **Sensor Trace History**: The agent maintains a local history of recent sensor readings per control point to infer temporal dynamics.
- **Masked Autoencoder for Imputation**: Missing sensor data is handled by a masked autoencoder trained to reconstruct missing values.

### Actor-Critic Architecture

#### Actor Network
- **Purpose**: Policy network for action selection.
- **Architecture**: Multi-layer perceptron with configurable activation functions.
- **Output**: Continuous action distributions (Gaussian).

#### Critic Network
- **Purpose**: Value function estimation.
- **Architecture**: Ensemble of Q-networks for value estimation.
- **Output**: State-action value estimates with uncertainty.

### Learning System Implementation

- **Q-Learning with Regularized Corrections (QRC)**: The critic is a gradient-based TD learning algorithm (QRC), based on the generalized projected Bellman equation framework.
- **Replay Buffer Strategy**: CoreRL composes buffer implementations from `lib_agent` (see [Shared Libraries](libraries.md)). (Mixed-history sampling, masking, recency weighting) are documented centrally there to avoid divergence.
- **Rolling Ensemble Resets**: Individual members of the critic ensemble are periodically reset to a random state to encourage exploration.


## API Endpoints

### Health Check
The health check API provides an endpoint to verify the service is running.

### Metrics & Status
The metrics and status API returns operational metrics, including agent status, learning progress, and performance.

### Model Management
The model management API provides endpoints for checkpointing, loading, and getting information about the current model.

## Metrics and Monitoring

### Performance Metrics
- **Action Latency**: Time from state input to action output
- **Learning Progress**: Training loss, value function accuracy
- **Control Performance**: Setpoint tracking, constraint satisfaction

### Business Metrics
- **Value Estimation**: Estimated process value and costs
- **Constraint Violations**: Safety and operational constraint adherence

### Health Metrics
- **Data Drift**: Distribution shift detection in state/action spaces
- **NN Health**: Monitoring of neural network capacity and health.
- **Model Uncertainty**: Quantification of model uncertainty.
- **Prediction Error**: Tracking of prediction errors.

## Integration Points

### Data Sources
- **CoreIO**: Real-time sensor data and control commands
- **DataPipeline**: Processed and engineered features
- **ConfigDB**: Dynamic configuration updates

### External Systems
- **TimescaleDB**: Historical data storage and retrieval
- **CoreTelemetry**: Metrics reporting and alerting
- **Coredinator**: Service lifecycle management

## Future Enhancements

### Planned Features
- **Hierarchical RL**: Temporal abstraction for complex tasks
- **Meta-Learning**: Online hyperparameter tuning and architecture search.
- **Explainable AI**: Interpretability tools.
- **Coordination Interfaces**: CoreRL will continue to expose stable interfaces used by Coredinator for multi-agent coordination; coordination logic ownership remains in Coredinator (see its spec).

### Research Directions
- **Safety-Critical RL**: Formal verification of RL policies
