# CoreRL Service Technical Specification

## Overview

CoreRL is the central reinforcement learning engine that provides AI-powered decision-making capabilities for industrial control systems

## Deployment Considerations

### Resource Requirements
- **CPU**: 4+ cores for JAX computations
- **Memory**: 8GB+ for model and buffer storage

### Deployment
The CoreRL service is deployed as a bare-metal executable on Windows or Linux. Its lifecycle, including high availability and automatic restarts, is managed by the `coredinator` service. It includes reliability features such as graceful degradation to simpler control models and circuit breakers for critical failures.

## Architecture

### Core Components

#### Agent Framework
- **Actor-Critic Architecture**: Implements actor-critic methods for continuous control
- **Ensemble Methods**: Multiple critics for uncertainty estimation and robust decision-making
- **JAX-based Networks**: High-performance neural networks using JAX

#### Data Pipeline
- **Real-time Processing**: Sub-second data processing for control decisions
- **Feature Engineering**: Automated feature extraction and normalization
- **Missing Data Handling**: Robust imputation strategies for industrial sensor data
- **Drift Detection**: Online learning to detect and adapt to data distribution shifts
- **Anomaly Detection**: Real-time identification of anomalous data points

#### Learning System
- **Online Learning**: Continuous learning from real-time data
- **Offline Training**: Batch training on historical data
- **Process-Specific Agents**: Each agent is tailored to a specific industrial process.


## Technical Implementation

### Data Pipeline Implementation

- **Sensor Trace History**: The agent maintains a local history of recent sensor readings for each control point. This trace provides short-term context, allowing the agent to infer temporal dynamics such as rates of change.
- **Masked Autoencoder for Imputation**: Missing sensor data is handled by a masked autoencoder. The model is trained to reconstruct missing values by learning the manifold of the sensor data, enabling robust imputation even with significant data loss.

### Actor-Critic Architecture

#### Actor Network
- **Purpose**: Policy network that selects actions
- **Architecture**: Multi-layer perceptron with configurable activation functions
- **Output**: Continuous action distributions (typically Gaussian)

#### Critic Network
- **Purpose**: Value function estimation
- **Architecture**: Ensemble of Q-networks for robust value estimation
- **Output**: State-action value estimates with uncertainty quantification

### Learning System Implementation

- **Q-Learning with Regularized Corrections (QRC)**: The critic is implemented as a gradient-based temporal difference (TD) learning algorithm known as QRC. This method is based on the principles outlined in the generalized projected Bellman equation framework (Patterson et al.).
- **Mixed History Replay Buffer**: The replay buffer combines recent experiences with historical data. It uses ensemble masking and recency weighting to prioritize more relevant and recent data, improving learning stability and performance.
- **Rolling Ensemble Resets**: To encourage exploration and prevent convergence to suboptimal policies, individual members of the critic ensemble are periodically reset to a random state. This "rolling reset" mechanism is a key component of the continual learning strategy.


## API Endpoints

### Health Check
```http
GET /api/healthcheck
```
Returns a simple `200 OK` if the service is running and connected.

### Metrics & Status
```http
GET /api/metrics
```
Returns detailed operational metrics, including agent status, learning progress, and performance metrics.

### Model Management
```http
PUT /api/corerl/model/checkpoint
PUT /api/corerl/model/load
GET /api/corerl/model/info
```

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
- **NN Health**: Monitoring of neural network capacity and health (e.g., gradient norms, activation saturation)
- **Model Uncertainty**: Quantification of model uncertainty for risk assessment
- **Prediction Error**: Tracking of prediction errors against actual outcomes

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


## Future Enhancements

### Planned Features
- **Multi-Agent Coordination**: Distributed RL for multiple agents
- **Hierarchical RL**: Temporal abstraction for complex tasks
- **Meta-Learning**: Continual, online hyperparameter tuning and architecture search
- **Explainable AI**: Interpretability tools for industrial operators

### Research Directions
- **Safety-Critical RL**: Formal verification of RL policies
