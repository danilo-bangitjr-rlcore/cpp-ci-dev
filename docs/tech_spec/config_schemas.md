# RLTune Configuration Schema

**Version**: 1.0
**Date**: August 26, 2025

This document provides a reference for configuring the RLTune system using a commented example. The configuration is managed via a single YAML file that defines everything from data sources and agent behavior to hardware settings.

---

## Example Configuration File

Below is a comprehensive example of a configuration file. Each parameter is explained with inline comments. You can use this as a template for your own deployments.

```yaml
# agent_name: A unique identifier for the agent instance. This is used for logging,
# checkpointing, and distinguishing between multiple agents.
# Type: string
# Required: Yes
agent_name: "my_industrial_agent"

# is_simulation: Specifies whether the agent is running in a simulation or a live
# production environment. Set to false for real-world deployments.
# Type: boolean
# Default: True
is_simulation: False

# seed: An integer used to seed random number generators for both the agent and
# environment. This ensures that experiments are reproducible.
# Type: integer
# Default: 0
seed: 42

# log_path: Optional path to a directory where logs will be saved. If not provided,
# logs are handled by the telemetry service but not saved to a local file.
# Type: string (path)
# Default: None
log_path: "outputs/agent_logs"

# --------------------
# -- Infrastructure --
# --------------------
infra:
  # db: Configuration for connecting to the historical process data database.
  # This is typically a TimescaleDB instance.
  db:
    drivername: "postgresql+psycopg2"
    username: "postgres"
    password: "db_password"
    ip: "localhost"
    port: 5432
    db_name: "process_data"

# ----------------
# -- CoreIO --
# ----------------
coreio:
  # coreio_origin: The address for the CoreIO service.
  # Type: string (URI)
  # Default: "tcp://localhost:5559"
  coreio_origin: "tcp://localhost:5559"

  # opc_connections: A list of connections to OPC UA servers.
  opc_connections:
    - connection_id: "main_opc_server"
      opc_conn_url: "opc.tcp://192.168.1.100:4840"
      application_uri: "urn:agent"
      # security_policy: Defines the security settings for the OPC UA connection.
      security_policy:
        policy: basic256_sha256
        mode: sign_and_encrypt
        client_cert_path: "../certs/coreio_cert.der"
        client_key_path: "../certs/coreio_key.pem"
        server_cert_path: "../certs/opc_server_cert.der"
      # authentication_mode: Credentials for authenticating with the OPC UA server.
      authentication_mode:
        username: "opc_user"
        password: "opc_password"

# -----------------
# -- Interaction --
# -----------------
interaction:
  # obs_period: The time interval at which the agent observes the state of the
  # environment. Formatted as HH:MM:SS.
  obs_period: "00:05:00" # 5 minutes

  # action_period: The time interval at which the agent sends a new action
  # (setpoint) to the control system.
  action_period: "01:00:00" # 1 hour

  # update_period: The time interval for running the agent's learning algorithm.
  update_period: "00:00:30" # 30 seconds

  # warmup_period: A duration at the start of a deployment during which the agent
  # only observes data without taking actions, allowing its internal state to initialize.
  warmup_period: "03:00:00" # 3 hours

  # state_age_tol: The maximum acceptable age of the most recent data point. If data
  # is older than this, the agent will not produce a new setpoint.
  state_age_tol: "00:15:00" # 15 minutes

  # historical_batch_size: The number of data points to fetch from the historian every obs_period until all historical data is loaded.
  # Used to reduce upfront computational cost.
  historical_batch_size: 51840 # e.g., 180 days of data at a 5-minute observation period

# -----------
# -- Agent --
# -----------
agent:
  # gamma: The discount factor for future rewards, between 0 and 1. A value closer
  # to 1 gives more weight to future rewards.
  # Default: 0.9727
  gamma: 0.9727

# --------------
# -- Pipeline --
# --------------
pipeline:
  # reward: Defines the agent's reward function, which guides its learning.
  # It is structured as a list of prioritized goals.
  reward:
    priorities:
      # Priority 1: A set of soft constraints. The agent will always try to
      # satisfy these first.
      - op: "or"
        goals:
          - op: "down_to"
            tag: "H2S_OUTLET"
            thresh: 1000
          - op: "up_to"
            tag: "H2S_EFFICIENCY"
            thresh: 90
      # Priority 2: An optimization objective. Once soft constraints are met,
      # the agent will try to optimize this (e.g., minimize chemical costs).
      - tags: ["CHEM_A_USAGE", "CHEM_B_USAGE"]
        directions: "min"
        weights: [0.5, 1.5]

  # tags: A list of all data points (tags) the agent needs to read from or write to.
  # This is the most critical part of the configuration.
  # Required: Yes
  tags:
    # AI Setpoint Tag: This defines an action the agent can take.
    - name: "PH_SP"
      type: "ai_setpoint"
      connection_id: "main_opc_server"
      node_identifier: "ns=3;s=Some.Path.To.pH.Setpoint"
      operating_range: [9.0, 10.5]
      action_bounds: ["{PH_SP} - 0.2", "{PH_SP} + 0.2"]

    # Process Variable Tag: A standard input tag the agent observes.
    - name: "PH_PV"
      operating_range: [0.0, 14.0]
      expected_range: [9.0, 10.5]

    # Add all other relevant tags for state, reward, and monitoring here.
    - name: "H2S_OUTLET"
      operating_range: [0.0, 30000.0]

    - name: "H2S_EFFICIENCY"
      operating_range: [0.0, 100.0]

    - name: "CHEM_A_USAGE"
      operating_range: [0.0, 100.0]

    - name: "CHEM_B_USAGE"
      operating_range: [0.0, 100.0]
```
