# Config API

This module provides a simple REST API for managing and retrieving CoreRL configuration files and tag metadata. It is intended for development and demonstration purposes.

## Endpoints

### 1. Get Config

- **GET** `/api/configs/{config_name}`
- **Description:** Retrieve the full configuration as a JSON object.
- **Args:**  
  - `config_name` (str): Name of the config file (without `.yml` extension)
- **Response:**
  ```json
  {
    "config": {
      "agent_name": "main_backwash",
      "infra": { ... }
    }
  }
  ```

### 2. List Tags

- **GET** `/api/configs/{config_name}/tags`
- **Description:** List all tags defined in the configuration.
- **Response:**
  ```json
  {
    "tags": [
      { "name": "DEP_BP_FLOW_SP_WA" }
    ]
  }
  ```

### 3. Get Tag

- **GET** `/api/configs/{config_name}/tags/{tag_name}`
- **Description:** Retrieve a specific tag by name.
- **Response:**
  ```json
  {
    "tag": { "name": "DEP_BP_FLOW_SP_WA" }
  }
  ```

## Mock Configs Directory

All configuration files are read from and written to the `mock_configs` directory located alongside this API module. This is a **temporary measure** for development and testing. In production, configs should be managed in a dedicated, versioned file system or configuration service on the host machine.

- **Location:**  
  [`mock_configs/`](mock_configs/)