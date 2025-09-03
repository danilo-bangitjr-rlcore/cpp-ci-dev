# Config API

This module provides a simple REST API for managing and retrieving CoreRL configuration files and tag metadata. It is intended for development and demonstration purposes.

## Config Types

There are two types of configs:

- **Clean configs**: These are YAML files that have already been type validated. They represent the config source of truth.
- **Raw configs**: These are YAML files that are a work-in-progress in the UI and have not yet been type validated.

The API exposes endpoints for both types, but the distinction is hidden from the client. The backend handles which config type to use based on the endpoint.

## Endpoints

### 1. Get Clean Config

- **GET** `/api/configs/{config_name}`
- **Description:** Retrieve the type-validated ("clean") configuration as a JSON object.
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

### 2. List Clean Tags

- **GET** `/api/configs/{config_name}/tags`
- **Description:** List all tags defined in the type-validated ("clean") configuration.
- **Response:**
  ```json
  {
    "tags": [
      { "name": "DEP_BP_FLOW_SP_WA" }
    ]
  }
  ```

### 3. Get Clean Tag

- **GET** `/api/configs/{config_name}/tags/{tag_name}`
- **Description:** Retrieve a specific tag by name from the type-validated ("clean") configuration.
- **Response:**
  ```json
  {
    "tag": { "name": "DEP_BP_FLOW_SP_WA" }
  }
  ```

### 4. Get Raw Config

- **GET** `/api/raw-configs/{config_name}`
- **Description:** Retrieve the raw (not type-validated) configuration as a JSON object.
- **Response:** Same as "Get Clean Config".

### 5. List Raw Tags

- **GET** `/api/raw-configs/{config_name}/tags`
- **Description:** List all tags defined in the raw configuration.
- **Response:** Same as "List Clean Tags".

### 6. Get Raw Tag

- **GET** `/api/raw-configs/{config_name}/tags/{tag_name}`
- **Description:** Retrieve a specific tag by name from the raw configuration.
- **Response:** Same as "Get Clean Tag".

## Mock Configs Directory

All configuration files are read from and written to the `mock_configs` directory located alongside this API module. This directory contains two subfolders:

- `clean/`: Type-validated YAML files.
- `raw/`: Raw YAML files (not type-validated, WIP in the UI).

This is a **temporary measure** for development and testing. In production, configs should be managed in a dedicated, versioned file system or configuration service on the host machine.