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

### 4. Add Clean Tag

- **POST** `/api/configs/{config_name}/tags`
- **Description:** Add a new tag to the type-validated ("clean") configuration.
- **Body:**  
  - `tag` (object): Tag object to add.
- **Response:**
  ```json
  {
    "message": "Tag created",
    "tag": { "name": "NEW_TAG" },
    "index": 2
  }
  ```

### 5. Update Clean Tag

- **PUT** `/api/configs/{config_name}/tags/{index}`
- **Description:** Update an existing tag by index in the type-validated ("clean") configuration.
- **Args:**  
  - `index` (int): Index of the tag to update.
- **Body:**  
  - `tag` (object): Updated tag object.
- **Response:**
  ```json
  {
    "message": "Tag updated",
    "tag": { "name": "UPDATED_TAG" },
    "index": 2
  }
  ```

### 6. Delete Clean Tag

- **DELETE** `/api/configs/{config_name}/tags/{index}`
- **Description:** Delete a tag by index from the type-validated ("clean") configuration.
- **Args:**  
  - `index` (int): Index of the tag to delete.
- **Response:**
  ```json
  {
    "message": "Tag deleted",
    "tag": { "name": "REMOVED_TAG" },
    "index": 2
  }
  ```

### 7. List Clean Config Names

- **GET** `/api/configs`
- **Description:** List all available type-validated ("clean") configuration names.
- **Response:**
  ```json
  {
    "configs": [
      "main_backwash",
      "secondary_config"
    ]
  }
  ```

### 8. Get Raw Config

- **GET** `/api/raw-configs/{config_name}`
- **Description:** Retrieve the raw (not type-validated) configuration as a JSON object.
- **Response:** Same as "Get Clean Config".

### 9. List Raw Tags

- **GET** `/api/raw-configs/{config_name}/tags`
- **Description:** List all tags defined in the raw configuration.
- **Response:** Same as "List Clean Tags".

### 10. Get Raw Tag

- **GET** `/api/raw-configs/{config_name}/tags/{tag_name}`
- **Description:** Retrieve a specific tag by name from the raw configuration.
- **Response:** Same as "Get Clean Tag".

### 11. Add Raw Tag

- **POST** `/api/raw-configs/{config_name}/tags`
- **Description:** Add a new tag to the raw configuration.
- **Body:**  
  - `tag` (object): Tag object to add.
- **Response:** Same as "Add Clean Tag".

### 12. Update Raw Tag

- **PUT** `/api/raw-configs/{config_name}/tags/{index}`
- **Description:** Update an existing tag by index in the raw configuration.
- **Args:**  
  - `index` (int): Index of the tag to update.
- **Body:**  
  - `tag` (object): Updated tag object.
- **Response:** Same as "Update Clean Tag".

### 13. Delete Raw Tag

- **DELETE** `/api/raw-configs/{config_name}/tags/{index}`
- **Description:** Delete a tag by index from the raw configuration.
- **Args:**  
  - `index` (int): Index of the tag to delete.
- **Response:** Same as "Delete Clean Tag".

### 14. List Raw Config Names

- **GET** `/api/raw-configs`
- **Description:** List all available raw configuration names.
- **Response:** Same as "List Clean Config Names".

## Mock Configs Directory

All configuration files are read from and written to the `mock_configs` directory located alongside this API module. This directory contains two subfolders:

- `clean/`: Type-validated YAML files.
- `raw/`: Raw YAML files (not type-validated, WIP in the UI).

This is a **temporary measure** for development and testing. In production, configs should be managed in a dedicated, versioned file system or configuration service on the host machine.