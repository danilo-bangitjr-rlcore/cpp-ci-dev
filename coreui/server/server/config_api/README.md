# Config API

This module provides a simple REST API for managing and retrieving CoreRL configuration files and tag metadata. It is intended for development and demonstration purposes.

## Endpoints

### 1. Get Config

- **GET** `/api/configs/{config_name}`
- **Description:** Retrieve the configuration as a JSON object.
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
- **Description:** Retrieve a specific tag by name from the configuration.
- **Response:**
  ```json
  {
    "tag": { "name": "DEP_BP_FLOW_SP_WA" }
  }
  ```

### 4. Add Tag

- **POST** `/api/configs/{config_name}/tags`
- **Description:** Add a new tag to the configuration.
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

### 5. Update Tag

- **PUT** `/api/configs/{config_name}/tags/{index}`
- **Description:** Update an existing tag by index in the configuration.
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

### 6. Delete Tag

- **DELETE** `/api/configs/{config_name}/tags/{index}`
- **Description:** Delete a tag by index from the configuration.
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

### 7. List Config Names

- **GET** `/api/configs/list`
- **Description:** List all available configuration names.
- **Response:**
  ```json
  {
    "configs": [
      "main_backwash",
      "secondary_config"
    ]
  }
  ```

### 8. Create Config

- **POST** `/api/configs/configs`
- **Description:** Create a new configuration with the given name.
- **Body:**  
  - `config_name` (str): Name of the new configuration.
- **Response:**
  ```json
  {
    "message": "Config created",
    "config": { "agent_name": "new_config" },
    "name": "new_config"
  }
  ```

### 9. Delete Config

- **DELETE** `/api/configs/configs`
- **Description:** Delete an existing configuration by name.
- **Body:**  
  - `config_name` (str): Name of the configuration to delete.
- **Response:**
  ```json
  {
    "message": "Config deleted",
    "name": "deleted_config"
  }
  ```

### 10. Get Config File Path

- **GET** `/api/configs/{config_name}/config_path`
- **Description:** Retrieve the file path of the specified configuration.
- **Response:**
  ```json
  {
    "config_path": "/path/to/configs/main_backwash.yaml"
  }
  ```

### 11. Get Agents Missing Config

- **GET** `/api/configs/agents/missing-config`
- **Description:** Retrieve agents that are active in coredinator but do not have a configuration available.
- **Response:**
  ```json
  {
    "agents": ["orphan_agent_1", "orphan_agent_2"]
  }
  ```

## Mock Configs Directory

All configuration files are read from and written to the `mock_configs` directory located alongside this API module.

This is a **temporary measure** for development and testing. In production, configs should be managed in a dedicated, versioned file system or configuration service on the host machine.