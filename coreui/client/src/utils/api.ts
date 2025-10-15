export const API_BASE_URL =
  import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

const COREGATEWAY_BASE_URL = 'http://localhost:8001/api';

// Optional: More specific endpoints for better organization
export const API_ENDPOINTS = {
  health: `${API_BASE_URL}/health`,
  opc: {
    connect: `${API_BASE_URL}/v1/opc/connect`,
    disconnect: `${API_BASE_URL}/v1/opc/disconnect`,
    status: `${API_BASE_URL}/v1/opc/status`,
  },
  configs: {
    list_raw: `${API_BASE_URL}/v1/config/raw/list`,
    list_clean: `${API_BASE_URL}/v1/config/clean/list`,
    clean: (configName: string) => `${API_BASE_URL}/configs/${configName}`,
    raw: (configName: string) => `${API_BASE_URL}/v1/config/raw/${configName}`,
    agent_name: (configName: string) =>
      `${API_BASE_URL}/v1/config/raw/${configName}/agent_name`,
    mutate_raw_config: `${API_BASE_URL}/v1/config/raw/configs`,
    // TEMPORARY PLACEHOLDER FOR TAGS -- HARD CODED CONFIG NAME
    tags: `${API_BASE_URL}/v1/config/raw/main_backwash/tags`,
    delete_raw_tag: (configName: string, tagIndex: number) =>
      `${API_BASE_URL}/v1/config/raw/${configName}/tags/${tagIndex}`,
    add_raw_tag: (configName: string) =>
      `${API_BASE_URL}/v1/config/raw/${configName}/tags`,
    update_raw_tag: (configName: string, tagIndex: number) =>
      `${API_BASE_URL}/v1/config/raw/${configName}/tags/${tagIndex}`,
    get_clean_config_path: (configName: string) =>
      `${API_BASE_URL}/v1/config/${configName}/config_path`,
    agents_missing_config: `${API_BASE_URL}/v1/config/agents/missing-config`,
  },
  coredinator: {
    agent_status: (agentId: string) =>
      `${COREGATEWAY_BASE_URL}/v1/coredinator/api/agents/${agentId}/status`,
    io_status: (ioId: string) =>
      `${COREGATEWAY_BASE_URL}/v1/coredinator/api/io/${ioId}/status`,
    start_agent: `${COREGATEWAY_BASE_URL}/v1/coredinator/api/agents/start`,
    stop_agent: (agentId: string) =>
      `${COREGATEWAY_BASE_URL}/v1/coredinator/api/agents/${agentId}/stop`,
    start_io: `${COREGATEWAY_BASE_URL}/v1/coredinator/api/io/start`,
    stop_io: (ioId: string) =>
      `${COREGATEWAY_BASE_URL}/v1/coredinator/api/io/${ioId}/stop`,
    list_io: `${COREGATEWAY_BASE_URL}/v1/coredinator/api/io/`,
  },
} as const;

// Utility functions for HTTP requests
export const post = (url: string, options?: RequestInit) =>
  fetch(url, { method: 'POST', ...options });
export const get = (url: string, options?: RequestInit) =>
  fetch(url, { method: 'GET', ...options });
export const del = (url: string, options?: RequestInit) =>
  fetch(url, { method: 'DELETE', ...options });
