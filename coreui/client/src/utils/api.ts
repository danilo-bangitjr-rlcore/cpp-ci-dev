export const API_BASE_URL =
  import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

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
    mutate_raw_config: `${API_BASE_URL}/v1/config/raw/configs`,
    // TEMPORARY PLACEHOLDER FOR TAGS -- HARD CODED CONFIG NAME
    tags: `${API_BASE_URL}/v1/config/main_backwash/tags`,
  },
} as const;

// Utility functions for HTTP requests
export const post = (url: string, options?: RequestInit) =>
  fetch(url, { method: 'POST', ...options });
export const get = (url: string, options?: RequestInit) =>
  fetch(url, { method: 'GET', ...options });
export const del = (url: string, options?: RequestInit) =>
  fetch(url, { method: 'DELETE', ...options });
