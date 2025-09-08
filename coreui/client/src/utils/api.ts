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
    clean: (configName: string) => `${API_BASE_URL}/configs/${configName}`,
    raw: (configName: string) => `${API_BASE_URL}/raw-configs/${configName}`,
  },
} as const;

// Utility functions for HTTP requests
export const post = (url: string, options?: RequestInit) => fetch(url, { method: 'POST', ...options });
export const get = (url: string, options?: RequestInit) => fetch(url, { method: 'GET', ...options });
