import { useQuery, useQueries } from '@tanstack/react-query';
import { API_ENDPOINTS, get } from '../../utils/api';

// Config API functions
const fetchConfigList = async (): Promise<string[]> => {
  const response = await get(API_ENDPOINTS.configs.list_raw);
  if (!response.ok) {
    throw new Error('Failed to fetch config list');
  }
  const data: { configs: string[] } = await response.json();
  return data.configs;
};

const fetchRawConfig = async (
  configName: string
): Promise<Record<string, any>> => {
  const response = await get(API_ENDPOINTS.configs.raw(configName));
  if (!response.ok) {
    throw new Error(`Failed to fetch raw config for ${configName}`);
  }
  const data: { config: Record<string, any> } = await response.json();
  return data.config;
};

const fetchAgentName = async (configName: string): Promise<string> => {
  const response = await get(API_ENDPOINTS.configs.agent_name(configName));
  if (!response.ok) {
    throw new Error(`Failed to fetch agent name for ${configName}`);
  }
  const data: { agent_name: string } = await response.json();
  return data.agent_name;
};

// Hook for fetching config list
export const useConfigListQuery = () => {
  return useQuery({
    queryKey: ['configList'],
    queryFn: fetchConfigList,
  });
};

// Hook for fetching all raw configs
export const useRawConfigsQueries = (configNames?: string[]) => {
  const names = configNames ?? [];
  return useQueries({
    queries: names.map((name) => ({
      queryKey: ['rawConfig', name],
      queryFn: () => fetchRawConfig(name),
      enabled: names.length > 0,
    })),
  });
};

// Hook for fetching all agent names
export const useAgentNamesQueries = (configNames?: string[]) => {
  const names = configNames ?? [];
  return useQueries({
    queries: names.map((name) => ({
      queryKey: ['agentName', name],
      queryFn: () => fetchAgentName(name),
      enabled: names.length > 0,
    })),
  });
};

// Hook for fetching a single agent name
export const useAgentNameQuery = (configName?: string) => {
  return useQuery({
    queryKey: ['agentName', configName],
    queryFn: () => fetchAgentName(configName!),
    enabled: !!configName,
  });
};
