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

const fetchRawConfig = async (configName: string): Promise<Record<string, any>> => {
  const response = await get(API_ENDPOINTS.configs.raw(configName));
  if (!response.ok) {
    throw new Error(`Failed to fetch raw config for ${configName}`);
  }
  const data: { config: Record<string, any> } = await response.json();
  return data.config;
};

// Hook for fetching config list
export const useConfigListQuery = () => {
  return useQuery({
    queryKey: ['configList'],
    queryFn: fetchConfigList,
  });
};

// Hook for fetching all raw configs
export const useRawConfigsQueries = (configNames: string[] | undefined) => {
  return useQueries({
    queries: (configNames || []).map((configName) => ({
      queryKey: ['rawConfig', configName],
      queryFn: () => fetchRawConfig(configName),
      enabled: !!configNames,
    })),
  });
};
