import {
  useQuery,
  useQueries,
  type UseQueryOptions,
} from '@tanstack/react-query';
import { API_ENDPOINTS, get } from './api';
import type { AgentStatusResponse, IOStatusResponse, IOListResponse } from '../types/agent-types';

// Config API functions (unchanged)
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

const fetchIOStatus = async (ioName: string): Promise<IOStatusResponse> => {
  const response = await get(API_ENDPOINTS.coredinator.io_status(ioName));
  if (!response.ok) {
    throw new Error(`Failed to fetch I/O status for ${ioName}`);
  }
  const data: IOStatusResponse = await response.json();
  return data;
};

// Shared agent status fetch function
const fetchAgentStatus = async (
  configName: string
): Promise<AgentStatusResponse> => {
  const response = await get(
    API_ENDPOINTS.coredinator.agent_status(configName)
  );
  if (!response.ok) {
    throw new Error(`Failed to fetch status for agent ${configName}`);
  }
  const data: AgentStatusResponse = await response.json();
  return {
    state: data.state,
    config_path: data.config_path,
    service_statuses: data.service_statuses,
    id: data.id,
    version: '1.0.0',
    uptime: data.state === 'running' ? '5 hours' : '0 minutes',
  };
};

const fetchAgentsMissingConfig = async (): Promise<string[]> => {
  const response = await get(API_ENDPOINTS.coredinator.agents_missing_config);
  if (!response.ok) {
    throw new Error('Failed to fetch agents missing config');
  }
  const data: { agents: string[] } = await response.json();
  return data.agents;
};

const fetchConfigPath = async (configName: string): Promise<string> => {
  const response = await get(
    API_ENDPOINTS.configs.get_clean_config_path(configName)
  );
  if (!response.ok) {
    throw new Error(`Failed to fetch config path for ${configName}`);
  }
  const data: { config_path: string } = await response.json();
  return data.config_path;
};


export const fetchIOs = async (): Promise<IOListResponse> => {
  const response = await get(API_ENDPOINTS.coredinator.list_io);
  if (!response.ok) {
    throw new Error('Failed to fetch I/O list');
  }
  const data: IOListResponse = await response.json();
  return data;
}

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

// Hook for fetching single agent status
export const useAgentStatusQuery = (configName: string, isPolling: boolean) => {
  return useQuery({
    queryKey: ['agent-status', configName],
    queryFn: () => fetchAgentStatus(configName),
    enabled: !!configName,
    refetchInterval: isPolling ? 60000 : false,
    staleTime: 30000,
  });
};

// Hook for fetching multiple agent statuses
export const useAgentStatusQueries = (
  configNames: string[],
  isPolling: boolean = true
) => {
  return useQueries({
    queries: configNames.map(
      (configName): UseQueryOptions<AgentStatusResponse, Error> => ({
        queryKey: ['agent-status', configName],
        queryFn: () => fetchAgentStatus(configName),
        enabled: true,
        refetchInterval: isPolling ? 60000 : false,
        staleTime: 30000,
      })
    ),
  });
};

export const useIOStatusQuery = (ioName: string) => {
  return useQuery({
    queryKey: ['io-status', ioName],
    queryFn: async () => {
      const data = await fetchIOStatus(ioName);
      return data;
    },
    enabled: true,
    refetchInterval: 60000,
    staleTime: 30000,
  });
};

export const useAgentsMissingConfigQuery = (isPolling: boolean = true) => {
  return useQuery({
    queryKey: ['agents-missing-config'],
    queryFn: fetchAgentsMissingConfig,
    refetchInterval: isPolling ? 60000 : false,
    staleTime: 30000,
  });
};

export const useConfigPathQuery = (configName: string, enabled: boolean = true) => {
  return useQuery({
    queryKey: ['config-path', configName],
    queryFn: () => fetchConfigPath(configName),
    enabled: !!configName && enabled,
  });
};

export const useIOListQuery = (enabled: boolean = true) => {
  return useQuery({
    queryKey: ['io-list'],
    queryFn: fetchIOs,
    enabled,
  });
};