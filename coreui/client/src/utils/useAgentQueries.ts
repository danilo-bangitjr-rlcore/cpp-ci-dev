import {
  useQuery,
  useQueries,
  useMutation,
  useQueryClient,
  type UseQueryOptions,
} from '@tanstack/react-query';
import { API_ENDPOINTS, get } from './api';

// Define the agent status response type
type AgentStatusResponse = {
  state: string;
  configPath: string;
  serviceStatuses: Record<string, string>;
  id: string;
  version: string;
  uptime: string;
};

// Mock state storage - in real implementation this would be handled by the backend
const mockAgentStates = new Map<string, 'running' | 'stopped'>();
const mockIOStates = new Map<string, 'running' | 'stopped'>();

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
  const data: {
    state: string;
    config_path: string;
    service_statuses: Record<string, string>;
    id: string;
  } = await response.json();
  return {
    state: data.state,
    configPath: data.config_path,
    serviceStatuses: data.service_statuses,
    id: data.id,
    version: '1.0.0',
    uptime: data.state === 'running' ? '5 hours' : '0 minutes',
  };
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

// Hook for fetching single agent status
export const useAgentStatusQuery = (configName: string) => {
  return useQuery({
    queryKey: ['agent-status', configName],
    queryFn: () => fetchAgentStatus(configName),
    enabled: !!configName,
    refetchInterval: 60000, // Poll every 1 minute
    staleTime: 30000, // Data considered fresh for 30s
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
        refetchInterval: isPolling ? 60000 : false, // Only poll when isPolling is true
        staleTime: 30000, // Data considered fresh for 30s
      })
    ),
  });
};

export const useAgentToggleMutation = (configName: string) => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({
      configName,
      action,
    }: {
      configName: string;
      action: 'start' | 'stop';
    }) => {
      await new Promise((resolve) => setTimeout(resolve, 300));

      // Update mock state
      const newState = action === 'start' ? 'running' : 'stopped';
      mockAgentStates.set(configName, newState);

      return {
        success: true,
        message: `Agent ${configName} ${action}ed successfully`,
        state: newState,
      };
    },
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: ['agent-status', configName],
      });
    },
  });
};

export const useIOStatusQuery = (configName: string) => {
  return useQuery({
    queryKey: ['io-status', configName],
    queryFn: async () => {
      await new Promise((resolve) => setTimeout(resolve, 300));

      const currentState = mockIOStates.get(configName) ?? 'stopped';

      return {
        state: currentState,
      };
    },
    enabled: true,
  });
};

export const useIOToggleMutation = (configName: string) => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({
      configName,
      action,
    }: {
      configName: string;
      action: 'start' | 'stop';
    }) => {
      await new Promise((resolve) => setTimeout(resolve, 300));

      const newState = action === 'start' ? 'running' : 'stopped';
      mockIOStates.set(configName, newState);

      return {
        success: true,
        message: `I/O service ${configName} ${action}ed successfully`,
        state: newState,
      };
    },
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: ['io-status', configName],
      });
    },
  });
};
