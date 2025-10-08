import { useMutation, useQueryClient } from '@tanstack/react-query';
import { API_ENDPOINTS, del, post } from './api';

// Delete config mutation
const deleteConfig = async (configName: string): Promise<void> => {
  const response = await del(API_ENDPOINTS.configs.mutate_raw_config, {
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      config_name: configName,
    }),
  });

  if (!response.ok) {
    throw new Error(`Failed to delete config ${configName}`);
  }
};

export const useDeleteAgentMutation = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: deleteConfig,
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ['configList'] });
    },
    onError: (error, configName) => {
      console.error(`Failed to delete agent ${configName}:`, error);
    },
  });
};

// Add config mutation
const addConfig = async (configName: string): Promise<void> => {
  const response = await post(API_ENDPOINTS.configs.mutate_raw_config, {
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      config_name: configName,
    }),
  });

  if (!response.ok) {
    let message = '';
    try {
      const data = await response.json();
      if (data && typeof data.error === 'string') {
        message = data.error;
      }
    } catch {
      console.log('Failed to parse error response as JSON');
    }
    throw new Error(message);
  }
};

const startAgent = async (config_path: string, coreio_id?: string): Promise<void> => {
  const body = coreio_id
    ? { config_path, coreio_id }
    : { config_path };

  const response = await post(API_ENDPOINTS.coredinator.start_agent, {
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    throw new Error(`Failed to start agent with config path ${config_path}`);
  }
};

const stopAgent = async (agentId: string): Promise<void> => {
  const response = await post(API_ENDPOINTS.coredinator.stop_agent(agentId), {
    headers: {
      'Content-Type': 'application/json',
    },
  });

  if (!response.ok) {
    throw new Error(`Failed to stop agent with ID ${agentId}`);
  }
};

const startIO = async (configPath: string, ioId?: string): Promise<void> => {
  const body = ioId
    ? { config_path: configPath, io_id: ioId }
    : { config_path: configPath };

  const response = await post(API_ENDPOINTS.coredinator.start_io, {
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    throw new Error(`Failed to start I/O with config path ${configPath}`);
  }
};

const stopIO = async (ioId: string): Promise<void> => {
  const response = await post(API_ENDPOINTS.coredinator.stop_io(ioId), {
    headers: {
      'Content-Type': 'application/json',
    },
  });

  if (!response.ok) {
    throw new Error(`Failed to stop I/O with ID ${ioId}`);
  }
};

export const useAddAgentMutation = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: addConfig,
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ['configList'] });
    },
    onError: (error, configName) => {
      console.error(`Failed to add agent ${configName}:`, error);
    },
  });
};

export const useAgentToggleMutation = (
  configPath: string,
  agentId: string,
  existingIOId?: string
) => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({ action }: { action: 'start' | 'stop' }) => {
      if (action === 'start') {
        await startAgent(configPath, existingIOId || agentId);
      } else {
        await stopAgent(agentId);
      }
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ['agent-status', configPath] });
    },
    onError: (error, { action }) => {
      console.error(`Failed to ${action} agent ${configPath}:`, error);
    },
  });
};

export const useIOToggleMutation = (configPath: string, ioId: string) => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({ action }: { action: 'start' | 'stop' }) => {
      if (action === 'start') {
        await startIO(configPath, ioId);
      } else {
        await stopIO(ioId);
      }
    },
    onSettled: () => {
      // Invalidate both agent status and IO status queries
      queryClient.invalidateQueries({ queryKey: ['agent-status', configPath] });
      if (ioId) {
        queryClient.invalidateQueries({ queryKey: ['io-status', ioId] });
      }
    },
    onError: (error, { action }) => {
      console.error(`Failed to ${action} I/O ${configPath}:`, error);
    },
  });
};
