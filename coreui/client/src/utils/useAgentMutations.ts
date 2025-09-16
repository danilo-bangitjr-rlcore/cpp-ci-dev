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
