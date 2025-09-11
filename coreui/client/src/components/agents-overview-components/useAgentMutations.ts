import { useMutation, useQueryClient } from '@tanstack/react-query';
import { API_ENDPOINTS, del } from '../../utils/api';

// Delete config mutation
const deleteConfig = async (configName: string): Promise<void> => {
  const response = await del(API_ENDPOINTS.configs.delete_raw_config, {
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
    onSuccess: (_, configName) => {
      // Invalidate and refetch the config list
      queryClient.invalidateQueries({ queryKey: ['configList'] });
      // Also invalidate individual config queries
      queryClient.invalidateQueries({ queryKey: ['rawConfig', configName] });
    },
    onError: (error, configName) => {
      console.error(`Failed to delete agent ${configName}:`, error);
    },
  });
};
