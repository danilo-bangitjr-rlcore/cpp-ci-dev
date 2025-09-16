import { useQuery } from '@tanstack/react-query';
import { API_ENDPOINTS, get } from './api';

export function useConfigExists(configName: string) {
  return useQuery({
    queryKey: ['config-exists', configName],
    queryFn: async () => {
      if (!configName) return false;
      const response = await get(API_ENDPOINTS.configs.raw(configName));
      if (response.status === 404) return false;
      if (response.ok) return true;
      throw new Error(
        `Error checking agent: ${response.status} ${response.statusText}`
      );
    },
    enabled: !!configName,
  });
}
