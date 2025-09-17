import { useQuery } from '@tanstack/react-query';
import { API_ENDPOINTS, get } from '../../utils/api';

export function useAgentExists(agentName: string) {
  return useQuery({
    queryKey: ['agent-exists', agentName],
    queryFn: async () => {
      if (!agentName) return false;
      const response = await get(API_ENDPOINTS.configs.raw(agentName));
      if (response.status === 404) return false;
      if (response.ok) return true;
      throw new Error(
        `Error checking agent: ${response.status} ${response.statusText}`
      );
    },
    enabled: !!agentName,
  });
}
