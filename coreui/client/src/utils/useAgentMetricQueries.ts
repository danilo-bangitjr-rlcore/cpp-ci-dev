import { useQuery, useQueries } from '@tanstack/react-query';
import { API_ENDPOINTS, get } from './api';

const fetchAgentMetrics = async (
  agentId: string,
  metric: string,
  start_time?: string,
  end_time?: string
): Promise<any> => {
  const params = new URLSearchParams({ metric });
  if (start_time) params.append('start_time', start_time);
  if (end_time) params.append('end_time', end_time);

  const url = `${API_ENDPOINTS.coretelemetry.agent_metrics(agentId)}?${params.toString()}`;
  const response = await get(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch agent metrics for ${agentId}`);
  }
  const data = await response.json();
  return data;
};

export const useAgentMetricsQuery = (
  agentId: string,
  metric: string,
  start_time?: string,
  end_time?: string,
  enabled: boolean = true
) => {
  return useQuery({
    queryKey: ['agent-metrics', agentId, metric, start_time, end_time],
    queryFn: () => fetchAgentMetrics(agentId, metric, start_time, end_time),
    enabled: enabled && !!agentId && !!metric,
  });
};

const fetchAvailableMetrics = async (agentId: string): Promise<string[]> => {
  const url = API_ENDPOINTS.coretelemetry.available_metrics(agentId);
  const response = await get(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch available metrics for ${agentId}`);
  }
  const data: { agent_id: string; data: string[] } = await response.json();
  return data.data;
};

export const useAvailableMetricsQuery = (
  agentId: string,
  enabled: boolean = true
) => {
  return useQuery({
    queryKey: ['available-metrics', agentId],
    queryFn: () => fetchAvailableMetrics(agentId),
    enabled: enabled && !!agentId,
  });
};

interface FilteredMetric {
  name: string;
  description: string;
}

const fetchFilteredMetrics = async (
  agentId: string
): Promise<FilteredMetric[]> => {
  const url = API_ENDPOINTS.coretelemetry.filtered_metrics(agentId);
  const response = await get(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch filtered metrics for ${agentId}`);
  }
  const data: { agent_id: string; data: FilteredMetric[] } =
    await response.json();
  return data.data;
};

export const useFilteredMetricsQuery = (
  agentId: string,
  enabled: boolean = true
) => {
  return useQuery({
    queryKey: ['filtered-metrics', agentId],
    queryFn: () => fetchFilteredMetrics(agentId),
    enabled: enabled && !!agentId,
  });
};

export const useMultipleAgentMetricsQueries = (
  agentId: string,
  metrics: string[],
  enabled: boolean = true
) => {
  return useQueries({
    queries: metrics.map((metric: string) => ({
      queryKey: ['agent-metrics', agentId, metric],
      queryFn: async () => {
        const url = `${API_ENDPOINTS.coretelemetry.agent_metrics(agentId)}?metric=${metric}`;
        const response = await get(url);
        if (!response.ok) {
          throw new Error(`Failed to fetch ${metric}`);
        }
        return response.json();
      },
      enabled: enabled && !!agentId && !!metric,
      refetchInterval: 5000,
    })),
  });
};
