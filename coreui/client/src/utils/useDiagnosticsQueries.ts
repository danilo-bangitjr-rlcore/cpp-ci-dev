import { useQuery } from '@tanstack/react-query';
import { API_ENDPOINTS } from './api';

const REFETCH_INTERVAL = 5000; // 5 seconds

const useSystemMetricQuery = (key: string, endpoint: string) =>
  useQuery({
    queryKey: [key],
    queryFn: async () => {
      const response = await fetch(endpoint);
      if (!response.ok) throw new Error(`Failed to fetch ${key}`);
      return response.json();
    },
    refetchInterval: REFETCH_INTERVAL,
    refetchIntervalInBackground: false,
  });

export const useUiHealthQuery = () =>
  useSystemMetricQuery('uiHealth', API_ENDPOINTS.health);

export const useSystemHealthQuery = () =>
  useSystemMetricQuery('systemHealth', API_ENDPOINTS.system_metrics.health);

export const useSystemMetricsQuery = () =>
  useSystemMetricQuery('systemMetrics', API_ENDPOINTS.system_metrics.all);
