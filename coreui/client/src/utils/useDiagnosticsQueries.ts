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

export const useSystemPlatformQuery = () =>
  useSystemMetricQuery('systemPlatform', API_ENDPOINTS.system_metrics.platform);

export const useSystemCpuQuery = () =>
  useSystemMetricQuery('systemCpu', API_ENDPOINTS.system_metrics.cpu);

export const useSystemCpuPerCoreQuery = () =>
  useSystemMetricQuery(
    'systemCpuPerCore',
    API_ENDPOINTS.system_metrics.cpu_per_core
  );

export const useSystemRamQuery = () =>
  useSystemMetricQuery('systemRam', API_ENDPOINTS.system_metrics.ram);

export const useSystemDiskQuery = () =>
  useSystemMetricQuery('systemDisk', API_ENDPOINTS.system_metrics.disk);
