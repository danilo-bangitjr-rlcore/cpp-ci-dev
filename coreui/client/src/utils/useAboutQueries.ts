import { useQuery } from '@tanstack/react-query';
import { API_ENDPOINTS } from './api';

const REFETCH_INTERVAL = 5000; // 5 seconds

export const useUiHealthQuery = () =>
  useQuery({
    queryKey: ['uiHealth'],
    queryFn: async () => {
      const response = await fetch(API_ENDPOINTS.health);
      if (!response.ok) throw new Error('Health check failed');
      return response.json();
    },
    refetchInterval: REFETCH_INTERVAL,
  });

export const useSystemHealthQuery = () =>
  useQuery({
    queryKey: ['systemHealth'],
    queryFn: async () => {
      const response = await fetch(API_ENDPOINTS.system_metrics.health);
      if (!response.ok) throw new Error('Health check failed');
      return response.json();
    },
    refetchInterval: REFETCH_INTERVAL,
  });

export const useSystemPlatformQuery = () =>
  useQuery({
    queryKey: ['systemPlatform'],
    queryFn: async () => {
      const response = await fetch(API_ENDPOINTS.system_metrics.platform);
      if (!response.ok) throw new Error('Failed to fetch platform');
      return response.json();
    },
    refetchInterval: REFETCH_INTERVAL,
  });

export const useSystemCpuQuery = () =>
  useQuery({
    queryKey: ['systemCpu'],
    queryFn: async () => {
      const response = await fetch(API_ENDPOINTS.system_metrics.cpu);
      if (!response.ok) throw new Error('Failed to fetch CPU');
      return response.json();
    },
    refetchInterval: REFETCH_INTERVAL,
  });

export const useSystemCpuPerCoreQuery = () =>
  useQuery({
    queryKey: ['systemCpuPerCore'],
    queryFn: async () => {
      const response = await fetch(API_ENDPOINTS.system_metrics.cpu_per_core);
      if (!response.ok) throw new Error('Failed to fetch CPU per core');
      return response.json();
    },
    refetchInterval: REFETCH_INTERVAL,
  });

export const useSystemRamQuery = () =>
  useQuery({
    queryKey: ['systemRam'],
    queryFn: async () => {
      const response = await fetch(API_ENDPOINTS.system_metrics.ram);
      if (!response.ok) throw new Error('Failed to fetch RAM');
      return response.json();
    },
    refetchInterval: REFETCH_INTERVAL,
  });

export const useSystemDiskQuery = () =>
  useQuery({
    queryKey: ['systemDisk'],
    queryFn: async () => {
      const response = await fetch(API_ENDPOINTS.system_metrics.disk);
      if (!response.ok) throw new Error('Failed to fetch disk');
      return response.json();
    },
    refetchInterval: REFETCH_INTERVAL,
  });
