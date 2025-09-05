import { useQuery, useQueries } from '@tanstack/react-query';
import { API_BASE_URL } from '../../utils/api';
import type { NodeInfo, NodeDetails } from './types';

export const useOpcData = (
  expandedNodes: Set<string>,
  selectedNodeId: string | null
) => {
  // Fetch root nodes
  const { data: rootNodes, isLoading: isLoadingRoot } = useQuery({
    queryKey: ['opc-browse-root'],
    queryFn: async (): Promise<NodeInfo[]> => {
      const response = await fetch(`${API_BASE_URL}/v1/opc/browse`);
      if (!response.ok) throw new Error('Failed to fetch root nodes');
      return response.json();
    },
    enabled: true,
  });

  // Fetch children of expanded nodes using useQueries (top-level hooks)
  const childrenQueries = useQueries({
    queries: Array.from(expandedNodes).map((nodeId) => ({
      queryKey: ['opc-browse-node', nodeId],
      queryFn: async (): Promise<NodeInfo[]> => {
        const response = await fetch(`${API_BASE_URL}/v1/opc/browse/${nodeId}`);
        if (!response.ok)
          throw new Error(`Failed to fetch children of ${nodeId}`);
        return response.json();
      },
      enabled: true,
    })),
  });

  // Create a map for quick lookup of query results by nodeId
  const childrenMap = new Map(
    Array.from(expandedNodes).map((nodeId, index) => [
      nodeId,
      childrenQueries[index],
    ])
  );

  // Fetch details of selected node
  const { data: selectedNodeDetails, isLoading: isLoadingDetails } = useQuery<
    NodeDetails | undefined
  >({
    queryKey: ['opc-node-details', selectedNodeId],
    queryFn: async (): Promise<NodeDetails> => {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 3000); // Reduced to 3 seconds

      try {
        const response = await fetch(
          `${API_BASE_URL}/v1/opc/node/${selectedNodeId}`,
          {
            signal: controller.signal,
          }
        );
        clearTimeout(timeoutId);

        if (!response.ok) throw new Error('Failed to fetch node details');
        return await response.json();
      } catch (error) {
        clearTimeout(timeoutId);
        if ((error as Error).name === 'AbortError') {
          throw new Error('Request timed out');
        }
        throw error;
      }
    },
    enabled: !!selectedNodeId,
    staleTime: 60000, // Cache for 1 minute
    gcTime: 300000, // Keep in cache for 5 minutes
    retry: 1,
    retryDelay: 500, // Faster retry
    refetchOnWindowFocus: false, // Don't refetch when window regains focus
    refetchOnReconnect: false, // Don't refetch on reconnect
    refetchInterval: selectedNodeId ? 1000 : false, // Poll every 1 second when a node is selected
    refetchIntervalInBackground: false, // Don't poll when window is not focused
  });

  return {
    rootNodes,
    isLoadingRoot,
    childrenMap,
    selectedNodeDetails,
    isLoadingDetails,
  };
};
