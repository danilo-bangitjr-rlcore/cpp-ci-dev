import { createFileRoute } from '@tanstack/react-router';
import { useQuery } from '@tanstack/react-query';
import { API_ENDPOINTS } from '../utils/api';
import { OpcConnectionCard } from '../components/OpcConnectionCard';

export const Route = createFileRoute('/opc-navigation')({
  component: RouteComponent,
});

interface StatusResponse {
  connected: boolean;
  server_url: string | null;
  message?: string;
  error?: string;
  server_info?: {
    node_id: string;
    display_name: string;
  };
}

function RouteComponent() {
  // Shared OPC status query that can be used by all components in this route
  const {
    data: statusData,
    error: statusError,
    isLoading: isStatusLoading,
  } = useQuery<StatusResponse>({
    queryKey: ['opc-status'],
    queryFn: async () => {
      const response = await fetch(API_ENDPOINTS.opc.status);
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(
          errorData.detail || `HTTP error! status: ${response.status}`
        );
      }
      return response.json();
    },
    refetchInterval: (query) => {
      const data = query.state.data;
      return data?.server_url ? 10000 : false;
    },
    retry: 1,
    enabled: true,
  });

  return (
    <div className="p-2 min-h-screen bg-gray-200 rounded-lg">
      <OpcConnectionCard
        statusData={statusData}
        statusError={statusError}
        isStatusLoading={isStatusLoading}
      />

      {/* Future components can also use the status data */}
      {/* statusData?.connected && <OpcDataViewer statusData={statusData} /> */}
    </div>
  );
}
