import { createFileRoute } from '@tanstack/react-router';
import { useQuery } from '@tanstack/react-query';
import { ExampleComponent } from '../components/ExampleComponent';
import { API_ENDPOINTS } from '../utils/api';

export const Route = createFileRoute('/about')({
  component: About,
});

function About() {
  // Health check using TanStack Query
  const {
    data: healthData,
    isLoading: isHealthLoading,
    error: healthError,
  } = useQuery({
    queryKey: ['health'],
    queryFn: async () => {
      const response = await fetch(API_ENDPOINTS.health);
      if (!response.ok) {
        throw new Error('Health check failed');
      }
      return response.json();
    },
    refetchInterval: 10000, // Check every 10 seconds
  });

  return (
    // h-full so this gray panel fills the available vertical space inside the flex main container
    <div className="p-2 h-full bg-gray-200 rounded-lg flex flex-col">
      <ExampleComponent title="Hello from the about page!" />

      {/* Health Status */}
      <div className="mt-4 p-4 bg-white rounded-lg border border-gray-200 shadow-sm">
        <h2 className="text-lg font-semibold text-gray-900 mb-2">
          Server Status
        </h2>
        <div className="flex items-center gap-2">
          <span className="text-sm text-gray-600">Health Check:</span>
          <span
            className={`px-2 py-1 rounded text-xs font-medium ${
              isHealthLoading
                ? 'bg-yellow-100 text-yellow-800 border border-yellow-200'
                : healthError
                  ? 'bg-red-100 text-red-800 border border-red-200'
                  : healthData?.status === 'ok'
                    ? 'bg-green-100 text-green-800 border border-green-200'
                    : 'bg-gray-100 text-gray-600 border border-gray-200'
            }`}
          >
            {isHealthLoading
              ? 'Checking...'
              : healthError
                ? 'Offline'
                : healthData?.status === 'ok'
                  ? 'Online'
                  : 'Unknown'}
          </span>
        </div>
        {healthData && (
          <div className="mt-2 text-xs text-gray-500 bg-gray-50 p-2 rounded border inline-block">
            Response: {JSON.stringify(healthData)}
          </div>
        )}
      </div>
    </div>
  );
}
