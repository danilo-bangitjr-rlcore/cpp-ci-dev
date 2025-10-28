import { createFileRoute, useParams } from '@tanstack/react-router';
import {
  useFilteredMetricsQuery,
  useMultipleAgentMetricsQueries,
} from '../../../utils/useAgentMetricQueries';

export const Route = createFileRoute('/agents/$config-name/monitor')({
  component: RouteComponent,
});

// Constants
const MAX_DECIMAL_PLACES = 4;


// Format timestamp helper
const formatTimestamp = (timestamp: string) => {
  return new Date(timestamp)
    .toLocaleString('en-CA', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false,
    })
    .replace(',', '');
};

// Format value to max decimal places
const formatValue = (value: number) => {
  return value.toFixed(MAX_DECIMAL_PLACES).replace(/\.?0+$/, '');
};

function RouteComponent() {
  const params = useParams({ from: '/agents/$config-name/monitor' });
  const configName = params['config-name'];

  const {
    data: filteredMetricsData,
    isLoading: metricsLoading,
    error: metricsError,
  } = useFilteredMetricsQuery(configName);

  // Extract metric names and create a map for descriptions
  const filteredMetrics = filteredMetricsData?.map((m) => m.name) || [];
  const metricDescriptions = new Map(
    filteredMetricsData?.map((m) => [m.name, m.description]) || []
  );

  // Query data for each filtered metric
  const metricQueries = useMultipleAgentMetricsQueries(
    configName,
    filteredMetrics
  );

  if (metricsLoading) {
    return <div>Loading available metrics...</div>;
  }

  if (metricsError) {
    return <div>Error loading available metrics: {metricsError.message}</div>;
  }

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <div className="bg-white shadow-lg rounded-lg overflow-hidden">
        <div className="px-6 py-4 bg-gray-50 border-b border-gray-200">
          <h2 className="text-xl font-semibold text-gray-800">
            Available Metrics ({filteredMetrics.length})
          </h2>
        </div>

        {filteredMetrics.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Metric
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Description
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Timestamp
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Value
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {filteredMetrics.map((metric: string, index: number) => {
                  const queryResult = metricQueries[index];
                  const metricData = queryResult?.data?.[0]; // Assume exactly one timestamp/value tuple

                  return (
                    <tr key={metric} className="hover:bg-gray-50">
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                        {metric}
                      </td>
                      <td className="px-6 py-4 text-sm text-gray-500 max-w-xs">
                        {metricDescriptions.get(metric)}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {metricData ? (
                          formatTimestamp(metricData.timestamp)
                        ) : queryResult?.isLoading ? (
                          <span className="text-blue-500">Loading...</span>
                        ) : queryResult?.error ? (
                          <span className="text-red-500">Error</span>
                        ) : (
                          <span className="text-gray-400">-</span>
                        )}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {metricData ? (
                          formatValue(metricData.value)
                        ) : queryResult?.isLoading ? (
                          <span className="text-blue-500">Loading...</span>
                        ) : queryResult?.error ? (
                          <span className="text-red-500">Error</span>
                        ) : (
                          <span className="text-gray-400">-</span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="px-6 py-8 text-center text-gray-500">
            No filtered metrics available
          </div>
        )}
      </div>
    </div>
  );
}
