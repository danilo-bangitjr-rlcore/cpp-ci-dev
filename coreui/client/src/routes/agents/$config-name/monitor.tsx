import { createFileRoute, useParams } from '@tanstack/react-router';
import {
  useAvailableMetricsQuery,
  useMultipleAgentMetricsQueries,
} from '../../../utils/useAgentMetricQueries';

export const Route = createFileRoute('/agents/$config-name/monitor')({
  component: RouteComponent,
});

// Constants
const MAX_DECIMAL_PLACES = 4;

// Filter patterns (case invariant) and their descriptions
const FILTER_CONFIGS = [
  { pattern: /^q$/i, description: 'Quality (value) of the action' },
  {
    pattern: /^q_ensemble_variance$/i,
    description: "The agent's uncertainty about the action it chose",
  },
  {
    pattern: /^observed_a_q[a-z0-9_]*$/i,
    description:
      "The difference between observed_a_q{label} and partial_return{label} can be interpreted as the agent's prediction accuracy",
  },
  {
    pattern: /^partial_return[a-z0-9_]*$/i,
    description:
      "The difference between observed_a_q{label} and partial_return{label} can be interpreted as the agent's prediction accuracy",
  },
  {
    pattern: /^pdf_plot_action_[a-z0-9_]+$/i,
    description: 'Probability of choosing each unrealized alternative',
  },
  {
    pattern: /^qs_plot_action_[a-z0-9_]+$/i,
    description: 'The value of unrealized alternatives',
  },
  {
    pattern: /^critic[0-9]*_loss$/i,
    description:
      'How well the agent understands the data seen so far. Less is better (think of it like error)',
  },
  {
    pattern: /^pipeline_[a-z0-9_]+_[a-z0-9_]+_[a-z0-9_]+_num_non_nan$/i,
    description:
      'They tell us statistics about the tags flowing into the agent',
  },
  {
    pattern: /^pipeline_[a-z0-9_]+_[a-z0-9_]+_[a-z0-9_]+_num_nan$/i,
    description:
      'They tell us statistics about the tags flowing into the agent',
  },
  {
    pattern: /^pipeline_[a-z0-9_]+_[a-z0-9_]+_[a-z0-9_]+_mean$/i,
    description:
      'They tell us statistics about the tags flowing into the agent',
  },
  {
    pattern: /^pipeline_[a-z0-9_]+_[a-z0-9_]+_[a-z0-9_]+_variance$/i,
    description:
      'They tell us statistics about the tags flowing into the agent',
  },
  {
    pattern: /^pipeline_[a-z0-9_]+_[a-z0-9_]+_[a-z0-9_]+_max$/i,
    description:
      'They tell us statistics about the tags flowing into the agent',
  },
  {
    pattern: /^pipeline_[a-z0-9_]+_[a-z0-9_]+_[a-z0-9_]+_min$/i,
    description:
      'They tell us statistics about the tags flowing into the agent',
  },
  {
    pattern: /^pipeline_[a-z0-9_]+_[a-z0-9_]+_[a-z0-9_]+_50th_percentile$/i,
    description:
      'They tell us statistics about the tags flowing into the agent',
  },
  {
    pattern: /^pipeline_[a-z0-9_]+_[a-z0-9_]+_[a-z0-9_]+_90th_percentile$/i,
    description:
      'They tell us statistics about the tags flowing into the agent',
  },
  {
    pattern: /^pipeline_[a-z0-9_]+_[a-z0-9_]+_[a-z0-9_]+_percent_nan$/i,
    description:
      'They tell us statistics about the tags flowing into the agent',
  },
  {
    pattern:
      /^pipeline_[a-z0-9_]+_[a-z0-9_]+_[a-z0-9_]+_average_length_of_nan_chunks$/i,
    description:
      'These could all be exposed for the INIT stage of the pipeline. They tell us statistics about the tags flowing into the agent',
  },
  {
    pattern:
      /^pipeline_[a-z0-9_]+_[a-z0-9_]+_[a-z0-9_]+_average_length_of_non_nan_chunks$/i,
    description:
      'These could all be exposed for the INIT stage of the pipeline. They tell us statistics about the tags flowing into the agent',
  },
  {
    pattern: /^ae-num_nan_obs$/i,
    description: 'Number of missing values in observations',
  },
  { pattern: /^ae-imputed$/i, description: 'Number of values imputed' },
  {
    pattern: /^ae-quality-[a-z0-9_]+_trace_[a-z0-9_]+$/i,
    description:
      'Quality of state variables input to agent. Can be made more interpretable by averaging across decays. This gives a quality per tag',
  },
] as const;

// Helper to get description for a metric
const getMetricDescription = (metric: string): string | undefined => {
  const config = FILTER_CONFIGS.find((config) => config.pattern.test(metric));
  return config?.description;
};

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
    data: availableMetrics,
    isLoading: metricsLoading,
    error: metricsError,
  } = useAvailableMetricsQuery(configName);

  const filteredMetrics =
    availableMetrics?.filter((metric: string) =>
      FILTER_CONFIGS.some((config) => config.pattern.test(metric))
    ) || [];

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
                        {getMetricDescription(metric)}
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
