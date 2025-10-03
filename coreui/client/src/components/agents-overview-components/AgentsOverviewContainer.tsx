import { useState } from 'react';
import AgentCard from './AgentCard';
import AddAgentCard from './AddAgentCard';
import {
  useConfigListQuery,
  useAgentNamesQueries,
  useAgentStatusQueries,
} from '../../utils/useAgentQueries';

export interface Agent {
  agentName: string;
  configName: string;
  status: 'on' | 'off' | 'error';
}

const AgentsOverviewContainer: React.FC = () => {
  const [isPolling, setIsPolling] = useState(true);

  const {
    data: configNames = [],
    isLoading: isLoadingConfigList,
    error: configListError,
  } = useConfigListQuery();

  const agentNamesQueries = useAgentNamesQueries(configNames);

  const agentStatusQueries = useAgentStatusQueries(configNames, isPolling);

  // Convert API status to our status format
  const mapApiStatusToAgentStatus = (apiStatus: string): Agent['status'] => {
    switch (apiStatus) {
      case 'running':
        return 'on';
      case 'stopped':
        return 'off';
      case 'error':
        return 'error';
      default:
        return 'off';
    }
  };

  const isLoading =
    isLoadingConfigList ||
    agentNamesQueries.some((q) => q.isLoading) ||
    agentStatusQueries.some((q) => q.isLoading);

  const hasError =
    !!configListError ||
    agentNamesQueries.some((q) => q.error) ||
    agentStatusQueries.some((q) => q.error);

  const agents: Agent[] = configNames.map((name, idx) => {
    const nameQuery = agentNamesQueries[idx];
    const statusQuery = agentStatusQueries[idx];
    const agentName = nameQuery?.data ?? name;
    const statusData = statusQuery?.data;
    const status = statusData
      ? mapApiStatusToAgentStatus(statusData.state)
      : 'off';

    return {
      agentName,
      configName: name,
      status,
    };
  });

  // Manual refresh for all agent status queries
  const handleRefresh = () => {
    agentStatusQueries.forEach((query) => {
      if (query.refetch) query.refetch();
    });
  };

  if (isLoading) return <div className="p-6">Loading agents...</div>;
  if (hasError)
    return <div className="p-6 text-red-600">Failed to load agents</div>;

  return (
    <div className="p-2">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold">Agents Overview</h1>
        <div className="flex items-center space-x-2">
          <button
            onClick={handleRefresh}
            className="px-3 py-1 text-sm bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors"
          >
            Refresh Status
          </button>
          <button
            onClick={() => setIsPolling(!isPolling)}
            className={`px-3 py-1 text-sm rounded transition-colors ${
              isPolling
                ? 'bg-green-600 text-white hover:bg-green-700'
                : 'bg-gray-600 text-white hover:bg-gray-700'
            }`}
          >
            {isPolling ? 'Auto-refresh ON' : 'Auto-refresh OFF'}
          </button>
        </div>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {agents.map((agent) => (
          <AgentCard key={agent.configName} agent={agent} />
        ))}
        <AddAgentCard />
      </div>
    </div>
  );
};

export default AgentsOverviewContainer;
