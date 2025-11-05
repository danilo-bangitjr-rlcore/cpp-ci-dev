import { useState } from 'react';
import AgentCard from './AgentCard';
import AddAgentCard from './AddAgentCard';
import ConfigErrorCard from './ConfigErrorCard';
import {
  useConfigListQuery,
  useAgentNamesQueries,
  useAgentStatusQueries,
  useAgentsMissingConfigQuery,
} from '../../utils/useAgentQueries';
import type { Agent, AgentStatus } from '../../types/agent-types';

const AgentsOverviewContainer: React.FC = () => {
  const [isPolling, setIsPolling] = useState(true);

  const {
    data: configNames = [],
    isLoading: isLoadingConfigList,
    error: configListError,
  } = useConfigListQuery();

  const {
    data: missingConfigAgents = [],
    isLoading: isLoadingMissingConfig,
    error: missingConfigError,
  } = useAgentsMissingConfigQuery(isPolling);

  const agentNamesQueries = useAgentNamesQueries(configNames);

  // Separate valid configs from error configs
  const validConfigIndices: number[] = [];
  const validConfigNames: string[] = [];
  const errorConfigs: { configName: string; error: string }[] = [];

  configNames.forEach((name, idx) => {
    const nameQuery = agentNamesQueries[idx];
    if (nameQuery?.error) {
      errorConfigs.push({
        configName: name,
        error: nameQuery.error.message,
      });
    } else if (!nameQuery?.isLoading) {
      validConfigIndices.push(idx);
      validConfigNames.push(name);
    }
  });

  const agentStatusQueries = useAgentStatusQueries(validConfigNames, isPolling);

  const isNeverStarted = (statusData: any): boolean => {
    if (!statusData) return false;

    const hasNoConfigPath =
      statusData.config_path === null || statusData.config_path === '';
    const hasNoServices =
      !statusData.service_statuses ||
      (!statusData.service_statuses.corerl &&
        !statusData.service_statuses.coreio);

    return hasNoConfigPath && hasNoServices;
  };

  const mapApiStatusToAgentStatus = (
    apiStatus: string | undefined,
    hasStatusData: boolean,
    statusData: any
  ): AgentStatus => {
    if (hasStatusData && isNeverStarted(statusData)) {
      return 'never-started';
    }

    if (!hasStatusData || !apiStatus) {
      return 'off';
    }

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
    isLoadingMissingConfig ||
    (agentNamesQueries.some((q) => q.isLoading) && errorConfigs.length === 0) ||
    agentStatusQueries.some((q) => q.isLoading);

  const hasError =
    !!configListError ||
    !!missingConfigError ||
    agentStatusQueries.some((q) => q.error);

  const agents: Agent[] = validConfigIndices.map((originalIdx, validIdx) => {
    const nameQuery = agentNamesQueries[originalIdx];
    const statusQuery = agentStatusQueries[validIdx];
    const configName = configNames[originalIdx];
    const agentName = nameQuery?.data ?? configName;
    const statusData = statusQuery?.data;
    const hasStatusData = !!statusData;
    const status = mapApiStatusToAgentStatus(
      statusData?.state,
      hasStatusData,
      statusData
    );

    return {
      agentName,
      configName,
      status,
    };
  });

  const missingConfigAgentsList: Agent[] = missingConfigAgents.map((name) => ({
    agentName: name,
    configName: name,
    status: 'config-missing',
  }));

  const allAgents = [...agents, ...missingConfigAgentsList];

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

      {missingConfigAgentsList.length > 0 && (
        <div className="mb-4 p-4 bg-orange-50 border border-orange-200 rounded-lg">
          <div className="flex items-center space-x-2">
            <svg
              className="w-5 h-5 text-orange-600"
              fill="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                fillRule="evenodd"
                d="M8.485 2.495c.673-1.167 2.357-1.167 3.03 0l6.28 10.875c.673 1.167-.17 2.625-1.516 2.625H3.72c-1.347 0-2.189-1.458-1.515-2.625L8.485 2.495zM12 9a.75.75 0 01.75.75v3.5a.75.75 0 01-1.5 0v-3.5A.75.75 0 0112 9zm0 8a1 1 0 100-2 1 1 0 000 2z"
                clipRule="evenodd"
              />
            </svg>
            <p className="text-sm font-medium text-orange-800">
              {missingConfigAgentsList.length} agent
              {missingConfigAgentsList.length > 1 ? 's' : ''} running without a
              configuration file. These agents may have been started manually or
              their config files were deleted.
            </p>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {errorConfigs.map((errorConfig) => (
          <ConfigErrorCard
            key={errorConfig.configName}
            configName={errorConfig.configName}
            errorMessage={errorConfig.error}
          />
        ))}
        {allAgents.map((agent) => (
          <AgentCard key={agent.configName} agent={agent} />
        ))}
        <AddAgentCard />
      </div>
    </div>
  );
};

export default AgentsOverviewContainer;
