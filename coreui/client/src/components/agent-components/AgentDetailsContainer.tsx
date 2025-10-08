import { useState } from 'react';
import { useParams } from '@tanstack/react-router';
import {
  useAgentStatusQuery,
  useAgentsMissingConfigQuery,
  useConfigPathQuery,
  useIOListQuery,
} from '../../utils/useAgentQueries';
import {
  useAgentToggleMutation,
  useIOToggleMutation,
} from '../../utils/useAgentMutations';
import type {
  AgentStatusResponse,
  ServiceStatus,
} from '../../types/agent-types';
import AgentStatusMessages from './AgentStatusMessages';
import ServiceCardsContainer from './ServiceCardsContainer';
import ConfigurationMenu from './ConfigurationMenu';

type AgentData = {
  agentId: string;
  configPath: string;
  corerl: ServiceStatus;
  coreio: ServiceStatus;
  isNeverStarted: boolean;
  isConfigMissing: boolean;
};

function extractAgentData(
  agentStatusData: AgentStatusResponse | undefined,
  isConfigMissing: boolean
): AgentData | null {
  if (!agentStatusData) return null;

  const hasNoConfigPath =
    agentStatusData.config_path === null || agentStatusData.config_path === '';
  const hasNoServices =
    !agentStatusData.service_statuses?.corerl &&
    !agentStatusData.service_statuses?.coreio;
  const isNeverStarted = hasNoConfigPath && hasNoServices;

  const emptySvc: ServiceStatus = {
    id: '',
    state: '',
    intended_state: '',
    config_path: null,
  };

  return {
    agentId: agentStatusData.id || '',
    configPath: agentStatusData.config_path || '',
    corerl: agentStatusData.service_statuses?.corerl || emptySvc,
    coreio: agentStatusData.service_statuses?.coreio || emptySvc,
    isNeverStarted,
    isConfigMissing,
  };
}

export default function AgentDetailsContainer() {
  const params = useParams({ from: '/agents/$config-name/' });
  const configName = params['config-name'];
  const [isPolling, setIsPolling] = useState(true);
  const [selectedExistingIO, setSelectedExistingIO] = useState<string>('');

  const {
    data: agentStatusData,
    isLoading: isLoadingStatus,
    refetch: refetchStatus,
    error: statusError,
  } = useAgentStatusQuery(configName, isPolling);
  const { data: missingConfigAgents = [] } =
    useAgentsMissingConfigQuery(isPolling);

  const isConfigMissing = missingConfigAgents.includes(configName);
  const agentData = extractAgentData(agentStatusData, isConfigMissing);

  const { data: resolvedConfigPath, isLoading: isLoadingConfigPath } =
    useConfigPathQuery(configName, agentData?.isNeverStarted);
  const { data: ioListResponse, isLoading: isLoadingIOs } = useIOListQuery(
    agentData?.isNeverStarted ?? false
  );

  const agentToggleMutation = useAgentToggleMutation(
    resolvedConfigPath || agentData?.configPath || configName,
    agentData?.agentId || '',
    selectedExistingIO || undefined
  );

  const ioToggleMutation = useIOToggleMutation(
    resolvedConfigPath || agentData?.configPath || configName,
    agentData?.coreio?.id || ''
  );

  const handleToggleAgent = async () => {
    if (!agentData) return;
    const action = agentData.corerl.state === 'running' ? 'stop' : 'start';
    await agentToggleMutation.mutateAsync({ action });
    setSelectedExistingIO('');
    await refetchStatus();
  };

  const handleToggleIO = async () => {
    if (!agentData) return;
    const action = agentData.coreio.state === 'running' ? 'stop' : 'start';
    await ioToggleMutation.mutateAsync({ action });
    await refetchStatus();
  };

  if (isLoadingStatus && !agentData) {
    return <div className="p-6">Loading agent details...</div>;
  }

  if (statusError) {
    return (
      <div className="p-6 text-red-600">
        Error loading agent details: {statusError.message}
        <button
          onClick={() => refetchStatus()}
          className="ml-4 px-3 py-1 text-sm bg-red-600 text-white rounded hover:bg-red-700 transition-colors"
        >
          Retry
        </button>
      </div>
    );
  }

  if (!agentData) {
    return <div className="p-6">No agent data available</div>;
  }

  return (
    <div className="space-y-4">
      <div className="flex justify-center items-center mb-4">
        <div className="flex items-center space-x-2">
          <button
            onClick={() => refetchStatus()}
            disabled={isLoadingStatus}
            className="px-3 py-1 text-sm bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isLoadingStatus ? 'Refreshing...' : 'Refresh Status'}
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

      <AgentStatusMessages
        isConfigMissing={agentData.isConfigMissing}
        isNeverStarted={agentData.isNeverStarted}
        resolvedConfigPath={resolvedConfigPath}
        isLoadingConfigPath={isLoadingConfigPath}
        availableIOs={ioListResponse?.coreio_services ?? []}
        isLoadingIOs={isLoadingIOs}
        selectedExistingIO={selectedExistingIO}
        onSelectIO={setSelectedExistingIO}
        agentError={agentToggleMutation.error}
        ioError={ioToggleMutation.error}
      />

      <ServiceCardsContainer
        agentData={agentData}
        resolvedConfigPath={resolvedConfigPath}
        selectedExistingIO={selectedExistingIO}
        onToggleAgent={handleToggleAgent}
        onToggleIO={handleToggleIO}
        isTogglingAgent={agentToggleMutation.isPending}
        isTogglingIO={ioToggleMutation.isPending}
        isLoadingConfigPath={isLoadingConfigPath}
      />

      <ConfigurationMenu
        configName={configName}
        isConfigMissing={agentData.isConfigMissing}
      />
    </div>
  );
}
