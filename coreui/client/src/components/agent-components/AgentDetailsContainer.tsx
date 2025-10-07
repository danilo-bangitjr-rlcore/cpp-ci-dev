import React, { useState } from 'react';
import { Link, useParams } from '@tanstack/react-router';
import DetailsCard from '../DetailsCard';
import {
  useAgentStatusQuery,
  useAgentsMissingConfigQuery,
} from '../../utils/useAgentQueries';
import {
  useAgentToggleMutation,
  useIOToggleMutation,
} from '../../utils/useAgentMutations';
import type {
  AgentStatusResponse,
  ServiceStatus,
} from '../../types/agent-types';

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

function getServiceState(service: ServiceStatus): 'running' | 'stopped' {
  return service.state === 'running' ? 'running' : 'stopped';
}

function getServiceMetadata(service: ServiceStatus) {
  return [
    { label: 'Service ID', value: service.id || 'N/A' },
    { label: 'Intended State', value: service.intended_state || 'N/A' },
    {
      label: 'Config Path',
      value: service.config_path
        ? '.../' + service.config_path.split('/').slice(-2).join('/')
        : 'N/A',
    },
  ];
}

const AgentDetailsContainer: React.FC = () => {
  const params = useParams({ from: '/agents/$config-name/' });
  const configName = params['config-name'];
  const [isPolling, setIsPolling] = useState(true);

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

  const agentToggleMutation = useAgentToggleMutation(
    agentData?.configPath || configName,
    agentData?.agentId || ''
  );

  const ioToggleMutation = useIOToggleMutation(
    agentData?.configPath || configName,
    agentData?.coreio?.id || ''
  );

  const handleToggleAgent = async () => {
    if (!agentData) return;

    const currentState = getServiceState(agentData.corerl);
    const action = currentState === 'running' ? 'stop' : 'start';

    try {
      await agentToggleMutation.mutateAsync({ action });
      refetchStatus();
    } catch (error) {
      console.error('Failed to toggle agent status:', error);
    }
  };

  const handleToggleIO = async () => {
    if (!agentData) return;

    const currentState = getServiceState(agentData.coreio);
    const action = currentState === 'running' ? 'stop' : 'start';

    try {
      await ioToggleMutation.mutateAsync({ action });
      refetchStatus();
    } catch (error) {
      console.error('Failed to toggle I/O status:', error);
    }
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

  const agentName = agentData.agentId || configName;
  const ioName = agentData.coreio.id || 'I/O Service';

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

      {agentData.isConfigMissing && (
        <div className="mx-auto max-w-md p-4 bg-orange-50 border border-orange-200 rounded-lg">
          <div className="flex items-start space-x-2">
            <svg
              className="w-5 h-5 text-orange-600 flex-shrink-0 mt-0.5"
              fill="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                fillRule="evenodd"
                d="M8.485 2.495c.673-1.167 2.357-1.167 3.03 0l6.28 10.875c.673 1.167-.17 2.625-1.516 2.625H3.72c-1.347 0-2.189-1.458-1.515-2.625L8.485 2.495zM12 9a.75.75 0 01.75.75v3.5a.75.75 0 01-1.5 0v-3.5A.75.75 0 0112 9zm0 8a1 1 0 100-2 1 1 0 000 2z"
                clipRule="evenodd"
              />
            </svg>
            <div>
              <p className="text-sm font-medium text-orange-800">
                Configuration file missing for this agent
              </p>
              <p className="text-xs text-orange-700 mt-1">
                This agent is running without an associated config file. It may
                have been started manually or the config was deleted. You can
                still manage the running services, but configuration options are
                not available.
              </p>
            </div>
          </div>
        </div>
      )}

      {agentData.isNeverStarted && (
        <div className="mx-auto max-w-md p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
          <div className="flex items-center space-x-2">
            <svg
              className="w-5 h-5 text-yellow-600"
              fill="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                fillRule="evenodd"
                d="M8.485 2.495c.673-1.167 2.357-1.167 3.03 0l6.28 10.875c.673 1.167-.17 2.625-1.516 2.625H3.72c-1.347 0-2.189-1.458-1.515-2.625L8.485 2.495zM12 9a.75.75 0 01.75.75v3.5a.75.75 0 01-1.5 0v-3.5A.75.75 0 0112 9zm0 8a1 1 0 100-2 1 1 0 000 2z"
                clipRule="evenodd"
              />
            </svg>
            <p className="text-sm font-medium text-yellow-800">
              This agent has never been started. Service information is not
              available.
            </p>
          </div>
        </div>
      )}

      {(agentToggleMutation.error || ioToggleMutation.error) && (
        <div className="mx-auto max-w-md p-4 bg-red-50 border border-red-200 rounded-lg">
          <h4 className="text-sm font-medium text-red-800 mb-2">
            Recent Errors:
          </h4>
          {agentToggleMutation.error && (
            <p className="text-sm text-red-700 mb-1">
              Agent: {agentToggleMutation.error.message}
            </p>
          )}
          {ioToggleMutation.error && (
            <p className="text-sm text-red-700">
              I/O: {ioToggleMutation.error.message}
            </p>
          )}
        </div>
      )}

      <div className="flex flex-row gap-6 flex-wrap justify-center p-10">
        <DetailsCard
          entityName={agentName}
          state={getServiceState(agentData.corerl)}
          onToggleStatus={handleToggleAgent}
          isLoading={agentToggleMutation.isPending}
          metadata={getServiceMetadata(agentData.corerl)}
          metadataTitle="Agent Metadata"
        />
        <DetailsCard
          entityName={ioName}
          state={getServiceState(agentData.coreio)}
          onToggleStatus={handleToggleIO}
          isLoading={ioToggleMutation.isPending}
          metadata={getServiceMetadata(agentData.coreio)}
          metadataTitle="I/O Metadata"
        />
      </div>

      {!agentData.isConfigMissing && (
        <div className="max-w-md mx-auto justify-center">
          <div className="bg-white p-4 rounded-lg border border-gray-200 shadow-sm">
            <h3 className="text-lg font-semibold text-gray-900 mb-3">
              Configuration Options
            </h3>
            <ul className="space-y-2">
              <li>
                <Link
                  to={'/agents/$config-name/general-settings'}
                  params={{ 'config-name': configName }}
                  className="block p-2 text-blue-600 hover:text-blue-800 hover:bg-blue-50 rounded transition-colors"
                >
                  General Settings
                </Link>
              </li>
              <li>
                <Link
                  to={'/agents/$config-name/monitor'}
                  params={{ 'config-name': configName }}
                  className="block p-2 text-blue-600 hover:text-blue-800 hover:bg-blue-50 rounded transition-colors"
                >
                  Monitor
                </Link>
              </li>
            </ul>
          </div>
        </div>
      )}

      {agentData.isConfigMissing && (
        <div className="max-w-md mx-auto justify-center">
          <div className="bg-white p-4 rounded-lg border border-gray-200 shadow-sm">
            <h3 className="text-lg font-semibold text-gray-900 mb-3">
              Configuration Unavailable
            </h3>
            <p className="text-sm text-gray-600">
              Create a configuration file for this agent to access configuration
              options.
            </p>
          </div>
        </div>
      )}
    </div>
  );
};

export default AgentDetailsContainer;
