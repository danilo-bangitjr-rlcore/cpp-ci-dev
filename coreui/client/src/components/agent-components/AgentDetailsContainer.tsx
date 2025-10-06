import React, { useState } from 'react';
import { Link, useParams } from '@tanstack/react-router';
import DetailsCard from '../DetailsCard';
import {
  useAgentStatusQuery,
  type AgentStatusResponse,
} from '../../utils/useAgentQueries';
import {
  useAgentToggleMutation,
  useIOToggleMutation,
} from '../../utils/useAgentMutations';

type ServiceStatus = {
  id?: string;
  state?: string;
  intended_state?: string;
  config_path?: string;
};

type AgentData = {
  agentId: string;
  configPath: string;
  corerl: ServiceStatus;
  coreio: ServiceStatus;
};

function extractServiceStatus(statusObj: unknown): ServiceStatus {
  if (!statusObj) return {};
  if (Array.isArray(statusObj)) return statusObj[0] || {};
  return statusObj as ServiceStatus;
}

function extractAgentData(agentStatusData: AgentStatusResponse | undefined): AgentData | null {
  if (!agentStatusData) return null;

  return {
    agentId: agentStatusData.id || '',
    configPath: agentStatusData.config_path || '',
    corerl: extractServiceStatus(agentStatusData.service_statuses?.corerl),
    coreio: extractServiceStatus(agentStatusData.service_statuses?.coreio),
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

  const agentData = extractAgentData(agentStatusData);

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

      {(agentToggleMutation.error || ioToggleMutation.error) && (
        <div className="mx-auto max-w-md p-4 bg-red-50 border border-red-200 rounded-lg">
          <h4 className="text-sm font-medium text-red-800 mb-2">Recent Errors:</h4>
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
    </div>
  );
};

export default AgentDetailsContainer;
