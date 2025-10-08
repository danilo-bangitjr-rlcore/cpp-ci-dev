import React, { useState } from 'react';
import { Link, useParams } from '@tanstack/react-router';
import DetailsCard from '../DetailsCard';
import {
  useAgentStatusQuery,
  useAgentToggleMutation,
  useIOToggleMutation,
} from '../../utils/useAgentQueries';

type ServiceStatus = {
  id?: string;
  state?: string;
  intended_state?: string;
  config_path?: string;
};

function getServiceStatus(statusObj: unknown): ServiceStatus {
  if (!statusObj) return {};
  if (Array.isArray(statusObj)) return statusObj[0] || {};
  return statusObj as ServiceStatus;
}

function getMetadata(service: ServiceStatus) {
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
  } = useAgentStatusQuery(configName, isPolling);

  const agentToggleMutation = useAgentToggleMutation(configName);
  const ioToggleMutation = useIOToggleMutation(configName);

  const corerl = getServiceStatus(agentStatusData?.service_statuses?.corerl);
  const coreio = getServiceStatus(agentStatusData?.service_statuses?.coreio);

  const getState = (service: ServiceStatus) =>
    service.state === 'running' || service.state === 'stopped'
      ? service.state
      : 'stopped';

  const agentState = getState(corerl);
  const ioState = getState(coreio);

  const isAgentLoading = isLoadingStatus || agentToggleMutation.isPending;
  const isIOLoading = isLoadingStatus || ioToggleMutation.isPending;

  const handleToggleAgentStatus = async () => {
    try {
      await agentToggleMutation.mutateAsync({
        configName,
        action: agentState === 'running' ? 'stop' : 'start',
      });
    } catch (error) {
      console.error('Failed to toggle agent status:', error);
    }
  };

  const handleToggleIOStatus = async () => {
    try {
      await ioToggleMutation.mutateAsync({
        configName,
        action: ioState === 'running' ? 'stop' : 'start',
      });
    } catch (error) {
      console.error('Failed to toggle I/O status:', error);
    }
  };

  const handleRefresh = () => {
    refetchStatus();
  };

  const agentName = agentStatusData?.id || configName;
  const ioName = coreio.id || 'I/O Service';

  if (isLoadingStatus && !agentStatusData) {
    return <div className="p-6">Loading agent details...</div>;
  }

  return (
    <div className="space-y-4">
      <div className="flex justify-center items-center mb-4">
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

      <div className="flex flex-row gap-6 flex-wrap justify-center p-10">
        <DetailsCard
          entityName={agentName}
          state={agentState}
          onToggleStatus={handleToggleAgentStatus}
          isLoading={isAgentLoading}
          metadata={getMetadata(corerl)}
          metadataTitle="Agent Metadata"
        />
        <DetailsCard
          entityName={ioName}
          state={ioState}
          onToggleStatus={handleToggleIOStatus}
          isLoading={isIOLoading}
          metadata={getMetadata(coreio)}
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
