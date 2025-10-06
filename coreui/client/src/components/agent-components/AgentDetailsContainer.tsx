import React, { useState } from 'react';
import { Link, useParams } from '@tanstack/react-router';
import DetailsCard from '../DetailsCard';
import {
  useAgentStatusQuery,
  useAgentToggleMutation,
  useIOStatusQuery,
  useIOToggleMutation,
} from '../../utils/useAgentQueries';

const AgentDetailsContainer: React.FC = () => {
  const params = useParams({ from: '/agents/$config-name/' });
  const configName = params['config-name'];

  const [agentData] = useState({
    agentName: configName,
    version: '1.0.0',
    uptime: '5 hours',
  });

  // Agent status and controls
  const { data: agentStatusData, isLoading: isLoadingAgentStatus } =
    useAgentStatusQuery(configName);

  const agentToggleMutation = useAgentToggleMutation(configName);

  const agentState =
    agentStatusData?.state === 'running' || agentStatusData?.state === 'stopped'
      ? agentStatusData.state
      : 'stopped';
  const isAgentLoading = isLoadingAgentStatus || agentToggleMutation.isPending;

  // I/O status and controls
  const { data: ioStatusData, isLoading: isLoadingIOStatus } =
    useIOStatusQuery(configName);

  const ioToggleMutation = useIOToggleMutation(configName);

  const ioState =
    ioStatusData?.state === 'running' || ioStatusData?.state === 'stopped'
      ? ioStatusData.state
      : 'stopped';
  const isIOLoading = isLoadingIOStatus || ioToggleMutation.isPending;

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

  const agentMetadata = [
    { label: 'Version', value: agentData.version },
    { label: 'Uptime', value: agentData.uptime },
  ];

  const ioMetadata = [
    { label: 'Time since last ingress', value: '23s' },
    { label: 'Time since last action', value: '1h 12m 43s' },
    { label: 'Heartbeat', value: '9s' },
  ];

  return (
    <div className="space-y-4">
      <div className="flex flex-row gap-6 flex-wrap justify-center p-10">
        <DetailsCard
          entityName={agentData.agentName}
          state={agentState}
          onToggleStatus={handleToggleAgentStatus}
          isLoading={isAgentLoading}
          metadata={agentMetadata}
          metadataTitle="Agent Metadata"
        />
        <DetailsCard
          entityName="I/O Service"
          state={ioState}
          onToggleStatus={handleToggleIOStatus}
          isLoading={isIOLoading}
          metadata={ioMetadata}
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
