import { useState } from 'react';
import AgentCard from './AgentCard';
import AddAgentCard from './AddAgentCard';
import { useDeleteAgentMutation } from '../../utils/useAgentMutations';
import {
  useConfigListQuery,
  useAgentNamesQueries,
} from '../../utils/useAgentQueries';

export interface Agent {
  agentName: string;
  configName: string;
  status: 'on' | 'off' | 'error';
}

const AgentsOverviewContainer: React.FC = () => {
  // Temporary mock state for agent statuses
  const [statusMap, setStatusMap] = useState<Record<string, Agent['status']>>(
    {}
  );

  const deleteAgentMutation = useDeleteAgentMutation();

  const {
    data: configNames = [],
    isLoading: isLoadingConfigList,
    error: configListError,
  } = useConfigListQuery();
  const agentNamesQueries = useAgentNamesQueries(configNames);

  const isLoading =
    isLoadingConfigList || agentNamesQueries.some((q) => q.isLoading);
  const hasError = !!configListError || agentNamesQueries.some((q) => q.error);

  const agents: Agent[] = configNames.map((name, idx) => {
    const query = agentNamesQueries[idx];
    const agentName = query?.data ?? name;
    return {
      agentName,
      configName: name,
      status: statusMap[name] ?? 'off',
    } as Agent;
  });

  const handleUpdateAgent = (updated: Agent) => {
    setStatusMap((prev) => ({ ...prev, [updated.configName]: updated.status }));
  };

  const handleDeleteAgent = (agent: Agent) => {
    if (window.confirm(`Are you sure you want to delete ${agent.agentName}?`)) {
      deleteAgentMutation.mutate(agent.configName);
    }
  };

  if (isLoading) return <div className="p-6">Loading agents...</div>;
  if (hasError)
    return <div className="p-6 text-red-600">Failed to load agents</div>;

  return (
    <div className="p-2">
      <h1 className="text-2xl font-bold mb-6">Agents Overview</h1>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {agents.map((agent) => (
          <AgentCard
            key={agent.configName}
            agent={agent}
            onAgentChange={handleUpdateAgent}
            onDelete={() => handleDeleteAgent(agent)}
          />
        ))}
        <AddAgentCard />
      </div>
    </div>
  );
};

export default AgentsOverviewContainer;
