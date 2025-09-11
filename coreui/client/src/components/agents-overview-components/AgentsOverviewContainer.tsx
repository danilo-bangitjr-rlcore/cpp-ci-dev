import React, { useState, useEffect } from 'react';
import AgentCard from './AgentCard';
import AddAgentCard from './AddAgentCard';
import { useDeleteAgentMutation } from './useAgentMutations';
import { useConfigListQuery, useRawConfigsQueries } from './useAgentQueries';

export interface Agent {
  agentName: string;
  configName: string;
  status: 'on' | 'off' | 'error';
}

const AgentsOverviewContainer: React.FC = () => {
  const [agents, setAgents] = useState<Agent[]>([]);

  // Delete mutation
  const deleteAgentMutation = useDeleteAgentMutation();

  // Fetch config list and raw configs
  const { data: configNames, isLoading: isLoadingConfigList, error: configListError } = useConfigListQuery();
  const configQueries = useRawConfigsQueries(configNames);

  // Transform configs to agents when all queries are successful
  const allConfigsLoaded = configQueries.length > 0 && configQueries.every(query => query.isSuccess);
  const fetchedAgents = allConfigsLoaded
    ? configQueries.map((query, index) => ({
        agentName: query.data?.agent_name || configNames![index],
        configName: configNames![index],
        status: 'off' as const,
      }))
    : [];

  // Update agents state when fetched agents change (only for initial load)
  useEffect(() => {
    if (allConfigsLoaded && fetchedAgents.length > 0 && agents.length === 0) {
      setAgents(fetchedAgents);
    }
  }, [allConfigsLoaded, fetchedAgents, agents.length]);

  const handleAddAgent = () => {
    const newAgent: Agent = { agentName: '', configName: '', status: 'off' };
    setAgents([...agents, newAgent]);
  };

  const handleUpdateAgent = (index: number, updatedAgent: Agent) => {
    const newAgents = [...agents];
    newAgents[index] = updatedAgent;
    setAgents(newAgents);
  };

  const handleDeleteAgent = (index: number) => {
    const agentToDelete = agents[index];
    if (window.confirm(`Are you sure you want to delete ${agentToDelete.agentName}?`)) {
      // Call the delete mutation with configName
      deleteAgentMutation.mutate(agentToDelete.configName, {
        onSuccess: () => {
          // Remove from local state only after successful deletion
          const newAgents = agents.filter((_, i) => i !== index);
          setAgents(newAgents);
        },
        onError: (error) => {
          console.error('Failed to delete agent:', error);
          alert(`Failed to delete agent ${agentToDelete.agentName}. Please try again.`);
        },
      });
    }
  };

  const isLoading = isLoadingConfigList || (configNames && configQueries.some(query => query.isLoading));

  // Check for errors
  const hasError = configListError || configQueries.some(query => query.error);

  if (isLoading) {
    return <div className="p-6">Loading agents...</div>;
  }

  if (hasError) {
    return <div className="p-6 text-red-600">Failed to load agents</div>;
  }

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-6">Agents Overview</h1>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {agents.map((agent: Agent, index: number) => (
          <AgentCard
            key={index}
            agent={agent}
            onAgentChange={(updatedAgent) =>
              handleUpdateAgent(index, updatedAgent)
            }
            onDelete={() => handleDeleteAgent(index)}
          />
        ))}
        <AddAgentCard onAdd={handleAddAgent} />
      </div>
    </div>
  );
};

export default AgentsOverviewContainer;
