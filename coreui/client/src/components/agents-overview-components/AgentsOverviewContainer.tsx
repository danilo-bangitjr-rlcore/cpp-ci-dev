import React, { useState, useEffect } from 'react';
import { useQuery, useQueries } from '@tanstack/react-query';
import AgentCard from './AgentCard';
import AddAgentCard from './AddAgentCard';
import { API_ENDPOINTS, get } from '../../utils/api';

export interface Agent {
  name: string;
  status: 'on' | 'off' | 'error';
}

// Config API functions
const fetchConfigList = async (): Promise<string[]> => {
  const response = await get(API_ENDPOINTS.configs.list_raw);
  if (!response.ok) {
    throw new Error('Failed to fetch config list');
  }
  const data: { configs: string[] } = await response.json();
  return data.configs;
};

const fetchRawConfig = async (configName: string): Promise<Record<string, any>> => {
  const response = await get(API_ENDPOINTS.configs.raw(configName));
  if (!response.ok) {
    throw new Error(`Failed to fetch raw config for ${configName}`);
  }
  const data: { config: Record<string, any> } = await response.json();
  return data.config;
};

const AgentsOverviewContainer: React.FC = () => {
  const [agents, setAgents] = useState<Agent[]>([]);

  // Fetch config list using React Query
  const { data: configNames, isLoading: isLoadingConfigList, error: configListError } = useQuery({
    queryKey: ['configList'],
    queryFn: fetchConfigList,
  });

  // Fetch all raw configs using useQueries
  const configQueries = useQueries({
    queries: (configNames || []).map((configName) => ({
      queryKey: ['rawConfig', configName],
      queryFn: () => fetchRawConfig(configName),
      enabled: !!configNames,
    })),
  });

  // Transform configs to agents when all queries are successful
  const allConfigsLoaded = configQueries.length > 0 && configQueries.every(query => query.isSuccess);
  const fetchedAgents = allConfigsLoaded
    ? configQueries.map((query, index) => ({
        name: query.data?.agent_name || configNames![index],
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
    const newAgent: Agent = { name: '', status: 'off' };
    setAgents([...agents, newAgent]);
  };

  const handleUpdateAgent = (index: number, updatedAgent: Agent) => {
    const newAgents = [...agents];
    newAgents[index] = updatedAgent;
    setAgents(newAgents);
  };

  const handleDeleteAgent = (index: number) => {
    const newAgents = agents.filter((_, i) => i !== index);
    setAgents(newAgents);
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
