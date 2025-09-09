import { createFileRoute } from '@tanstack/react-router';
import AgentCard from '../components/agents-overview-components/AgentCard';
import AddAgentCard from '../components/agents-overview-components/AddAgentCard';
import { useState } from 'react';

export interface Agent {
  name: string;
  status: 'on' | 'off' | 'error';
  configPath: string;
}

export const Route = createFileRoute('/agents-overview')({
  component: RouteComponent,
});

function RouteComponent() {
  const [agents, setAgents] = useState<Agent[]>([
    { name: 'Agent 1', status: 'on', configPath: '/path/to/config1.yaml' },
    { name: 'Agent 2', status: 'off', configPath: '/path/to/config2.yaml' },
  ]);

  const handleAddAgent = () => {
    const newAgent: Agent = { name: '', status: 'off', configPath: '' };
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

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-6">Agents Overview</h1>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {agents.map((agent, index) => (
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
}
