import React, { useState, useEffect } from 'react';
import { type Agent } from './AgentsOverviewContainer';

interface AgentCardProps {
  agent: Agent;
  onAgentChange: (agent: Agent) => void;
  onDelete: () => void;
}

const AgentCard: React.FC<AgentCardProps> = ({
  agent,
  onAgentChange,
  onDelete,
}) => {
  const [localAgent, setLocalAgent] = useState<Agent>(agent);

  useEffect(() => {
    setLocalAgent(agent);
  }, [agent]);

  const handleToggleStatus = () => {
    const newStatus: 'on' | 'off' | 'error' =
      localAgent.status === 'on' ? 'off' : 'on';
    const updatedAgent = { ...localAgent, status: newStatus };
    setLocalAgent(updatedAgent);
    onAgentChange(updatedAgent);
  };

  const handleDelete = () => {
    if (window.confirm(`Are you sure you want to delete ${localAgent.name}?`)) {
      onDelete();
    }
  };

  return (
    <div className="border border-gray-300 bg-gray-100 p-4 rounded-lg shadow-sm min-h-[200px] flex flex-col justify-between">
      <div className="space-y-2">
        <div>
          <div
            className="text-xl font-bold bg-transparent border border-transparent p-1 rounded w-full"
          >
            {localAgent.name}
          </div>
        </div>
        <div className="flex items-center justify-between my-4">
          <div className="flex items-center space-x-2">
            <label className="text-sm font-medium text-gray-700">Status:</label>
            <span
              className={`px-3 py-1 rounded text-sm ${
                localAgent.status === 'on'
                  ? 'bg-green-100 text-green-800'
                  : localAgent.status === 'off'
                    ? 'bg-red-100 text-red-800'
                    : 'bg-yellow-100 text-yellow-800'
              }`}
            >
              {localAgent.status.toUpperCase()}
            </span>
          </div>
          <button
            onClick={handleToggleStatus}
            className={`text-white px-3 py-1 rounded text-sm ${localAgent.status === 'on' ? 'bg-gray-500 hover:bg-gray-600' : 'bg-blue-500 hover:bg-blue-600'}`}
          >
            {localAgent.status === 'on' ? 'Turn Off' : 'Turn On'}
          </button>
        </div>
        <button className="w-full bg-gray-500 text-white py-1 rounded hover:bg-gray-600">
          Configure
        </button>
        <button className="w-full bg-gray-500 text-white py-1 rounded hover:bg-gray-600">
          Monitor
        </button>
      </div>
      <div className="mt-4 flex justify-end">
        <button
          onClick={handleDelete}
          className="bg-red-100 text-red-800 px-3 py-1 rounded text-sm hover:bg-red-600 hover:text-white"
        >
          Delete Agent
        </button>
      </div>
    </div>
  );
};

export default AgentCard;
