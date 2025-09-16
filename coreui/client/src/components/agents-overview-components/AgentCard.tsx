import React from 'react';
import { Link } from '@tanstack/react-router';
import { TrashIcon } from '../icons';
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
  const handleToggleStatus = () => {
    const newStatus: 'on' | 'off' | 'error' =
      agent.status === 'on' ? 'off' : 'on';
    const updatedAgent = { ...agent, status: newStatus };
    onAgentChange(updatedAgent);
  };

  const handleDelete = () => {
    onDelete();
  };

  return (
    <div className="border border-gray-300 bg-gray-100 p-4 rounded-lg shadow-sm min-h-[200px] flex flex-col justify-between">
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <div className="text-xl font-bold bg-transparent border border-transparent p-1 rounded">
            {agent.agentName}
          </div>
          <button
            onClick={handleDelete}
            className="text-gray-500 hover:text-red-700 hover:bg-red-50 p-1 rounded transition-colors"
            title="Delete Agent"
          >
            <TrashIcon className="w-4 h-4" />
          </button>
        </div>
        <div className="flex items-center justify-between my-4">
          <div className="flex items-center space-x-2">
            <label className="text-sm font-medium text-gray-700">Status:</label>
            <span
              className={`px-3 py-1 rounded text-sm ${
                agent.status === 'on'
                  ? 'bg-green-100 text-green-800'
                  : agent.status === 'off'
                    ? 'bg-gray-200 text-red-800'
                    : 'bg-yellow-100 text-yellow-800'
              }`}
            >
              {agent.status.toUpperCase()}
            </span>
          </div>
          <button
            onClick={handleToggleStatus}
            className={`text-white px-3 py-1 rounded text-sm ${agent.status === 'on' ? 'bg-gray-500 hover:bg-gray-600' : 'bg-blue-500 hover:bg-blue-600'}`}
          >
            {agent.status === 'on' ? 'Turn Off' : 'Turn On'}
          </button>
        </div>
        <div className="flex justify-center">
          <Link
            to="/agents/$config-name"
            params={{ 'config-name': agent.configName }}
            className="bg-gray-500 text-white px-16 py-1 mt-4 rounded hover:bg-gray-600"
          >
            Details
          </Link>
        </div>
      </div>
    </div>
  );
};

export default AgentCard;
