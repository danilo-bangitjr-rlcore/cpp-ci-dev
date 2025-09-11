import React, { useState } from 'react';
import { useAddAgentMutation } from './useAgentMutations';

interface AddAgentCardProps {}

const AddAgentCard: React.FC<AddAgentCardProps> = () => {
  const [showForm, setShowForm] = useState(false);
  const [configName, setConfigName] = useState('');
  const addMutation = useAddAgentMutation();

  return (
    <div className="border-2 border-dashed border-gray-300 p-6 rounded-lg flex items-center justify-center min-h-[200px]">
      {showForm ? (
        <div className="flex flex-col items-center space-y-4">
          <input
            type="text"
            value={configName}
            onChange={(e) => setConfigName(e.target.value)}
            placeholder="Enter agent name"
            className="px-4 py-2 border border-gray-300 rounded"
          />
          <div className="flex space-x-2">
            <button
              onClick={() => {
                if (configName.trim()) {
                  const name = configName.trim();
                  addMutation.mutate(name, {
                    onError: () => {
                      window.alert(`Failed to add agent "${name}"`);
                    },
                  });
                  if (!addMutation.isPending) {
                    setConfigName('');
                    setShowForm(false);
                  }
                }
              }}
              disabled={!configName.trim() || addMutation.isPending}
              className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 disabled:opacity-50"
            >
              {addMutation.isPending ? 'Adding...' : 'Add'}
            </button>
            <button
              onClick={() => setShowForm(false)}
              className="bg-gray-500 text-white px-4 py-2 rounded hover:bg-gray-600"
            >
              Cancel
            </button>
          </div>
          {/* Error feedback now provided via window.alert in onError */}
        </div>
      ) : (
        <button
          onClick={() => setShowForm(true)}
          className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
        >
          Add New Agent
        </button>
      )}
    </div>
  );
};

export default AddAgentCard;
