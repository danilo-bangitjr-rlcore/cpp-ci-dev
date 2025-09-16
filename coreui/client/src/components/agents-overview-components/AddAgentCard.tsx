import React, { useState } from 'react';
import { useAddAgentMutation } from '../../utils/useAgentMutations';

const AddAgentCard: React.FC = () => {
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
                const name = configName.trim();
                if (!name) return;
                addMutation.mutate(name, {
                  onSuccess: () => {
                    setConfigName('');
                    setShowForm(false);
                  },
                  onError: (error) => {
                    const base = `Failed to add agent "${name}"`;
                    if (error instanceof Error) {
                      const extra = error.message?.trim();
                      const msg =
                        extra && extra !== base ? `${base}\n${extra}` : base;
                      window.alert(msg);
                    } else {
                      window.alert(base);
                    }
                  },
                });
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
