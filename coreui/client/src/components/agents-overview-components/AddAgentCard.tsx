import React from 'react';

interface AddAgentCardProps {
  onAdd: () => void;
}

const AddAgentCard: React.FC<AddAgentCardProps> = ({ onAdd }) => {
  return (
    <div className="border-2 border-dashed border-gray-300 p-6 rounded-lg flex items-center justify-center min-h-[200px]">
      <button
        onClick={onAdd}
        className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
      >
        Add New Agent
      </button>
    </div>
  );
};

export default AddAgentCard;
