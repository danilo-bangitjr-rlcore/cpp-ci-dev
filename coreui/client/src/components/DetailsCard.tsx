import React from 'react';

type DetailsCardProps = {
  entityName: string;
  state: 'running' | 'stopped' | 'never-started';
  onToggleStatus: () => void;
  isLoading: boolean;
  metadata: { label: string; value: string }[];
  metadataTitle: string;
  isFirstStart?: boolean;
  isUsingExisting?: boolean;
};

const PilotLight: React.FC<{ state: 'running' | 'stopped' | 'never-started' }> = ({ state }) => {
  const imagePath =
    state === 'running'
      ? '/app/assets/pilot_light_green.svg'
      : '/app/assets/pilot_light_red.svg';

  return (
    <div className="flex justify-center">
      <img src={imagePath} alt={`Agent ${state}`} className="w-32 h-24" />
    </div>
  );
};

const DetailsCard: React.FC<DetailsCardProps> = ({
  entityName,
  state,
  onToggleStatus,
  isLoading = false,
  metadata = [],
  metadataTitle = 'Metadata',
  isFirstStart = false,
  isUsingExisting = false,
}) => {
  const getButtonText = () => {
    if (isLoading) return 'Loading...';
    if (isFirstStart) {
      return isUsingExisting ? 'Start with Existing I/O' : 'Start for the First Time';
    }
    return state === 'running' ? 'Stop' : 'Start';
  };

  const getButtonStyle = () => {
    if (state === 'running') {
      return 'bg-red-500 hover:bg-red-600 text-white';
    }
    if (state === 'never-started' || isFirstStart) {
      return isUsingExisting
        ? 'bg-purple-500 hover:bg-purple-600 text-white'
        : 'bg-blue-500 hover:bg-blue-600 text-white';
    }
    return 'bg-green-500 hover:bg-green-600 text-white';
  };

  return (
    <div className="border border-gray-300 bg-gray-100 p-4 rounded-lg shadow-sm space-y-4 w-1/3">
      <div className="bg-white p-3 rounded-md border border-gray-200">
        <h3 className="text-xl font-bold text-gray-900 text-center">
          {entityName}
        </h3>
        {isUsingExisting && (
          <p className="text-xs text-purple-600 text-center mt-1">
            Using existing service
          </p>
        )}
      </div>

      <div className="bg-white p-3 rounded-md border border-gray-200">
        <h4 className="text-sm font-semibold text-gray-800 mb-3 text-center">
          Status
        </h4>
        <PilotLight state={state} />
      </div>

      <div className="bg-white p-3 rounded-md border border-gray-200">
        <h4 className="text-sm font-semibold text-gray-800 mb-2">Controls</h4>
        <button
          onClick={onToggleStatus}
          disabled={isLoading}
          className={`w-full py-2 px-4 rounded-md font-medium transition-colors ${getButtonStyle()} ${
            isLoading ? 'opacity-50 cursor-not-allowed' : ''
          }`}
        >
          {getButtonText()}
        </button>
      </div>

      {metadata.length > 0 && (
        <div className="bg-white p-3 rounded-md border border-gray-200">
          <h4 className="text-sm font-semibold text-gray-800 mb-2">
            {metadataTitle}
          </h4>
          <div className="space-y-2 text-sm">
            {metadata.map((item, index) => (
              <div key={index} className="flex justify-between">
                <span className="text-gray-600">{item.label}:</span>
                <span className="font-medium text-gray-900">
                  {item.value}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default DetailsCard;
