import React from 'react';
import { ErrorIcon } from '../icons';

interface ConfigErrorCardProps {
  configName: string;
  errorMessage: string;
}

const ConfigErrorCard: React.FC<ConfigErrorCardProps> = ({
  configName,
  errorMessage,
}) => {
  return (
    <div className="border border-red-400 bg-gray-100 p-4 rounded-lg shadow-sm min-h-[400px] flex flex-col justify-center items-center space-y-4">
      <ErrorIcon className="w-16 h-16 text-red-600" />

      <div className="text-center space-y-2">
        <h3 className="text-lg font-bold text-gray-900">Configuration Error</h3>
        <p className="text-sm font-mono text-gray-800 bg-white px-3 py-1 rounded border border-gray-300">
          {configName}
        </p>
        <p className="text-sm text-gray-700 max-w-xs">{errorMessage}</p>
      </div>

      <div className="text-xs text-gray-600 text-center max-w-xs">
        This configuration file cannot be loaded as an agent. Please check the
        YAML file for required fields.
      </div>
    </div>
  );
};

export default ConfigErrorCard;
