import React from 'react';

type ConfigDetailsProps = {
  name: string;
  onNameChange: (name: string) => void;
  connection_id: string;
  onConnectionIdChange: (id: string) => void;
};

export const ConfigDetails: React.FC<ConfigDetailsProps> = ({
  name,
  onNameChange,
  connection_id,
  onConnectionIdChange,
}) => (
  <div className="flex flex-col gap-4 w-60 justify-center">
    <input
      type="text"
      value={name}
      onChange={(e) => onNameChange(e.target.value)}
      placeholder="Config Name"
      className="text-lg font-bold bg-white border border-gray-300 p-2 rounded"
    />
    <input
      type="text"
      value={connection_id}
      onChange={(e) => onConnectionIdChange(e.target.value)}
      placeholder="OPC Address (connection_id)"
      className="bg-white border border-gray-300 p-2 rounded"
    />
  </div>
);
