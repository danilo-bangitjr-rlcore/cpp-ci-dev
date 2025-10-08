import type { IOStatusResponse } from '../../types/agent-types';

type ExistingIOSelectorProps = {
  availableIOs: IOStatusResponse[];
  selectedIO: string;
  onSelect: (ioId: string) => void;
  isLoading: boolean;
};

export default function ExistingIOSelector({
  availableIOs,
  selectedIO,
  onSelect,
  isLoading,
}: ExistingIOSelectorProps) {
  return (
    <div className="mt-3">
      <p className="text-xs text-blue-700 mb-2">
        You can optionally use an existing I/O service:
      </p>
      <select
        value={selectedIO}
        onChange={(e) => onSelect(e.target.value)}
        className="w-full text-xs p-2 border border-blue-300 rounded bg-white"
        disabled={isLoading}
      >
        <option value="">Create new I/O service</option>
        {availableIOs.map((io) => (
          io.status.state === 'running' && (  
          <option key={io.service_id} value={io.service_id}>
            {io.service_id}
          </option>
          )
        ))}
      </select>
      {selectedIO && (
        <p className="text-xs text-blue-600 mt-1">
          Will use existing I/O: {selectedIO}
        </p>
      )}
    </div>
  );
}