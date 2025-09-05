import type { NodeDetails } from './types';

interface NodeDetailsPanelProps {
  selectedNodeDetails?: NodeDetails;
  selectedNodeId: string | null;
  isLoading: boolean;
}

export const NodeDetailsPanel = ({
  selectedNodeDetails,
  selectedNodeId,
  isLoading,
}: NodeDetailsPanelProps) => {
  const renderValue = (value: unknown) => {
    if (value === undefined || value === null) return '-';
    return String(value);
  };

  return (
    <div className="flex-1">
      <h4 className="text-md font-semibold text-gray-900 mb-3">Node Details</h4>
      <div className="p-1.5 bg-gray-50 rounded-md border border-gray-200">
        <div className="space-y-0">
          <div className="flex justify-between px-1.5 py-1 bg-white rounded-sm">
            <strong className="text-gray-700 text-xs">Node ID:</strong>
            <span className="text-gray-900 text-xs break-all">
              {isLoading
                ? 'Loading...'
                : selectedNodeDetails?.node_id ||
                  (selectedNodeId ? 'Failed to load' : 'Select a node')}
            </span>
          </div>
          <div className="flex justify-between px-1.5 py-1 bg-gray-50 rounded-sm">
            <strong className="text-gray-700 text-xs">Display Name:</strong>
            <span className="text-gray-900 text-xs">
              {isLoading
                ? 'Loading...'
                : selectedNodeDetails?.display_name || '-'}
            </span>
          </div>
          <div className="flex justify-between px-1.5 py-1 bg-white rounded-sm">
            <strong className="text-gray-700 text-xs">Node Class:</strong>
            <span className="text-gray-900 text-xs">
              {isLoading
                ? 'Loading...'
                : selectedNodeDetails?.node_class || '-'}
            </span>
          </div>
          <div className="flex justify-between px-1.5 py-1 bg-gray-50 rounded-sm">
            <strong className="text-gray-700 text-xs">Data Type:</strong>
            <span className="text-gray-900 text-xs">
              {isLoading ? 'Loading...' : selectedNodeDetails?.data_type || '-'}
            </span>
          </div>
          <div className="flex justify-between px-1.5 py-1 bg-white rounded-sm">
            <strong className="text-gray-700 text-xs">Value:</strong>
            <span className="text-gray-900 text-xs break-all">
              {isLoading
                ? 'Loading...'
                : renderValue(selectedNodeDetails?.value)}
            </span>
          </div>
          <div className="flex justify-between px-1.5 py-1 bg-gray-50 rounded-sm">
            <strong className="text-gray-700 text-xs">Description:</strong>
            <span className="text-gray-900 text-xs break-all">
              {isLoading
                ? 'Loading...'
                : selectedNodeDetails?.description || '-'}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};
