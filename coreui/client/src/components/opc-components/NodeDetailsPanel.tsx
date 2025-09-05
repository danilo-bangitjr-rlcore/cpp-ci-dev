import { useState } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import type { NodeDetails } from './types';
import { API_BASE_URL } from '../../utils/api';

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
  const [writeValue, setWriteValue] = useState('');
  const queryClient = useQueryClient();

  const writeMutation = useMutation({
    mutationFn: async ({ nodeId, value }: { nodeId: string; value: string }) => {
      const encodedNodeId = encodeURIComponent(nodeId);
      const response = await fetch(`${API_BASE_URL}/v1/opc/write/${encodedNodeId}?value=${encodeURIComponent(value)}`, {
        method: 'POST',
      });
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(errorData.detail || 'Failed to write value');
      }
      return response.json();
    },
    onSuccess: (_, variables) => {
      // Invalidate the node details query to refresh the value
      queryClient.invalidateQueries({
        queryKey: ['opc-node-details', variables.nodeId],
      });
      // Also invalidate all node details queries to be safe
      queryClient.invalidateQueries({
        queryKey: ['opc-node-details'],
      });
      setWriteValue(''); // Clear the input
      // Reset success state after 3 seconds
      setTimeout(() => {
        writeMutation.reset();
      }, 3000);
    },
  });

  const handleWrite = () => {
    if (!selectedNodeId || !writeValue.trim()) return;
    writeMutation.mutate({ nodeId: selectedNodeId, value: writeValue.trim() });
  };

  const renderValue = (value: unknown) => {
    if (value === undefined || value === null) return '-';
    return String(value);
  };

  const isVariable = selectedNodeDetails && selectedNodeDetails.node_class === 'Variable' && !isLoading;

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
          {isVariable && !isLoading && (
            <div className="px-1.5 py-2 bg-gray-50 rounded-sm">
              <div className="flex items-center space-x-2">
                <input
                  type="text"
                  value={writeValue}
                  onChange={(e) => setWriteValue(e.target.value)}
                  placeholder="Enter new value"
                  className="flex-1 px-2 py-1 text-xs border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-blue-500"
                  disabled={writeMutation.isPending}
                />
                <button
                  onClick={handleWrite}
                  disabled={!writeValue.trim() || writeMutation.isPending}
                  className="px-3 py-1 text-xs bg-blue-600 text-white rounded hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed"
                >
                  {writeMutation.isPending ? 'Sending...' : 'Send'}
                </button>
              </div>
              {writeMutation.isError && (
                <p className="text-red-600 text-xs mt-1">
                  Error: {(writeMutation.error as Error)?.message}
                </p>
              )}
              {writeMutation.isSuccess && (
                <p className="text-green-600 text-xs mt-1">Value updated successfully</p>
              )}
            </div>
          )}
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
