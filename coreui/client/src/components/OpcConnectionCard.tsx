import { useState } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { API_ENDPOINTS } from '../utils/api';

interface StatusResponse {
  connected: boolean;
  server_url: string | null;
  message?: string;
  error?: string;
  server_info?: {
    node_id: string;
    display_name: string;
  };
}

interface OpcConnectionCardProps {
  statusData?: StatusResponse;
  statusError: Error | null;
  isStatusLoading: boolean;
  className?: string;
}

const isValidOpcUrl = (url: string): boolean => {
  const trimmed = url.trim();
  return trimmed.startsWith('opc.tcp://') && trimmed.length > 10;
};

export function OpcConnectionCard({ 
  statusData, 
  statusError, 
  isStatusLoading, 
  className = '' 
}: OpcConnectionCardProps) {
  const [url, setUrl] = useState('');
  const [showValidationError, setShowValidationError] = useState(false);
  const queryClient = useQueryClient();

  const connectMutation = useMutation({
    mutationFn: async (url: string) => {
      const response = await fetch(
        `${API_ENDPOINTS.opc.connect}?url=${encodeURIComponent(url)}`,
        {
          method: 'POST',
        }
      );

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(
          errorData.detail || `HTTP error! status: ${response.status}`
        );
      }

      return response.json();
    },
    onSuccess: () => {
      setShowValidationError(false);
      queryClient.invalidateQueries({ queryKey: ['opc-status'] });
    },
    onError: () => {
      queryClient.invalidateQueries({ queryKey: ['opc-status'] });
    },
  });

  const disconnectMutation = useMutation({
    mutationFn: async () => {
      const response = await fetch(API_ENDPOINTS.opc.disconnect, {
        method: 'POST',
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(
          errorData.detail || `HTTP error! status: ${response.status}`
        );
      }

      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['opc-status'] });
    },
  });

  const getDisplayStatus = (): string => {
    if (connectMutation.isPending) return 'Connecting...';
    if (disconnectMutation.isPending) return 'Disconnecting...';
    if (showValidationError) return 'Please enter a valid url';
    if (connectMutation.error && !showValidationError) return 'Error';
    if (statusError) return 'Error';
    if (isStatusLoading) return 'Checking...';

    if (statusData) {
      if (statusData.connected) return 'Connected';
      if (statusData.error) return 'Error';
      return 'Disconnected';
    }

    return 'Disconnected';
  };

  const isConnected = statusData?.connected === true;
  const displayStatus = getDisplayStatus();
  const hasError = connectMutation.error || statusError;

  const handleConnect = () => {
    setShowValidationError(false);

    if (!url.trim() || !isValidOpcUrl(url)) {
      setShowValidationError(true);
      return;
    }

    connectMutation.mutate(url);
  };

  const handleDisconnect = () => {
    setShowValidationError(false);
    disconnectMutation.mutate();
  };

  const getStatusColorClass = (): string => {
    if (statusData?.connected)
      return 'bg-green-100 text-green-800 border border-green-200';
    if (
      connectMutation.isPending ||
      disconnectMutation.isPending ||
      isStatusLoading
    )
      return 'bg-yellow-100 text-yellow-800 border border-yellow-200';
    if (
      showValidationError ||
      connectMutation.error ||
      statusError ||
      statusData?.error
    )
      return 'bg-red-100 text-red-800 border border-red-200';
    return 'bg-gray-100 text-gray-600 border border-gray-200';
  };

  return (
    <div className={`p-4 bg-white rounded-lg border border-gray-200 shadow-sm ${className}`}>
      <h2 className="text-lg font-semibold text-gray-900 mb-4">
        OPC Server Connection
      </h2>

      <div className="flex items-end gap-4">
        <div className="flex-1">
          {isConnected ? (
            <div>
              <label className="block text-sm font-medium text-gray-600 mb-1">
                Connected to
              </label>
              <div className="px-3 py-2 bg-green-50 border border-green-200 rounded-md text-gray-900">
                {statusData?.server_url}
              </div>
            </div>
          ) : (
            <>
              <label
                htmlFor="opc-url"
                className="block text-sm font-medium text-gray-600 mb-1"
              >
                OPC Server URL
              </label>
              <input
                id="opc-url"
                type="text"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                placeholder="opc.tcp://localhost:4840"
                className="w-full px-3 py-2 bg-white border border-gray-200 rounded-md shadow-sm text-gray-900 placeholder-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 disabled:bg-gray-50 disabled:text-gray-500"
              />
            </>
          )}
        </div>

        <div>
          <button
            onClick={isConnected ? handleDisconnect : handleConnect}
            disabled={
              connectMutation.isPending || disconnectMutation.isPending
            }
            className={`px-6 py-2 rounded-md font-medium text-white transition-colors whitespace-nowrap ${
              connectMutation.isPending || disconnectMutation.isPending
                ? 'bg-gray-400 cursor-not-allowed'
                : isConnected
                  ? 'bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500'
                  : 'bg-green-600 hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500'
            }`}
          >
            {connectMutation.isPending
              ? 'Connecting...'
              : disconnectMutation.isPending
                ? 'Disconnecting...'
                : isConnected
                  ? 'Disconnect'
                  : 'Connect'}
          </button>
        </div>

        <div className="min-w-[150px]">
          <div
            className={`px-3 py-2 rounded-md text-sm font-medium text-center ${getStatusColorClass()}`}
          >
            {displayStatus}
          </div>
        </div>
      </div>

      {hasError && (
        <div className="mt-4 text-xs text-gray-500 bg-red-50 p-2 rounded border inline-block">
          <strong>Error:</strong>{' '}
          {connectMutation.error?.message || statusError?.message}
        </div>
      )}
    </div>
  );
}
