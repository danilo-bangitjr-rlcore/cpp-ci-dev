import { createFileRoute } from '@tanstack/react-router';
import { useState } from 'react';

export const Route = createFileRoute('/opc-navigation')({
  component: RouteComponent,
});

function RouteComponent() {
  const [url, setUrl] = useState('');
  const [status, setStatus] = useState('Disconnected');
  const [isConnecting, setIsConnecting] = useState(false);

  const handleConnect = async () => {
    if (!url.trim()) {
      setStatus('Please enter a valid URL');
      return;
    }

    setIsConnecting(true);
    setStatus('Connecting...');

    // TODO: Implement actual OPC server connection
    // For now, simulate connection attempt
    setTimeout(() => {
      setStatus('Connected successfully');
      setIsConnecting(false);
    }, 1500);
  };

  return (
    <div className="min-h-screen bg-gray-900">
      <div className="max-w-6xl mx-auto p-4">
        <div className="bg-gray-800 rounded-lg shadow-md p-4">
          <h1 className="text-xl font-bold text-white mb-2">
            OPC Server Connection
          </h1>

          <div className="flex items-end gap-4">
            {/* URL Input */}
            <div className="flex-1">
              <label
                htmlFor="opc-url"
                className="block text-sm font-medium text-gray-300 mb-1"
              >
                OPC Server URL
              </label>
              <input
                id="opc-url"
                type="text"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                placeholder="opc.tcp://localhost:4840"
                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md shadow-sm text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              />
            </div>

            {/* Connect Button */}
            <div>
              <button
                onClick={handleConnect}
                disabled={isConnecting}
                className={`px-6 py-2 rounded-md font-medium text-white transition-colors whitespace-nowrap ${
                  isConnecting
                    ? 'bg-gray-400 cursor-not-allowed'
                    : 'bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500'
                }`}
              >
                {isConnecting ? 'Connecting...' : 'Connect'}
              </button>
            </div>

            {/* Status */}
            <div className="min-w-[150px]">
              <div
                className={`px-3 py-2 rounded-md text-sm font-medium text-center ${
                  status === 'Connected successfully'
                    ? 'bg-green-900 text-green-200'
                    : status === 'Connecting...'
                      ? 'bg-yellow-900 text-yellow-200'
                      : status.includes('Please enter')
                        ? 'bg-red-900 text-red-200'
                        : 'bg-gray-700 text-gray-300'
                }`}
              >
                {status}
              </div>
            </div>
          </div>
        </div>

        {/* Separator line */}
        <div className="border-b border-gray-600 mt-4"></div>
      </div>
    </div>
  );
}
