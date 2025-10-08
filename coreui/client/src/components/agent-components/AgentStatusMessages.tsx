import type { IOStatusResponse } from '../../types/agent-types';
import ExistingIOSelector from './ExistingIOSelector';

type AgentStatusMessagesProps = {
  isConfigMissing: boolean;
  isNeverStarted: boolean;
  resolvedConfigPath?: string;
  isLoadingConfigPath: boolean;
  availableIOs: IOStatusResponse[];
  isLoadingIOs: boolean;
  selectedExistingIO: string;
  onSelectIO: (ioId: string) => void;
  agentError?: Error | null;
  ioError?: Error | null;
};

export default function AgentStatusMessages({
  isConfigMissing,
  isNeverStarted,
  resolvedConfigPath,
  isLoadingConfigPath,
  availableIOs,
  isLoadingIOs,
  selectedExistingIO,
  onSelectIO,
  agentError,
  ioError,
}: AgentStatusMessagesProps) {
  return (
    <div className="space-y-4">
      {isConfigMissing && (
        <div className="mx-auto max-w-2xl p-4 bg-orange-50 border border-orange-200 rounded-lg">
          <div className="flex items-start space-x-2">
            <svg
              className="w-5 h-5 text-orange-600 flex-shrink-0 mt-0.5"
              fill="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                fillRule="evenodd"
                d="M8.485 2.495c.673-1.167 2.357-1.167 3.03 0l6.28 10.875c.673 1.167-.17 2.625-1.516 2.625H3.72c-1.347 0-2.189-1.458-1.515-2.625L8.485 2.495zM12 9a.75.75 0 01.75.75v3.5a.75.75 0 01-1.5 0v-3.5A.75.75 0 0112 9zm0 8a1 1 0 100-2 1 1 0 000 2z"
                clipRule="evenodd"
              />
            </svg>
            <div>
              <p className="text-sm font-medium text-orange-800">
                Configuration file missing for this agent
              </p>
              <p className="text-xs text-orange-700 mt-1">
                This agent is running without an associated config file. It may
                have been started manually or the config was deleted. You can
                still manage the running services, but configuration options are
                not available.
              </p>
            </div>
          </div>
        </div>
      )}

      {isNeverStarted && (
        <div className="mx-auto max-w-2xl p-4 bg-blue-50 border border-blue-200 rounded-lg">
          <div className="mx-auto max-w-2xl p-4 bg-blue-50 border border-blue-200 rounded-lg">
          <div className="flex items-start space-x-2">
            <svg
              className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5"
              fill="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                fillRule="evenodd"
                d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a.75.75 0 000 1.5h.253a.25.25 0 01.244.304l-.459 2.066A1.75 1.75 0 0010.747 15H11a.75.75 0 000-1.5h-.253a.25.25 0 01-.244-.304l.459-2.066A1.75 1.75 0 009.253 9H9z"
                clipRule="evenodd"
              />
            </svg>
            <div className="flex-1">
              <p className="text-sm font-medium text-blue-800">
                This agent has never been started
              </p>
              <p className="text-xs text-blue-700 mt-1">
                Before starting, please verify the configuration path:
              </p>
              {isLoadingConfigPath ? (
                <p className="text-xs text-blue-700 mt-2 font-mono bg-blue-100 p-2 rounded">
                  Loading config path...
                </p>
              ) : resolvedConfigPath ? (
                <p className="text-xs text-blue-700 mt-2 font-mono bg-blue-100 p-2 rounded break-all">
                  {resolvedConfigPath}
                </p>
              ) : (
                <p className="text-xs text-red-700 mt-2">
                  Could not resolve config path
                </p>
              )}
            </div>
          </div>
        </div>
          {availableIOs.length > 0 && (
            <ExistingIOSelector
              availableIOs={availableIOs}
              selectedIO={selectedExistingIO}
              onSelect={onSelectIO}
              isLoading={isLoadingIOs}
            />
          )}
        </div>
      )}

      {(agentError || ioError) && (
        <div className="mx-auto max-w-2xl p-4 bg-red-50 border border-red-200 rounded-lg">
          <h4 className="text-sm font-medium text-red-800 mb-2">
            Recent Errors:
          </h4>
          {agentError && (
            <p className="text-sm text-red-700 mb-1">
              Agent: {agentError.message}
            </p>
          )}
          {ioError && (
            <p className="text-sm text-red-700">
              I/O: {ioError.message}
            </p>
          )}
        </div>
      )}
    </div>
  );
}