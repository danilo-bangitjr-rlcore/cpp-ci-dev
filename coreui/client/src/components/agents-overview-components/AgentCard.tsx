import React from 'react';
import { Link } from '@tanstack/react-router';
import { type Agent } from './AgentsOverviewContainer';

interface AgentCardProps {
  agent: Agent;
}

type AlertType = 'yellow' | 'red' | null;

interface AlertProps {
  type: AlertType;
}

const AlertSection: React.FC<AlertProps> = ({ type }) => {
  if (!type) return null;

  const isYellow = type === 'yellow';
  const bgColor = isYellow ? 'bg-yellow-50' : 'bg-red-50';
  const textColor = isYellow ? 'text-yellow-800' : 'text-red-800';
  const borderColor = isYellow ? 'border-yellow-200' : 'border-red-200';

  return (
    <div className={`p-3 rounded-md border ${bgColor} ${borderColor}`}>
      <div className="flex items-center space-x-2">
        {isYellow ? (
          <svg
            className="w-5 h-5 text-yellow-600"
            fill="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              fillRule="evenodd"
              d="M8.485 2.495c.673-1.167 2.357-1.167 3.03 0l6.28 10.875c.673 1.167-.17 2.625-1.516 2.625H3.72c-1.347 0-2.189-1.458-1.515-2.625L8.485 2.495zM12 9a.75.75 0 01.75.75v3.5a.75.75 0 01-1.5 0v-3.5A.75.75 0 0112 9zm0 8a1 1 0 100-2 1 1 0 000 2z"
              clipRule="evenodd"
            />
          </svg>
        ) : (
          <svg
            className="w-5 h-5 text-red-600"
            fill="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              fillRule="evenodd"
              d="M12 2.25c-5.385 0-9.75 4.365-9.75 9.75s4.365 9.75 9.75 9.75 9.75-4.365 9.75-9.75S17.385 2.25 12 2.25zm-1.72 6.97a.75.75 0 10-1.06 1.06L10.94 12l-1.72 1.72a.75.75 0 101.06 1.06L12 13.06l1.72 1.72a.75.75 0 101.06-1.06L13.06 12l1.72-1.72a.75.75 0 10-1.06-1.06L12 10.94l-1.72-1.72z"
              clipRule="evenodd"
            />
          </svg>
        )}
        <span className={`text-sm font-medium ${textColor}`}>
          {isYellow ? 'Yellow Zone Violation' : 'Red Zone Violation'}
        </span>
      </div>
    </div>
  );
};

const MockOptimizationStats: React.FC = () => {
  return (
    <div className="bg-white p-3 rounded-md border border-gray-200">
      <h4 className="text-sm font-semibold text-gray-800 mb-2">
        Optimization Stats
      </h4>
      <div className="space-y-2 text-sm">
        <div className="flex justify-between">
          <span className="text-gray-600">Weekly baseline chemical costs:</span>
          <span className="font-medium text-gray-900">
            ${(Math.random() * 350.54).toFixed(2)}
          </span>
        </div>
        <div className="flex justify-between">
          <span className="text-gray-600">Weekly live chemical costs:</span>
          <span className="font-medium text-green-600">
            ${(Math.random() * 350.54).toFixed(2)}
          </span>
        </div>
        <div className="flex justify-between">
          <span className="text-gray-600">
            Consumption compared to baseline:
          </span>
          <span className="font-medium text-blue-600">
            {(Math.random() * 100).toFixed(2)}%
          </span>
        </div>
      </div>
    </div>
  );
};

const AgentCard: React.FC<AgentCardProps> = ({ agent }) => {
  // Mock alert state - in real implementation this would come from props or API
  const alertType: AlertType =
    Math.random() > 0.7 ? (Math.random() > 0.5 ? 'yellow' : 'red') : null;

  return (
    <div className="border border-gray-300 bg-gray-100 p-4 rounded-lg shadow-sm min-h-[400px] flex flex-col space-y-4">
      <div className="bg-white p-3 rounded-md border border-gray-200">
        <h3 className="text-xl font-bold text-gray-900 text-center">
          {agent.agentName}
        </h3>
      </div>

      <div className="bg-white p-3 rounded-md border border-gray-200 space-y-3">
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-gray-700">
              Agent Status:
            </span>
            <div className="bg-gray-100 px-3 py-1 rounded-md">
              <span
                className={`text-sm font-medium ${
                  agent.status === 'on'
                    ? 'text-green-700'
                    : agent.status === 'off'
                      ? 'text-gray-700'
                      : 'text-red-700'
                }`}
              >
                {agent.status.toUpperCase()}
              </span>
            </div>
          </div>

          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-gray-700">
              Current Objective:
            </span>
            <button
              className="bg-gray-100 hover:bg-gray-200 px-3 py-1 rounded-md transition-colors"
              onClick={() => console.log('Optimize clicked')}
            >
              <span className="text-sm font-medium text-gray-700">
                Optimize
              </span>
            </button>
          </div>
        </div>
      </div>

      <MockOptimizationStats />

      {alertType && (
        <div className="bg-white p-3 rounded-md border border-gray-200">
          <h4 className="text-sm font-semibold text-gray-800 mb-2">Alerts</h4>
          <AlertSection type={alertType} />
        </div>
      )}

      <div className="bg-white p-3 rounded-md border border-gray-200 mt-auto">
        <Link
          to="/agents/$config-name"
          params={{ 'config-name': agent.configName }}
          className="block w-full bg-gray-500 text-white text-center py-2 rounded hover:bg-gray-600 transition-colors font-medium"
        >
          Details
        </Link>
      </div>
    </div>
  );
};

export default AgentCard;
