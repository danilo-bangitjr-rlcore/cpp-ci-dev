import React from 'react';
import { Link } from '@tanstack/react-router';
import { WarningIcon, ErrorIcon } from '../icons';
import type { Agent } from '../../types/agent-types';

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
          <WarningIcon className="w-5 h-5 text-yellow-600" />
        ) : (
          <ErrorIcon className="w-5 h-5 text-red-600" />
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
  const alertType: AlertType =
    Math.random() > 0.7 ? (Math.random() > 0.5 ? 'yellow' : 'red') : null;

  const getStatusDisplay = (status: Agent['status']) => {
    switch (status) {
      case 'on':
        return { text: 'ON', colorClass: 'text-green-700' };
      case 'off':
        return { text: 'OFF', colorClass: 'text-gray-700' };
      case 'error':
        return { text: 'ERROR', colorClass: 'text-red-700' };
      case 'never-started':
        return { text: 'NEVER STARTED', colorClass: 'text-yellow-700' };
      case 'config-missing':
        return { text: 'CONFIG MISSING', colorClass: 'text-orange-700' };
      default:
        return { text: 'UNKNOWN', colorClass: 'text-gray-700' };
    }
  };

  const statusDisplay = getStatusDisplay(agent.status);
  const isConfigMissing = agent.status === 'config-missing';

  return (
    <div
      className={`border ${isConfigMissing ? 'border-orange-400' : 'border-gray-300'} bg-gray-100 p-4 rounded-lg shadow-sm min-h-[400px] flex flex-col space-y-4`}
    >
      <div className="bg-white p-3 rounded-md border border-gray-200">
        <h3 className="text-xl font-bold text-gray-900 text-center">
          {agent.agentName}
        </h3>
      </div>

      {isConfigMissing && (
        <div className="bg-orange-50 p-3 rounded-md border border-orange-200">
          <div className="flex items-start space-x-2">
            <WarningIcon className="w-5 h-5 text-orange-600 flex-shrink-0 mt-0.5" />
            <div className="flex-1">
              <p className="text-sm font-medium text-orange-800">
                Configuration file missing
              </p>
              <p className="text-xs text-orange-700 mt-1">
                This agent is running but has no associated config file. It may
                have been started manually or the config was deleted.
              </p>
            </div>
          </div>
        </div>
      )}

      <div className="bg-white p-3 rounded-md border border-gray-200 space-y-3">
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-gray-700">
              Agent Status:
            </span>
            <div
              className={`${isConfigMissing ? 'bg-orange-100' : 'bg-gray-100'} px-3 py-1 rounded-md`}
            >
              <span
                className={`text-sm font-medium ${statusDisplay.colorClass}`}
              >
                {statusDisplay.text}
              </span>
            </div>
          </div>

          {!isConfigMissing && (
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
          )}
        </div>
      </div>

      {!isConfigMissing && <MockOptimizationStats />}

      {!isConfigMissing && alertType && (
        <div className="bg-white p-3 rounded-md border border-gray-200">
          <h4 className="text-sm font-semibold text-gray-800 mb-2">Alerts</h4>
          <AlertSection type={alertType} />
        </div>
      )}

      <div className="bg-white p-3 rounded-md border border-gray-200 mt-auto">
        {isConfigMissing ? (
          <div className="text-center py-2">
            <p className="text-sm text-gray-600 mb-2">
              Cannot view details without config
            </p>
            <p className="text-xs text-gray-500">
              Create a config file to manage this agent
            </p>
          </div>
        ) : (
          <Link
            to="/agents/$config-name"
            params={{ 'config-name': agent.configName }}
            className="block w-full bg-gray-500 text-white text-center py-2 rounded hover:bg-gray-600 transition-colors font-medium"
          >
            Details
          </Link>
        )}
      </div>
    </div>
  );
};

export default AgentCard;
