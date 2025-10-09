import { Link } from '@tanstack/react-router';

type ConfigurationMenuProps = {
  configName: string;
  isConfigMissing: boolean;
};

export default function ConfigurationMenu({
  configName,
  isConfigMissing,
}: ConfigurationMenuProps) {
  if (isConfigMissing) {
    return (
      <div className="max-w-md mx-auto justify-center">
        <div className="bg-white p-4 rounded-lg border border-gray-200 shadow-sm">
          <h3 className="text-lg font-semibold text-gray-900 mb-3">
            Configuration Unavailable
          </h3>
          <p className="text-sm text-gray-600">
            Create a configuration file for this agent to access configuration
            options.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-md mx-auto justify-center">
      <div className="bg-white p-4 rounded-lg border border-gray-200 shadow-sm">
        <h3 className="text-lg font-semibold text-gray-900 mb-3">
          Configuration Options
        </h3>
        <ul className="space-y-2">
          <li>
            <Link
              to={'/agents/$config-name/general-settings'}
              params={{ 'config-name': configName }}
              className="block p-2 text-blue-600 hover:text-blue-800 hover:bg-blue-50 rounded transition-colors"
            >
              General Settings
            </Link>
          </li>
          <li>
            <Link
              to={'/agents/$config-name/monitor'}
              params={{ 'config-name': configName }}
              className="block p-2 text-blue-600 hover:text-blue-800 hover:bg-blue-50 rounded transition-colors"
            >
              Monitor
            </Link>
          </li>
        </ul>
      </div>
    </div>
  );
}
