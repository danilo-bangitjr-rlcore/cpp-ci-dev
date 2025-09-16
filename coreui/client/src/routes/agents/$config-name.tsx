import {
  createFileRoute,
  Link,
  Outlet,
  useParams,
} from '@tanstack/react-router';
import {
  AgentNotFound,
  useConfigExists,
} from '../../components/agent-components';
import { useAgentNameQuery } from '../../utils/useAgentQueries';

export const Route = createFileRoute('/agents/$config-name')({
  component: RouteComponent,
});

function RouteComponent() {
  const params = useParams({ from: '/agents/$config-name' });
  const configName = params['config-name'];
  const { data, isLoading, error } = useConfigExists(configName);
  const { data: agentName } = useAgentNameQuery(configName);
  const exists = data ?? false;
  const errorMessage = error?.message ?? null;

  // Show loading state
  if (isLoading) {
    return (
      <div className="flex-1 p-2 bg-gray-50 rounded">
        <div className="flex items-center justify-center min-h-[200px]">
          <div className="text-gray-500">Loading agent...</div>
        </div>
      </div>
    );
  }

  // Show error state
  if (errorMessage) {
    return (
      <div className="flex-1 p-2 bg-gray-50 rounded">
        <div className="bg-red-50 border border-red-200 rounded-md p-4">
          <div className="text-red-800 font-medium">Error checking agent</div>
          <div className="text-red-600 text-sm mt-1">{errorMessage}</div>
        </div>
      </div>
    );
  }

  // Show 404 if agent doesn't exist
  if (exists === false) {
    return (
      <div className="flex-1 p-2 bg-gray-50 rounded">
        <AgentNotFound configName={configName} />
      </div>
    );
  }

  // Agent exists, show normal content
  return (
    <div className="flex-1 p-2 bg-gray-50 rounded">
      <Link
        to="/agents/$config-name"
        params={{ 'config-name': configName }}
        className="block text-3xl font-bold text-gray-800 hover:text-blue-600 transition-colors duration-200 mb-6"
      >
        {agentName || configName}
      </Link>
      <Outlet />
    </div>
  );
}
