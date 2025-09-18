import { createFileRoute, Link, useParams } from '@tanstack/react-router';

export const Route = createFileRoute('/agents/$config-name/')({
  component: RouteComponent,
});

function RouteComponent() {
  const params = useParams({ from: '/agents/$config-name/' });
  const configName = params['config-name'];

  return (
    <div>
      <ul className="list-disc list-inside space-y-2">
        <li>
          <Link
            to={'/agents/$config-name/general-settings'}
            params={{ 'config-name': configName }}
            className="text-blue-600 hover:text-blue-800 hover:underline"
          >
            General Settings
          </Link>
        </li>
        <li>
          <Link
            to={'/agents/$config-name/monitor'}
            params={{ 'config-name': configName }}
            className="text-blue-600 hover:text-blue-800 hover:underline"
          >
            Monitor
          </Link>
        </li>
      </ul>
    </div>
  );
}
