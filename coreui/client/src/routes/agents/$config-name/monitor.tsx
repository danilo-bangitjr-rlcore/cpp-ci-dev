import { createFileRoute, useParams } from '@tanstack/react-router';

export const Route = createFileRoute('/agents/$config-name/monitor')({
  component: RouteComponent,
});

function RouteComponent() {
  const params = useParams({ from: '/agents/$config-name/monitor' });
  const configName = params['config-name'];
  return <div>Monitoring for {configName}!</div>;
}
