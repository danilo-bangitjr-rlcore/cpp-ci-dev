import { createFileRoute, useParams } from '@tanstack/react-router';

export const Route = createFileRoute('/agents/$config-name/general-settings')({
  component: RouteComponent,
});

function RouteComponent() {
  const params = useParams({ from: '/agents/$config-name/general-settings' });
  const configName = params['config-name'];

  return <div>General settings for {configName}!</div>;
}
