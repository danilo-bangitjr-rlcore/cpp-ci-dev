import { createFileRoute, useParams } from '@tanstack/react-router';

export const Route = createFileRoute('/agents/$agent-name/general-settings')({
  component: RouteComponent,
});

function RouteComponent() {
  const params = useParams({ from: '/agents/$agent-name/general-settings' });
  const agentName = params['agent-name'];

  return <div>General settings for {agentName}!</div>;
}
