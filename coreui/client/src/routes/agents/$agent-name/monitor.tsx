import { createFileRoute, useParams } from '@tanstack/react-router';

export const Route = createFileRoute('/agents/$agent-name/monitor')({
  component: RouteComponent,
});

function RouteComponent() {
  const params = useParams({ from: '/agents/$agent-name/monitor' });
  const agentName = params['agent-name'];
  return <div>Monitoring for {agentName}!</div>;
}
