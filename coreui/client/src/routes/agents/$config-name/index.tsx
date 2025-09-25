import { createFileRoute } from '@tanstack/react-router';
import AgentDetailsContainer from '../../../components/agent-components/AgentDetailsContainer';

export const Route = createFileRoute('/agents/$config-name/')({
  component: RouteComponent,
});

function RouteComponent() {
  return <AgentDetailsContainer />;
}
