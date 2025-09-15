import { createFileRoute } from '@tanstack/react-router';
import AgentsOverviewContainer from '../../components/agents-overview-components/AgentsOverviewContainer';

export const Route = createFileRoute('/agents/')({
  component: RouteComponent,
});

function RouteComponent() {
  return <AgentsOverviewContainer />;
}
