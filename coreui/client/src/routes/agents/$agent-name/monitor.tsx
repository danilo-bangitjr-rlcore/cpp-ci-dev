import { createFileRoute } from '@tanstack/react-router'

export const Route = createFileRoute('/agents/$agent-name/monitor')({
  component: RouteComponent,
})

function RouteComponent() {
  return <div>Hello "/agents/$agent-name/monitor"!</div>
}
