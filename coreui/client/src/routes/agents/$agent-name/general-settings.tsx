import { createFileRoute } from '@tanstack/react-router'

export const Route = createFileRoute('/agents/$agent-name/general-settings')({
  component: RouteComponent,
})

function RouteComponent() {
  return <div>Hello "/agents/$agent-name/configure"!</div>
}
