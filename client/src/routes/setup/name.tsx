import { createFileRoute } from '@tanstack/react-router'

export const Route = createFileRoute('/setup/name')({
  component: RouteComponent,
})

function RouteComponent() {
  return <div className="p-2">
    <p><code>/setup/name</code> nested route</p>
    <p>TODO: experiment name text input form</p>
  </div>
}
