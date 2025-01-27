import { createFileRoute, Outlet } from '@tanstack/react-router'

export const Route = createFileRoute('/setup')({
  component: RouteComponent,
})

function RouteComponent() {
  return <>
    <div className="p-2">
      <p><code>/setup</code> Base Route</p>
    </div>
    <Outlet />
  </>
}
