import { createFileRoute, Link } from '@tanstack/react-router'

export const Route = createFileRoute('/setup/')({
  component: RouteComponent,
})

function RouteComponent() {
  return <div className='p-2'>
    <p><code>/setup/index</code> nested route</p>
    <Link to="/setup/name">Go to /setup/name</Link>
  </div>
}
