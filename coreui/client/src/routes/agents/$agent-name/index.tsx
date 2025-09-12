import { createFileRoute, Link, useParams } from '@tanstack/react-router'

export const Route = createFileRoute('/agents/$agent-name/')({
  component: RouteComponent,
})

function RouteComponent() {
  const params = useParams({ from: '/agents/$agent-name/' })
  const agentName = params['agent-name']
  
  return (
    <div>
      <ul className="list-disc list-inside space-y-2">
        <li>
          <Link 
            to={'/agents/$agent-name/general-settings'} 
            params={{ 'agent-name': agentName }}
            className="text-blue-600 hover:text-blue-800 hover:underline"
          >
            General Settings
          </Link>
        </li>
        <li>
          <Link 
            to={'/agents/$agent-name/monitor'} 
            params={{ 'agent-name': agentName }}
            className="text-blue-600 hover:text-blue-800 hover:underline"
          >
            Monitor
          </Link>
        </li>
      </ul>
    </div>
  )
}
