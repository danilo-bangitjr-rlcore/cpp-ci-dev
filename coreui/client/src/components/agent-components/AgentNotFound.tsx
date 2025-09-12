import { Link } from '@tanstack/react-router'

interface AgentNotFoundProps {
  agentName: string
}

export function AgentNotFound({ agentName }: AgentNotFoundProps) {
  return (
    <div className="flex flex-col items-center justify-center min-h-[400px] bg-white rounded-lg shadow-md p-8">
      <div className="text-center">
        <div className="text-6xl text-gray-300 mb-4">404</div>
        <h1 className="text-2xl font-bold text-gray-800 mb-2">Agent Not Found</h1>
        <p className="text-gray-600 mb-6">
          The agent <span className="font-mono bg-gray-100 px-2 py-1 rounded">{agentName}</span> could not be found.
        </p>
        <div className="space-y-3">
          <p className="text-sm text-gray-500">
            This agent may not exist or may have been removed.
          </p>
          <Link 
            to="/agents" 
            className="inline-block bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 transition-colors duration-200"
          >
            ← Back to Agents
          </Link>
        </div>
      </div>
    </div>
  )
}
