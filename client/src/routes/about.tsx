import { createFileRoute } from '@tanstack/react-router'
import createClient from 'openapi-react-query'
import { getApiFetchClient } from '../utils/api'

const About = () => {
  const client = createClient(getApiFetchClient())
  const { status, error, data, refetch } = client.useQuery('get', '/health', {
    headers: { Accept: 'application/json' },
  })

  return (
    <div className="p-2">
      <h2 className="text-2xl">About</h2>
      <p>
        About page. TBD: show core-rl version information, web client details,
        experiment/agent details.
      </p>
      <div className="p-2">
        <h3 className="text-xl">Health</h3>
        <p>{status}</p>
        {error && <div>Error: {`${error}`}</div>}
        {data && <pre>{JSON.stringify(data, null, 2)}</pre>}
        <button className="text-white bg-blue-500 px-4 py-2 rounded hover:bg-blue-600"
          onClick={() => {
            void refetch()
          }}
        >
          Re-fetch Health Status
        </button>
      </div>
    </div>
  )
}

export const Route = createFileRoute('/about')({
  component: About,
})
