import { createLazyFileRoute } from '@tanstack/react-router'
import { useQuery } from '@tanstack/react-query'
import { getServerOrigin, fetchWithTimeout } from '../utils/api'

interface HealthPayload { status: string, time: string }

const About = () => {
  const { status, error, data, refetch } = useQuery({
    queryKey: ['health'],
    queryFn: async (): Promise<HealthPayload> => {
      const response = await fetchWithTimeout(
        `${getServerOrigin()}/health`,
      )
      const resp: HealthPayload = await response.json() as HealthPayload
      return resp
    }
  })

  return <div className="p-2">
    <h2 className="text-2xl">About</h2>
    <p>About page. TBD: show core-rl version information, web client details, experiment/agent details.</p>
    <div className='p-2'>
      <h3 className="text-xl">Health</h3>
      <p>{status}</p>
      {error && <p>Error: {error.message}</p>}
      {data && <pre>{JSON.stringify(data, null, 2)}</pre>}
      <button onClick={() => { void refetch() }}>Re-fetch Health Status</button>
    </div>
  </div>
}

export const Route = createLazyFileRoute('/about')({
  component: About,
})
