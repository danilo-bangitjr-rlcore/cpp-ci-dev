import { useEffect, useState } from 'react'
import { API_ENDPOINTS, get } from '../../utils/api'

interface UseAgentExistsResult {
  exists: boolean | null // null means loading
  loading: boolean
  error: string | null
}

export function useAgentExists(agentName: string): UseAgentExistsResult {
  const [exists, setExists] = useState<boolean | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!agentName) {
      setExists(false)
      setLoading(false)
      return
    }

    const checkAgentExists = async () => {
      setLoading(true)
      setError(null)
      
      try {
        const response = await get(API_ENDPOINTS.configs.raw(agentName))
        
        if (response.status === 404) {
          setExists(false)
        } else if (response.ok) {
          setExists(true)
        } else {
          // Other error status codes
          setError(`Error checking agent: ${response.status} ${response.statusText}`)
          setExists(false)
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to check agent existence')
        setExists(false)
      } finally {
        setLoading(false)
      }
    }

    checkAgentExists()
  }, [agentName])

  return { exists, loading, error }
}
