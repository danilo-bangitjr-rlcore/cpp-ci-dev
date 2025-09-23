import { createFileRoute } from '@tanstack/react-router'
import { useRete } from 'rete-react-plugin'
import { createEditor } from '../components/reward-components/Editor'

export const Route = createFileRoute('/reward')({
  component: RouteComponent,
})

function RouteComponent() {
  const [ref, editor] = useRete(createEditor)
  return (
    <div className='RouteComponent'>
     <div ref={ref} className="rete" style={{ height: "100vh", width: "100vw" }}></div>
     </div>
  )
}
