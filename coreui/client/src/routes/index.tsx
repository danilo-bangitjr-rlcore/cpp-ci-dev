import { createFileRoute } from '@tanstack/react-router'
import { ExampleComponent } from '../components/ExampleComponent'

export const Route = createFileRoute('/')({
  component: Index,
})

function Index() {
  return (
    <div className="p-2">
      <ExampleComponent title="Hello from the home page" />
    </div>
  )
}