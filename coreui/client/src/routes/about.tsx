import { createFileRoute } from '@tanstack/react-router'
import { ExampleComponent } from '../components/ExampleComponent'

export const Route = createFileRoute('/about')({
  component: About,
})

function About() {
  return (
    <div className="p-2">
      <ExampleComponent title="Hello from the about page!" />
    </div>
  )
}