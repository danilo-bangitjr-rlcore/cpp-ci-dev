import { createFileRoute, useParams } from '@tanstack/react-router';

export const Route = createFileRoute('/agents/$config-name/monitor')({
  component: RouteComponent,
});

function RouteComponent() {
  const params = useParams({ from: '/agents/$config-name/monitor' });
  const configName = params['config-name'];

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <div className="bg-white shadow-lg rounded-lg overflow-hidden">
        <div className="px-6 py-8 text-gray-600">
          <h2 className="text-xl mb-2">
            Monitoring for {configName} coming soon!
          </h2>
        </div>
      </div>
    </div>
  );
}
