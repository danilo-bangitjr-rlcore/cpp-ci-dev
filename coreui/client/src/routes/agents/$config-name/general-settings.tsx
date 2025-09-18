import { createFileRoute, useParams } from '@tanstack/react-router';
import GeneralSettings from '../../../components/config-components/general-config/GeneralSettings';

export const Route = createFileRoute('/agents/$config-name/general-settings')({
  component: RouteComponent,
});

function RouteComponent() {
  const params = useParams({ from: '/agents/$config-name/general-settings' });
  const configName = params['config-name'];
  console.log('Config Name:', configName); // Debugging line

  return (
    <div>
      <GeneralSettings />
    </div>
  );
}
