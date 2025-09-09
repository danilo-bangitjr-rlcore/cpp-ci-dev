import { createRootRoute, Outlet } from '@tanstack/react-router';
import { TanStackRouterDevtools } from '@tanstack/react-router-devtools';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { GlobalHeader } from '../components/navigation/GlobalHeader';
import { LeftNav } from '../components/navigation/LeftNav';

const navItems = [
  {
    label: 'Home',
    to: '/',
    icon: <img src="/app/assets/corei.svg" style={{ maxWidth: '75%' }} />,
  },
  { label: 'Agents Overview', to: '/agents-overview' },
  { label: 'OPC Navigation', to: '/opc-navigation' },
  { label: 'About', to: '/about' },
  { label: 'Tags', to: '/tags' },
];

const headerItems = [
  { label: 'Settings', onClick: () => console.log('Settings clicked') },
];

const queryClient = new QueryClient();

export const Route = createRootRoute({
  component: () => (
    <QueryClientProvider client={queryClient}>
      <div className="min-h-screen bg-white">
        <GlobalHeader items={headerItems} />
        <div className="flex">
          <LeftNav items={navItems} />
          <main className="flex-1 p-4">
            <Outlet />
          </main>
        </div>
        <TanStackRouterDevtools />
      </div>
    </QueryClientProvider>
  ),
});
