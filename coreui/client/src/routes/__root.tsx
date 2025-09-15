import { createRootRoute, Outlet } from '@tanstack/react-router';
import { TanStackRouterDevtools } from '@tanstack/react-router-devtools';
import { GlobalHeader } from '../components/navigation/GlobalHeader';
import { LeftNav } from '../components/navigation/LeftNav';

const navItems = [
  {
    label: 'Home',
    to: '/',
    icon: <img src="/app/assets/corei.svg" style={{ maxWidth: '75%' }} />,
  },
  { label: 'Agents Overview', to: '/agents' },
  { label: 'OPC Navigation', to: '/opc-navigation' },
  { label: 'About', to: '/about' },
  { label: 'Tags', to: '/tags' },
];

const headerItems = [
  { label: 'Settings', onClick: () => console.log('Settings clicked') },
];

export const Route = createRootRoute({
  component: () => (
    <div className="flex flex-col h-screen bg-white">
      <GlobalHeader items={headerItems} />
      <div className="flex flex-1 min-h-0">
        <LeftNav items={navItems} />
        {/* Make main a flex column so route components using h-full can stretch */}
        <main className="flex-1 p-4 overflow-auto flex flex-col min-h-0">
          <Outlet />
        </main>
      </div>
      <TanStackRouterDevtools />
    </div>
  ),
});
